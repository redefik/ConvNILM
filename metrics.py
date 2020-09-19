import tensorflow as tf

class EnergyBasedF1(tf.keras.metrics.Metric):
  """
  This class implements the Energy-based F1 score - a common metric used to evaluate NILM models.
  (See for example 'Denoising Autoencoders for Non-Intrusive Load Monitoring:
  Improvements and Comparative Evaluation (Bonfigli et al.)')
  """
  def __init__(self, rescaling=None, min_value=0.0, max_value=1.0, 
               mean_value=0.0, std_value=1.0, name='energy_based_f1', **kwargs):
    super(EnergyBasedF1, self).__init__(name=name, **kwargs)
    self.pr_num = self.add_weight(name='pr_num', 
                                        initializer='zeros')
    self.p_den = self.add_weight(name='p_den', initializer='zeros')
    self.r_den = self.add_weight(name='r_den', initializer='zeros')
    self.rescaling = rescaling
    self.min_value = min_value
    self.max_value = max_value
    self.mean_value = mean_value
    self.std_value = std_value
    
  def update_state(self, y_true, y_pred, sample_weight=None):
    """
    Updates the metric's state on iteration end.
    """
    # Recover the original scale of data if needed.
    if self.rescaling == 'standardize':
      y_pred_rescaled = y_pred * (self.std_value)
      y_pred_rescaled += self.mean_value
      y_true_rescaled = y_true * (self.std_value)
      y_true_rescaled += self.mean_value
    elif self.rescaling == 'normalize':
      y_pred_rescaled = y_pred * (self.max_value - self.min_value)
      y_pred_rescaled += self.min_value
      y_true_rescaled = y_true * (self.max_value - self.min_value)
      y_true_rescaled += self.min_value
    else:
      y_pred_rescaled = y_pred
      y_true_rescaled = y_true
    # Squash to 0 negative values if there are any
    comparison = tf.less(y_pred_rescaled, tf.constant(0.0))
    y_pred_positive = tf.where(comparison, tf.zeros_like(y_pred_rescaled), y_pred_rescaled)

    self.p_den.assign_add(tf.math.reduce_sum(y_pred_positive))
    self.r_den.assign_add(tf.math.reduce_sum(y_true_rescaled))
    self.pr_num.assign_add(tf.math.reduce_sum(tf.math.minimum(y_pred_positive,y_true_rescaled)))
    
  def result(self):
    """
    Returns the value of the metric on iteration end.
    """
    epsilon = 1e-8
    precision = tf.math.divide(self.pr_num, self.p_den + epsilon)
    recall = tf.math.divide(self.pr_num, self.r_den + epsilon)
    num = 2 * tf.math.multiply(precision, recall)
    den = tf.math.add(precision, recall)
    return tf.math.divide(num, den)

  def reset_states(self):
    """
    Resets the metric's state on epoch end.
    """
    self.pr_num.assign(0.)
    self.p_den.assign(0.)
    self.r_den.assign(0.)

def compute_F1_score(predicted_values, ground_truth):
  """
  Computes the Energy-Based F1 score given the predicted values and ground truth.
  Args:
    predicted_values (ndarray)
    ground_truth (ndarray)
  Returns:
    ndarray: F1 score
  """
  pr_num = 0.0
  p_den = 0.0
  r_den = 0.0
  epsilon = 1e-8
  for i in range(len(ground_truth)):  
    p_den += predicted_values[i]
    r_den += ground_truth[i]
    pr_num += min(predicted_values[i], ground_truth[i])
  precision = pr_num / (p_den + epsilon)
  recall = pr_num / (r_den + epsilon)
  f1 = 2 * precision * recall / (precision + recall)
  return f1