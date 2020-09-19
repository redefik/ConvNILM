import tensorflow as tf
from tensorflow.keras.regularizers import l2

def build_dishwasher_cnn(window_size, evaluation_metric=None):
  """
  Defines and compiles the convolutive neural network to be trained 
  to infer dishwasher's power consumption.

  Args:
    window_size (int): size of the sliding window used in the seq-to-point approach.
    evaluation_metric (keras.Metric): metric used to evaluate the model (optional) 
  Returns:
    tensorflow.keras.Model: network to be trained.
  """
  model = tf.keras.models.Sequential()
  weight_initializer = 'he_normal'

  model.add(tf.keras.layers.Conv1D(filters=30, kernel_size=10, padding='same',
                                   kernel_initializer=weight_initializer,
                                   input_shape=(window_size,1),
                                   activation='relu'))
  model.add(tf.keras.layers.Conv1D(filters=30, kernel_size=8, padding='same',
                                   kernel_initializer=weight_initializer,
                                   activation='relu'))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(units=256, kernel_initializer=weight_initializer,
                                  activation='relu'))
  model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

  optimizer = tf.keras.optimizers.Adam(epsilon=1e-8)

  if evaluation_metric is not None:
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[evaluation_metric])
  else:
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
  return model

def build_fridge_cnn(window_size, evaluation_metric=None):
  """
  Defines and compiles the convolutive neural network to be trained 
  to infer fridge's power consumption.

  Args:
    window_size (int): size of the sliding window used in the seq-to-point approach.
    evaluation_metric (keras.Metric): metric used to evaluate the model (optional) 
  Returns:
    tensorflow.keras.Model: network to be trained.
  """
  model = tf.keras.models.Sequential()
  weight_initializer = 'he_normal'
  reg_factor = 0.0001

  model.add(tf.keras.layers.Conv1D(filters=15, kernel_size=10, padding='same',
                                   kernel_initializer=weight_initializer, activation='relu',
                                   kernel_regularizer=l2(reg_factor), bias_regularizer=l2(reg_factor),
                                   input_shape=(window_size, 1)))
  model.add(tf.keras.layers.Conv1D(filters=15, kernel_size=8, padding='same',
                                   kernel_initializer=weight_initializer, activation='relu',
                                   kernel_regularizer=l2(reg_factor), bias_regularizer=l2(reg_factor)))
  model.add(tf.keras.layers.Conv1D(filters=20, kernel_size=6, padding='same',
                                   kernel_initializer=weight_initializer, activation='relu',
                                   kernel_regularizer=l2(reg_factor), bias_regularizer=l2(reg_factor)))
  model.add(tf.keras.layers.Conv1D(filters=25, kernel_size=5, padding='same',
                                   kernel_initializer=weight_initializer, activation='relu',
                                   kernel_regularizer=l2(reg_factor), bias_regularizer=l2(reg_factor)))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(units=512, kernel_initializer=weight_initializer,
                                  kernel_regularizer=l2(reg_factor), bias_regularizer=l2(reg_factor)))
  model.add(tf.keras.layers.Dropout(0.05))
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dense(units=1, activation='linear'))
  
  optimizer = tf.keras.optimizers.Adam(epsilon=1e-8, lr=0.0001)

  if evaluation_metric is not None:
    model.compile(optimizer=optimizer, loss='mae', metrics=[evaluation_metric])
  else:
    model.compile(optimizer=optimizer, loss='mae')
  
  return model