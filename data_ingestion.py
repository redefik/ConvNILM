import tensorflow as tf
import numpy as np
import math

class DataIngestor(tf.keras.utils.Sequence):
  """
  Ingests batches of (b_mains, b_appliance) pairs into a neural network,
  where b_mains is a sequence of W overall power consmuptions and b_appliance
  is the power consumption of an appliance at the midpoint W // 2.

  Attributes
    ----------
    mains: numpy.ndarray
        a sequence of overall power consumptions.
    appliances : numpy.ndarray
        a sequence of power consumptions of an appliance.
    window_size : int
        value of W.
    batch_size : int
        the number of pairs in a batch.
    shuffle: boolean
        whether data should be shuffled or not.
  """

  def __init__(self, mains, appliances,
               window_size, batch_size, shuffle=False):
    self.mains = mains
    self.appliances = appliances
    self.window_size = window_size
    self.batch_size = batch_size
    if self.window_size % 2 == 0:
      self.indices = np.arange(len(self.mains) - self.window_size)
    else:
      self.indices = np.arange(len(self.mains) - self.window_size - 1)
    self.shuffle = shuffle
  
  def __len__(self):
    """
    Returns the number of batches.
    """
    return math.ceil(len(self.indices) / self.batch_size)
  
  def __getitem__(self, idx):
    """
    Returns the batch with index idx.
    """
    mains_batch = []
    appliances_batch = []
    if idx == self.__len__() - 1:
      inds = self.indices[idx * self.batch_size:] # for data shuffling (if enabled)
    else:
      inds = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size] # for data shuffling (if enabled)
    for i in inds:
      main_sample = self.mains[i:i+self.window_size]
      appliance_sample = self.appliances[i+math.ceil(self.window_size/2)]
      mains_batch.append(main_sample)
      appliances_batch.append(appliance_sample)

    # Reshape is needed to make data compatible with the network input_shape.
    mains_batch_np = np.array(mains_batch)
    mains_batch_np = np.reshape(mains_batch_np, 
                                (mains_batch_np.shape[0],
                                 mains_batch_np.shape[1],
                                 1))
      
    return mains_batch_np, np.array(appliances_batch)

  def on_epoch_end(self):
    """
    If shuffling is enabled, it shuffles data indices on epoch end.
    """
    if self.shuffle:
      np.random.shuffle(self.indices)