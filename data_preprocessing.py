import numpy as np
from math import ceil

def train_test_split(time_series, train_end_timestamp):
  """
  Splits a given time series in training data and test data.

  Args:
    time_series (pandas.core.series.Series): input time series.
    train_end_timestamp (string): final timestamp of training data.
  Returns:
    numpy.ndarray: training time series
    numpy.ndarray: test time series
  """
  train = time_series[:train_end_timestamp].values
  test = time_series[train_end_timestamp:].values

  return train, test

def train_val_test_split(time_series, train_start_timestamp, train_end_timestamp, val_end_timestamp):
  """
  Splits a given time series in training data, validation data and test data.

  Args:
    time_series (pandas.core.series.Series): input time series.
    train_start_timestamp (string): initial timestamp of training data
    train_end_timestamp (string): final timestamp of training data
    val_end_timestamp (string): final timestamp of validation data
  Returns:
    numpy.ndarray: training time series
    numpy.ndarray: validation time series
    numpy.ndarray: test time series
  """
  train = time_series[train_start_timestamp:train_end_timestamp].values
  val = time_series[train_end_timestamp:val_end_timestamp].values
  test = time_series[val_end_timestamp:].values

  return train, val, test

def normalize_data(data, min_value=0.0, max_value=1.0):
  """
  Rescales the input subtracting min_power and dividing by max_value - min_value.

  Args:
    data (numpy.ndarray): data to be normalized.
    min_value (float)
    max_value (float)
  Returns:
    numpy.ndarray: normalized data.
  """
  data -= min_value
  data /= max_value - min_value
  return data

def standardize_data(data, mu=0.0, sigma=1.0):
  """
  Rescales the input subtracting mu and dividing by sigma.

  Args:
    data (numpy.ndarray): data to be standardized.
    mu (float)
    sigma (float)
  Returns:
    numpy.ndarray: standardized data
  """
  data -= mu
  data /= sigma
  return data

def zero_pad(data, window_size):
  """
  Pads with window_size / 2 zeros the given input.

  Args:
    data (numpy.ndarray): data to be padded.
    window_size (int): parameter that controls the size of padding.

  Returns:
    numpy.ndarray: padded data.
  """
  pad_width = ceil(window_size / 2)
  padded = np.pad(data, (pad_width, pad_width), 'constant', constant_values=(0,0))
  return padded
