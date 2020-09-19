import pandas as pd
import matplotlib.pyplot as plt

def data_explore(power_series):
  """
  Conducts an explorative analysis of the provided time series.

  Args:
    power_series (pandas.core.series.Series): power consumption time series
  Returns:
    None
  """
  print('------------------')
  print('First elements')
  print('------------------')
  print(power_series.head())
  print('------------------')
  print('Summary Statistics')
  print('------------------')
  print(power_series.describe())
  duplicated_data = len(power_series.to_frame().reset_index()) - len(power_series.to_frame().reset_index().drop_duplicates())
  print('------------------')
  print('Duplicated data: {}'.format(duplicated_data))
  print('------------------')
  na_data = len(power_series)-len(power_series.dropna())
  print('------------------')
  print('Missing values: {}'.format(na_data))
  print('------------------')

def plot_interval(power_series, start_time, end_time, title):
  """
  Plots the values of power consumption during a given time interval.

  Args:
    power_series (pandas.core.series.Series): power consumption time series.
    start_time (string): initial timestamp of the plotting interval.
    end_time (string): final timestamp of the plotting interval.
    title (string): title of the plot.

  Returns:
    None
  """
  plt.figure() # Create a new plot
  data_to_plot = power_series[start_time:end_time]
  data_to_plot.plot(figsize=(15,10), ylim=(0.0, 8000.0), xticks=[], use_index=False, title=title)
