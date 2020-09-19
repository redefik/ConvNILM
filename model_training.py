import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import data_loading
import data_preprocessing
import data_ingestion
import metrics
import math

def train_model(appliance_name, main_path, appliance_path,  
                window_size, batch_size, build_model_func,  
                epochs, patience=None, train_end_timestamp=None, 
                early_stopping=False, rescaling=None,
                split=False, plot_model=False):
  """
  Trains a model to infer an appliance's power consumption from the overall power
  consumption.

  Args:
    appliance_name (string): name of the appliance.
    main_path (string): path of the csv file containing training data
                        about overall power consumption.
    appliance_path (string): path of the csv file containing training
                             data about an appliance's power consumption.
    train_end_timestamp (string): final timestamp of training data. Ignored if
                                  split is False.
    window_size (int): size of the sliding window used to process power
                                 consumption's data.
    batch_size (int): number of samples in a batch.
    build_model_func (function): a function that returns the model to be trained.
    epochs (int): training epochs. If early stopping is True, this is a maximum
                  number of epochs.
    patience (int): Number of epochs with no improvement after which training will be stopped.
                    Ignored if early_stopping is False.
    early_stopping (boolean): indicates whether or not to do early stopping.
                              Ignored if split is False.                          
    rescaling (string): a string ('normalize' or 'standardize') that indicates
                        rescaling strategy. If None, no rescaling is applied.
    split (boolean): indicates whether or not to split the dataset in training set
                     and validation set.
    plot_model (boolean): indicates whether or not to plot the trained model.
                          If True the model is saved to the file {app}_model.png
                          where {app} is appliance_name.
  Returns:
    tensorflow.keras.Model: the trained model.
  
  """
  main_val = None
  appliance_val = None
  val_ingestor = None
  val_steps = None
  # Data Loading
  print('Data Loading...', end='')
  appliance_power = data_loading.read_csv_data(appliance_path)
  main_power = data_loading.read_csv_data(main_path)
  print('Done.')
  # Data splitting (if required)
  if split:
    print('Data splitting...', end='')
    appliance_train, appliance_val = data_preprocessing.train_test_split(appliance_power,
                                                                        train_end_timestamp)
    main_train, main_val = data_preprocessing.train_test_split(main_power,
                                                              train_end_timestamp)
    print('Done.')
  else:
    appliance_train = appliance_power.values
    main_train = main_power.values
  # Compute statistics on training data and log them
  print('Statistics of interest:')
  main_min_power = np.min(main_train)
  main_max_power = np.max(main_train)
  main_mean_power = np.mean(main_train)
  main_std_power = np.std(main_train)
  appliance_min_power = np.min(appliance_train)
  appliance_max_power = np.max(appliance_train)
  appliance_mean_power = np.mean(appliance_train)
  appliance_std_power = np.std(appliance_train)

  print('Overall min power: {}, Overall max power: {}'.format(main_min_power,
                                                              main_max_power))
  print('Overall mean power: {}, Overall std power: {}'.format(main_mean_power,
                                                               main_std_power))
  print('Min {} power: {}, Max {} power: {}'.format(appliance_name,
                                                    appliance_min_power,
                                                    appliance_name,
                                                    appliance_max_power))
  print('Mean {} power: {}, Std {} power: {}'.format(appliance_name,
                                                     appliance_mean_power,
                                                     appliance_name,
                                                     appliance_std_power))
  # Zero-pad the original sequences to perform seq2point learning
  print('Zero padding...', end='')
  main_train = data_preprocessing.zero_pad(main_train, window_size)
  appliance_train = data_preprocessing.zero_pad(appliance_train, window_size)
  if split:
    main_val = data_preprocessing.zero_pad(main_val, window_size)
    appliance_val = data_preprocessing.zero_pad(appliance_val, window_size)
  print('Done.')

  # Rescaling is applied if required
  if rescaling is not None:
    print('Rescaling...', end='')
    if rescaling == 'standardize':
      main_train = data_preprocessing.standardize_data(main_train, main_mean_power,
                                                      main_std_power)
      appliance_train = data_preprocessing.standardize_data(appliance_train, appliance_mean_power,
                                                            appliance_std_power)
      if split:
        main_val = data_preprocessing.standardize_data(main_val, main_mean_power,
                                                       main_std_power)
        appliance_val = data_preprocessing.standardize_data(appliance_val, appliance_mean_power,
                                                            appliance_std_power)
    if rescaling == 'normalize':
      main_train = data_preprocessing.normalize_data(main_train, main_min_power,
                                                    main_max_power)
      appliance_train = data_preprocessing.normalize_data(appliance_train, appliance_min_power,
                                                          appliance_max_power)
      if split:
        main_val = data_preprocessing.normalize_data(main_val, main_min_power,
                                                     main_max_power)
        appliance_val = data_preprocessing.normalize_data(appliance_val, appliance_min_power,
                                                          appliance_max_power)
    print('Done.')

  # Data Ingestion: create generators to feed the model
  print('Preparing data ingestion...', end='')
  train_ingestor = data_ingestion.DataIngestor(main_train, appliance_train,
                                               window_size, batch_size, shuffle=True)
  if split:
    val_ingestor = data_ingestion.DataIngestor(main_val, appliance_val,
                                               window_size, batch_size)
  print('Done.')
  # Model definition
  print('Building model...', end='')
  if split:
    if rescaling == 'standardize':
      energy_based_f1_score = metrics.EnergyBasedF1(rescaling='standardize', 
                                            mean_value=appliance_mean_power,
                                            std_value=appliance_std_power)
    elif rescaling == 'normalize':
      energy_based_f1_score = metrics.EnergyBasedF1(rescaling='normalize',
                                            min_value=appliance_min_power,
                                            max_value=appliance_max_power)
    else:
      energy_based_f1_score = metrics.EnergyBasedF1()

    model = build_model_func(window_size, evaluation_metric=energy_based_f1_score)
  else:
    model = build_model_func(window_size)
  print('Done.')
  # Plot model
  if plot_model:
    model_plot_file = '{}_model.png'.format(appliance_name)
    tf.keras.utils.plot_model(model, to_file=model_plot_file, show_shapes=True, 
                            show_layer_names=False, rankdir='LR')
  # Training
  train_callbacks = []
  if split and early_stopping:
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_energy_based_f1',
                                                      mode='max',
                                                      patience=patience, verbose=1)
    train_callbacks.append(early_stopping)
  train_steps = train_ingestor.__len__()
  if split:
    val_steps = val_ingestor.__len__()
  print('Model training...')
  if split:  
    history = model.fit(x=train_ingestor, epochs=epochs, steps_per_epoch=train_steps,
                        validation_data=val_ingestor, validation_steps=val_steps,
                        callbacks=train_callbacks)
  else:
    history = model.fit(x=train_ingestor, epochs=epochs, steps_per_epoch=train_steps)
  print('Training completed.')
  # Plot learning curves
  history_dict = history.history
  plt.title('Loss during training')
  plt.plot(np.arange(1, len(history.epoch) + 1), history_dict['loss'], marker='o')
  if split:
    plt.plot(np.arange(1, len(history.epoch) + 1), history_dict['val_loss'], marker='o')
    plt.legend(['train', 'val'])
    plt.show()
    plt.title('F1 during training')
    plt.plot(np.arange(1, len(history.epoch) + 1), history_dict['energy_based_f1'], marker='o')
    plt.plot(np.arange(1, len(history.epoch) + 1), history_dict['val_energy_based_f1'], marker='o')
    plt.legend(['train', 'val'])
  plt.show()

  return model


    
  

    