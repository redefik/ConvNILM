# ConvNILM

Energy disaggregation, a.k.a. NILM (Non Intrusive Load Monitoring), is a techinque that aims
to infer the power consumption of an appliance given the overall power consumption of a
building.

Nowadays deep learning represents the standard de facto in this field of research.

This repository contains the Keras scripts written to perform the NILM task assigned
during an academic course. 
To solve the problem, a convolutional neural network was used. The network resembles
the one described by Zhang et al. (https://arxiv.org/abs/1612.09106).

The code is organized as follows:
* ```data_loading.py``` contains the function used to load the dataset (see below).
* ```data_understaning.py``` contains some utilty functions used to make an explorative analysis
   of the dataset.
* ```data_preprocessing.py``` contains the utilities needed to process data before
  injecting them in the neural network.
* ```data_ingestion.py``` contains the keras.Sequence class implemented to feed the neural
  network in an efficient manner.
* ```data_model.py``` contains the definition of the neural networks.
* ```metrics.py``` contains the implementation of the metric (Energy-Based F1 score) used to
  evaluate the model.
* ```model_training.py``` is the entry point for training.
* ```model_testing.py``` is the entry point for testing.
* ```TrainingNotebook.ipynb``` is the notebook used to train the models.
* ```TestingNotebook.ipynb``` can be used to test the models.
* ```Report.pdf``` is a brief report of the work (in Italian). 

The Drive directories ```dishwasher_model``` and ```fridge_model``` provide the trained models for future reuse.

Dishwasher model: https://drive.google.com/drive/folders/1v0bCfsVNiltqXi0U_fWMfTSu1pwK9x8M?usp=sharing

Fridge model: https://drive.google.com/drive/folders/1MFTWkrSeeACwT0xr5c75iJb-bLqoNyy-

The models were trained on Google Colaboratory.

## Dataset

The dataset contains the appliance-level power consumption and whole power usage of a
building. The readings, expressed in Watts, were recorded with a sampling frequency of 1 Hz. The measurements
of 2 types of appliances are available: fridge and dishwasher. The training set contains
power consumptions for the period from 2019-01-01 00:00:00 to 2019-03-14 23:59:59.
The training set is organized in three CSV files (main_train.csv, fridge_train.csv and
dishwasher_train.csv) available at:

https://drive.google.com/drive/folders/1bOuv7G0R6_hoFkEPb5SsxZp-7Tky0pZI?usp=sharing 

The test set, relative to the period
from 2019-03-15 00:00:00 to 2019-03-31 23:59:59 was not made available by the instructors.

## Test Instructions

To test a neural network execute the TestingNotebook Jupyter Notebook providing the path
of the model, the path of the main_test file and the path of the appliance_test file.
