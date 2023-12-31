import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
from tqdm import tqdm
from IPython.display import clear_output 
from data_prepare import *
from Network_structure import *
from loss_function import *
from train_method import *
from save_method import *
import sys
import os
#sys.path.append('../')
# from Novel_CNN import *

# EEGdenoiseNet V2
# Author: Haoming Zhang 
# Here is the main part of the denoising neurl network, We can adjust all the parameter in the user-defined area.
##################################################### User-defined ########################################################

epochs = 60    # training epoch
batch_size  = 40    # training batch size
sampling_rate = 256 # Sampling Rate
num_seconds = 2 # Number of seconds
combin_num = 10    # combin EEG and noise ? timdenoise_network = 'Simple_CNN'    # fcNN & Simple_CNN & Complex_CNN & RNN_lstm  & Novel_CNN 
denoise_network = 'fcNN'    # fcNN & Simple_CNN & Complex_CNN & RNN_lstm  & Novel_CNN 
noise_type = 'tRNS' # Type of noise to be applied to EEG recordings
optimizer_name = "Adam"
case_file = "1"
execution_run = 1 # We run each NN for 10 times to increase  the  statistical  power  of  our  results

result_location = r'../../results/'     #  Where to export network results   ############ change it to your own location #########
foldername = f'EEGdenoiseNet-{denoise_network}_{noise_type}_e{epochs}_b{batch_size}_o{optimizer_name}_{case_file}'            # the name of the target folder (should be change when we want to train a new network)
os.environ['CUDA_VISIBLE_DEVICES']='0'
save_train = True
save_vali = True
save_test = True

################################################## optimizer adjust parameter  ####################################################
#rmsp=tf.optimizers.RMSprop(learning_rate=0.00005, rho=0.9)
adam=tf.optimizers.Adam(learning_rate=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
#sgd=tf.optimizers.SGD(learning_rate=0.0002, momentum=0.9, decay=0.0, nesterov=False)
###################################################################################################################################

if optimizer_name == "Adam":
  optimizer = adam
# elif optimizer_name == "RMSP":
#   optimizer = rmsp
# elif optimizer_name == "SGD":
#   optimizer = sgd

if noise_type in ['EOG', 'tDCS', 'tRNS']:
  datanum = sampling_rate * num_seconds
elif noise_type == 'EMG':
  datanum = sampling_rate * num_seconds * 2 # At least for the EEGdenoiseNet benchmarks (i.e. 2s segments)

# We have reserved an example of importing an existing network
'''
path = os.path.join(result_location, foldername, "denoised_model")
denoiseNN = tf.keras.models.load_model(path)
'''
#################################################### Import data #####################################################

file_location = '../../data/'                    ############ change it to your own location #########
EEG_all = np.load( file_location + 'EEG_all_epochs.npy')                              
if noise_type == 'EOG':
  noise_all = np.load( file_location + 'EOG_all_epochs.npy') 
elif noise_type == 'EMG':
  noise_all = np.load( file_location + 'EMG_all_epochs.npy') 
elif noise_type == 'tDCS':
  noise_tDCS_folder = 'tDCS_all_epochs-1.5mA.npy'
  noise_all = np.load(file_location + noise_tDCS_folder)
elif noise_type == 'tRNS':
  noise_tRNS_folder = 'tRNS_all_epochs-1.5mA.npy' 
  noise_all = np.load(file_location + noise_tRNS_folder)

############################################################# Running #############################################################
#for i in range(10):
noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(EEG_all = EEG_all, noise_all = noise_all, combin_num = 10, train_per = 0.8, noise_type = noise_type)

if denoise_network == 'fcNN':
  model = fcNN(datanum)

elif denoise_network == 'Simple_CNN':
  model = simple_CNN(datanum)

elif denoise_network == 'Complex_CNN':
  model = Complex_CNN(datanum)

elif denoise_network == 'RNN_lstm':
  model = RNN_lstm(datanum)

elif denoise_network == 'Novel_CNN':
  model = Novel_CNN(datanum)
  
else: 
  print('NN name arror')


saved_model, history = train(model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, 
                      epochs, batch_size,optimizer, denoise_network, 
                      result_location, foldername , train_num = str(execution_run))

#denoised_test, test_mse = test_step(saved_model, noiseEEG_test, EEG_test)

# save signal
save_eeg(saved_model, result_location, foldername, save_train, save_vali, save_test, 
                    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, 
                    train_num = str(execution_run))
np.save(result_location +'/'+ foldername + '/'+ str(execution_run)  +'/'+ "nn_output" + '/'+ 'loss_history.npy', history)

#save model
# path = os.path.join(result_location, foldername, str(i+1), "denoise_model")
# tf.keras.models.save_model(saved_model, path)