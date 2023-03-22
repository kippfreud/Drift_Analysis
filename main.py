#from https://github.com/CYHSM/DeepInsight/blob/master/notebooks/static/ephys_example.ipynb mostly


import deepinsight

# Other imports
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()

train_path_in = f"{args.name}_train.h5" # This will be the processed HDF5 file
val_path_in = f"{args.name}_test.h5"

hdf5_file = h5py.File('data/' + train_path_in, mode='r')
wavelets = hdf5_file['inputs/wavelets']
frequencies = hdf5_file['inputs/fourier_frequencies']

fig, axes = plt.subplots(3,1, figsize=(18,10))

for idx, ax in enumerate(axes):
    this_channel = (wavelets[:,:,idx] - np.mean(wavelets[:,:,idx], axis=0)) / (np.std(wavelets[:,:,idx], axis=0))
    ax.matshow(this_channel[0:2000,:].transpose(), aspect='auto', vmin=-2, vmax=2, cmap='viridis')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_yticks(np.arange(0, len(frequencies)))
    ax.set_yticklabels(frequencies, fontsize=7)
axes[-1].set_xlabel('Time')
#fig.show()
#plt.show()

hdf5_file.close()

# Define loss functions and train model
loss_functions = {'position' : 'euclidean_loss',
                  #'direction' : 'cyclical_mae_rad',
                  #'speed' : 'mae'
                  }

if args.name[0:5] in ["E-200", "F-200", "G-200", "H-200", "I-200"]:
    loss_weights = {'position': 1,
                    #'speed': 20,
                    #'head_direction': 25,
                    #'direction': 200,
                    #'direction_delta': 25,
                    }
elif args.name[0:7] in ["Rat_112", "Rat_113"]:
    loss_weights = {'position': 1,
                    'speed': 20,
                    #'head_direction': 25,
                    'direction': 200,
                    #'direction_delta': 25,
                    }

deepinsight.train.run_from_path(train_path_in, val_path_in, loss_functions, loss_weights)

# Get loss and shuffled loss for influence plot, both is also stored back to HDF5 file
losses, output_predictions, indices, _ = deepinsight.analyse.get_model_loss(train_path_in, stepsize=100)
shuffled_losses = deepinsight.analyse.get_shuffled_model_loss(train_path_in, axis=1, stepsize=100)

# Plot influence across behaviours
#deepinsight.visualize.plot_residuals(train_path_in, losses=losses, shuffled_losses=shuffled_losses, frequency_spacing=2, output_names=['Position', 'Head Direction', 'Speed'])
plt.show()
exit(0)