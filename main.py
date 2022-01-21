
import deepinsight

# Other imports
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np

fp_deepinsight = 'train.h5' # This will be the processed HDF5 file

hdf5_file = h5py.File(fp_deepinsight, mode='r')
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
                  'direction' : 'cyclical_mae_rad',
                  'speed' : 'mae'}
loss_weights = {'position' : 1,
                'direction' : 25,
                'speed' : 20}
deepinsight.train.run_from_path(fp_deepinsight, loss_functions, loss_weights)

# Get loss and shuffled loss for influence plot, both is also stored back to HDF5 file
losses, output_predictions, indices, _ = deepinsight.analyse.get_model_loss(fp_deepinsight, stepsize=100)
shuffled_losses = deepinsight.analyse.get_shuffled_model_loss(fp_deepinsight, axis=1, stepsize=100)

# Plot influence across behaviours
deepinsight.visualize.plot_residuals(fp_deepinsight, losses=losses, shuffled_losses=shuffled_losses, frequency_spacing=2, output_names=['Position', 'Head Direction', 'Speed'])
plt.show()
exit(0)