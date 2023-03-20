import deepinsight
import numpy as np
from scipy.io import loadmat
import h5py
import os
import argparse
from deepinsight import util
import pandas as pd
import gc
import tensorflow as tf
import matplotlib.pyplot as plt

gc.enable()
MODEL_PATH = "final_models"
N = 1000

# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--br', type=str, required=True)
args = parser.parse_args()

RAT_NAME = args.name

files = os.listdir('data/')
files = [f for f in files if f[0] == RAT_NAME]
files = [f for f in files if "test" in f]

if args.br == "all":
    files = [f for f in files if "PC" not in f]
    files = [f for f in files if "PFC" not in f]
else:
    files = [f for f in files if "-"+args.br in f]

error_data = {}
shuffle_error_data = {}

allmses = []
allshufflemses = []
print(f"Starting for {args.br}...")

for train_file in files:

    train_date = train_file[2:12]
    error_data[train_date] = {}
    shuffle_error_data[train_date] = {}

    for test_file in files:

        tf.keras.backend.clear_session()

        test_date = test_file[2:12]

        (model, training_generator_og, testing_generator, opts) = util.hdf5.load_model_with_opts(MODEL_PATH + '/' + train_file[:-8] + "_train_model_4.h5")
        training_generator_og.shuffle = False
        training_generator_og.random_batches = False

        if test_date != train_date:
            hdf5_in = test_file
        else:
            hdf5_in = test_file[:-8] + ".h5"
        #hdf5_in = 'frey.h5'
        val_hdf5_file = h5py.File('data/'+hdf5_in, mode='r')
        val_t_w = val_hdf5_file['inputs/wavelets']
        tmp_opts = util.opts.get_opts(hdf5_in, train_test_times=(np.array([]), np.array([])))
        exp_indices = np.arange(0, val_t_w.shape[0] - (tmp_opts['model_timesteps'] * tmp_opts['batch_size']))
        cv_splits = np.array_split(exp_indices, 5)
        cv_run = 2
        cvs = cv_splits[cv_run]
        training_indices = np.setdiff1d(exp_indices, cvs)  # All except the test indices
        testing_indices = cvs
        loss_functions = {'position' : 'euclidean_loss',
                          'head_direction' : 'cyclical_mae_rad',
                          'speed' : 'mae'}
        loss_weights = {'position' : 1,
                        'head_direction' : 250,
                        'speed' : 20}
        # opts = util.opts.get_opts(hdf5_in, train_test_times=(training_indices, testing_indices))
        opts = util.opts.get_opts(hdf5_in, train_test_times=(exp_indices, np.array([0])))
        opts['loss_functions'] = loss_functions.copy()
        opts['loss_weights'] = loss_weights
        opts['loss_names'] = list(loss_functions.keys())
        opts['num_cvs'] = 5
        opts['shuffle'] = True
        opts['random_batches'] = True
        opts['batch_size'] = 8
        (training_generator, testing_generator) = util.data_generator.create_train_and_test_generators(opts)

        (shuf_training_generator, _) = util.data_generator.create_shuffle_train_and_test_generators(opts)



        ERROR = []
        SPEEDS = []

        n = 1
        for inp, true_out in training_generator:
            pred_out = model.predict(inp)
            true_loc_batch = true_out[0]
            for i in range(true_loc_batch.shape[0]):
                true_loc = true_loc_batch[i]
                pred_loc = pred_out[0][i]

                true_dir = true_out[1][0][-1][0]
                pred_dir = pred_out[1][0][-1][0]

                true_spd = true_out[2][0][-1][0]
                pred_spd = pred_out[2][0][-1][0]

                index = 0
                true_loc = true_loc[-1]

                pred_loc = pred_loc[-1]
                # pred_loc = np.array([np.random.uniform(266,705), np.random.uniform(59,462)])
                #pred_loc = np.array([np.random.uniform(0, 720), np.random.uniform(0, 576)])
                #pred_loc = np.array(np.mean(training_generator_og.outputs[0], axis=0))

                xer = (true_loc[0] - pred_loc[0]) * (1.7 / 720.)
                yer = (true_loc[1] - pred_loc[1]) * (1.3 / 576.)
                err = (xer ** 2 + yer ** 2) ** 0.5
                ERROR.append(err)
                SPEEDS.append(true_spd)

                n += 1
            if n > N:
                 break
        ERROR = [max(e, 2) for e in ERROR]
        mse = np.mean(ERROR)
        plt.hist(ERROR)
        plt.title(
            f"Rat {RAT_NAME} Train: {train_date} Test: {test_date} (mean = {round(np.mean(ERROR), 1)} max = {round(max(ERROR), 1)}) ")
        plt.savefig(f"hists/error_hist_figs/{RAT_NAME}-Tr{train_date}-Te{test_date}.png")
        plt.clf()
        allmses.append(mse)
        error_data[train_date][test_date] = mse



        (model, training_generator_og, testing_generator, opts) = util.hdf5.load_model_with_opts(f"{MODEL_PATH}/SHUFFLE-" + train_file[:-8] + "_train_model_4.h5")
        shERROR = []
        shSPEEDS = []
        n = 1
        for inp, true_out in shuf_training_generator:
            pred_out = model.predict(inp)
            true_loc_batch = true_out[0]
            for i in range(true_loc_batch.shape[0]):
                true_loc = true_loc_batch[i]
                pred_loc = pred_out[0][i]

                true_dir = true_out[1][0][-1][0]
                pred_dir = pred_out[1][0][-1][0]

                true_spd = true_out[2][0][-1][0]
                pred_spd = pred_out[2][0][-1][0]

                index = 0
                true_loc = true_loc[-1]

                pred_loc = pred_loc[-1]
                # pred_loc = np.array([np.random.uniform(266,705), np.random.uniform(59,462)])
                # pred_loc = np.array([np.random.uniform(0, 720), np.random.uniform(0, 576)])
                # pred_loc = np.array(np.mean(training_generator_og.outputs[0], axis=0))

                xer = (true_loc[0] - pred_loc[0]) * (1.7 / 720.)
                yer = (true_loc[1] - pred_loc[1]) * (1.3 / 576.)
                err = (xer ** 2 + yer ** 2) ** 0.5
                shERROR.append(err)
                shSPEEDS.append(true_spd)

                n += 1
            if n > N:
                break
        shERROR = [max(e, 2) for e in shERROR]
        mse_shuffle = np.mean(shERROR)
        plt.hist(shERROR)
        plt.title(
            f"Rat {RAT_NAME} Train: {train_date} Test: {test_date} (mean = {round(np.mean(shERROR), 1)} max = {round(max(shERROR), 1)}) ")
        plt.savefig(f"hists/shuffle_error_hist_figs/{RAT_NAME}-Tr{train_date}-Te{test_date}.png")
        plt.clf()
        allshufflemses.append(mse_shuffle)
        shuffle_error_data[train_date][test_date] = mse_shuffle

        plt.hist(ERROR/mse_shuffle)
        plt.title(
            f"Rat {RAT_NAME} Train: {train_date} Test: {test_date} (mean = {round(np.mean(ERROR/mse_shuffle), 1)} max = {round(max(ERROR/mse_shuffle), 1)}) ")
        plt.savefig(f"hists/normed_error_histograms/{RAT_NAME}-Tr{train_date}-Te{test_date}.png")
        plt.clf()
        allmses.append(mse)
        error_data[train_date][test_date] = mse

        gc.collect()

df = pd.DataFrame(error_data)
output_file = "csvs/" + RAT_NAME + "_" + args.br + ".csv"
df.to_csv(output_file)