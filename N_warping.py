import os

import math
import itertools
import numpy as np
from os import path

from affinewarp import SpikeData
from affinewarp import ShiftWarping, PiecewiseWarping
from affinewarp.crossval import heldout_transform

import pickle


# load dataset to be analysed
path_datasets_folder = "datasets"
# [neuron, trial, position]
dataset = np.load(os.path.join(path_datasets_folder, "dataset_NP46_2019-12-02_18-47-02.npy"))

trials = [int(x) for x in dataset[1]]
spike_positions = [x for x in dataset[2]]
spike_IDs = [int(x) for x in dataset[0]]

spike_IDs_set = list(set(spike_IDs))
print(spike_IDs_set)
spike_IDs_set.sort()
new_spike_IDs = range(len(spike_IDs_set))

for id in range(len(spike_IDs)):
    neuron = spike_IDs[id]
    new_index = spike_IDs_set.index(neuron)
    spike_IDs[id] = new_index

# create data object
data = SpikeData(
    trials,
    spike_positions,
    spike_IDs,
    tmin=0,  # start of trials
    tmax=150,  # end of trials
)

############### Hyperparameters ###############################################
nbins_grid = [50, 120]
iterations_grid = [50]
smooth_reg_grid = [3, 8]
warp_reg_grid = [0, 0.6]
l2_reg_grid = [1e-7, 1e-5]
max_lag_grid = [0.1, 0.25, 0.4]
print(iterations_grid)

"""
# shift warping
NBINS = int(120)    # Number of time bins per trial
SMOOTH_REG = 10.0   # Strength of roughness penalty
WARP_REG = 0.6      # Strength of penalty on warp magnitude, penalizing the warping functions based on their distance from the identity line
L2_REG = 0.0        # Strength of L2 penalty on template magnitude
# TODO: what does this do, why does it allow shifts to 800?
MAXLAG = 0.2        # Maximum amount of shift allowed.
ITERATIONS = 80
"""
# piecewise warping
NKNOTS_LINEAR = 0
NKNOTS_1 = 1
NKNOTS_2 = 2
NKNOTS_3 = 3
###############################################################################


for NBINS, ITERATIONS, SMOOTH_REG, WARP_REG, L2_REG, MAX_LAG in itertools.product(nbins_grid, iterations_grid, smooth_reg_grid, warp_reg_grid, l2_reg_grid, max_lag_grid):
    print(NBINS, ITERATIONS, SMOOTH_REG, WARP_REG, L2_REG, MAX_LAG)

    shift_model = ShiftWarping(
        maxlag=MAX_LAG,
        smoothness_reg_scale=SMOOTH_REG,
        warp_reg_scale=WARP_REG,
        l2_reg_scale=L2_REG,
    )

    linear_model = PiecewiseWarping(
        n_knots=NKNOTS_LINEAR,
        warp_reg_scale=WARP_REG,
        smoothness_reg_scale=SMOOTH_REG,
        l2_reg_scale=L2_REG,
    )

    pwise1_model = PiecewiseWarping(
        n_knots=NKNOTS_1,
        warp_reg_scale=WARP_REG,
        smoothness_reg_scale=SMOOTH_REG,
        l2_reg_scale=L2_REG,
    )

    pwise2_model = PiecewiseWarping(
        n_knots=NKNOTS_2,
        warp_reg_scale=WARP_REG,
        smoothness_reg_scale=SMOOTH_REG,
        l2_reg_scale=L2_REG,
    )

    pwise3_model = PiecewiseWarping(
        n_knots=NKNOTS_3,
        warp_reg_scale=WARP_REG,
        smoothness_reg_scale=SMOOTH_REG,
        l2_reg_scale=L2_REG,
    )

    models = [shift_model, linear_model, pwise1_model, pwise2_model]

    # iterate over models
    for model, label in zip(models, ('shift', 'linear', 'pwise-1', 'pwise-2')):

        # check whether file already exists
        saves_folder = "saves"
        model_folder = str(label)
        filename = "heldout_validated_alignments_" + str(label) + "_warpreg" + str(WARP_REG) + "_nbins" + str(NBINS) + "_iterations" + str(ITERATIONS) + "_l2reg" + str(L2_REG) + "_smoothreg" + str(SMOOTH_REG)
        if label.startswith("shift"):
            filename = filename + "_maxlag" + str(MAX_LAG)

        path_file = os.path.join(saves_folder, model_folder, filename)
        if path.exists(path_file):
            print("already computed, skip")
            continue

        # Fit and apply warping to held out neurons.
        validated_alignments = heldout_transform(
            model, data.bin_spikes(NBINS), data, iterations=ITERATIONS)

        # save validated alignments
        pickle_out = open(path_file, 'wb')
        pickle.dump(validated_alignments, pickle_out)
        pickle_out.close()


# run with: python N_warping.py [path to dataset] [optional: model, smooth reg, warp reg, l2 reg, max lag, n bins, n iterations]
