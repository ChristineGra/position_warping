import os

import math
import numpy as np
import matplotlib.pyplot as plt

from affinewarp import SpikeData
from affinewarp import ShiftWarping, PiecewiseWarping
from affinewarp.crossval import heldout_transform


# load dataset to be analysed
path_datasets_folder = "datasets"
# [spike position, trial number, spikeID]
dataset = np.load(os.path.join(path_datasets_folder, "dataset_NP46_2019-12-02_18-47-02.npy"))
trials = [int(x) for x in dataset[1]]
spike_positions = [int(x) for x in dataset[2]]
spike_IDs = [int(x) for x in dataset[0]]

spike_IDs_set = list(set(spike_IDs))
print(spike_IDs_set)
spike_IDs_set.sort()
new_spike_IDs = range(len(spike_IDs_set))

for id in range(len(spike_IDs)):
    neuron = spike_IDs[id]
    new_index = spike_IDs_set.index(neuron)
    spike_IDs[id] = new_index
    
print(set(spike_IDs))


# create data object
data = SpikeData(
    trials,
    spike_positions,
    spike_IDs,
    tmin=50,  # start of trials
    tmax=255,  # end of trials
)

############### Hyperparameters ###############################################
# shift warping
NBINS = int(200)    # Number of time bins per trial
SMOOTH_REG = 10.0   # Strength of roughness penalty
WARP_REG = 0.0      # Strength of penalty on warp magnitude
L2_REG = 0.0        # Strength of L2 penalty on template magnitude
# TODO: what does this do, why does it allow shifts to 800?
MAXLAG = 0.2        # Maximum amount of shift allowed.

# piecewise warping
NKNOTS_LINEAR = 0
NKNOTS_1 = 1
NKNOTS_2 = 2
NKNOTS_3 = 3
WARP_REG_SCALE = 0
SMOOTHNESS_REG_SCALE = 10.0
###############################################################################
def plot_column(axes, spike_data):
    raster_kws = dict(s=4, c='k', lw=0)
    for n, ax in zip(neurons_to_plot, axes):
        ax.scatter(
            spike_data.spiketimes[spike_data.neurons == n],
            spike_data.trials[spike_data.neurons == n],
            **raster_kws,
        )
        ax.set_xlabel("position")
        ax.set_ylabel("trial")


shift_model = ShiftWarping(
    maxlag=MAXLAG,
    smoothness_reg_scale=SMOOTH_REG,
    warp_reg_scale=WARP_REG,
    l2_reg_scale=L2_REG,
)

linear_model = PiecewiseWarping(
    n_knots=NKNOTS_LINEAR,
    warp_reg_scale=WARP_REG_SCALE,
    smoothness_reg_scale=SMOOTHNESS_REG_SCALE,
)

pwise1_model = PiecewiseWarping(
    n_knots=NKNOTS_1,
    warp_reg_scale=WARP_REG_SCALE,
    smoothness_reg_scale=SMOOTHNESS_REG_SCALE,
)

pwise2_model = PiecewiseWarping(
    n_knots=NKNOTS_2,
    warp_reg_scale=WARP_REG_SCALE,
    smoothness_reg_scale=SMOOTHNESS_REG_SCALE,
)

pwise3_model = PiecewiseWarping(
    n_knots=NKNOTS_3,
    warp_reg_scale=WARP_REG_SCALE,
    smoothness_reg_scale=SMOOTHNESS_REG_SCALE,
)

models = [shift_model, linear_model, pwise1_model, pwise2_model, pwise3_model]

neurons_to_plot = [i for i in new_spike_IDs if spike_IDs.count(i) > 5]
print("neurons to plot: ", neurons_to_plot)

for model, label in zip(models, ('shift', 'linear', 'pwise-1', 'pwise-2', 'pwise-3')):

    # Fit and apply warping to held out neurons.
    validated_alignments = heldout_transform(
        model, data.bin_spikes(NBINS), data, iterations=80)

    # NOTE: various preprocessing and normalizations schemes (z-scoring,
    # square-root-transforming the spike counts, etc.) could be tried here.

    # Create figure.

    fig, axes = plt.subplots(len(neurons_to_plot), 3, figsize=(9.5, 6))

    # First column, raw data.
    plot_column(
        axes[:, 0], data
    )

    # Second column, re-sorted trials by warping function.
    plot_column(
        axes[:, 1],
        data.reorder_trials(model.argsort_warps())
    )

    # Third column, shifted alignment.
    plot_column(
        axes[:, 2],
        validated_alignments
    )

    fig.suptitle("Data")
    axes[0, 0].set_title("raw data")
    axes[0, 1].set_title("sorted by warp (" + label + " warp)")
    axes[0, 2].set_title("aligned by model (" + label + " warp)")

    for index, axis in enumerate(axes[:, 0]):
        axis.set_ylabel("n. " + str(neurons_to_plot[index]))
    fig.tight_layout()
    fig.subplots_adjust(hspace=.9, top=0.9)

# TODO save plots
plt.show(block=True)

# line to add conda to path: export PATH=$PATH:/storage2/perentos/code/python/conda/anaconda/bin
# https://askubuntu.com/questions/186808/every-command-fails-with-command-not-found-after-changing-bash-profile


