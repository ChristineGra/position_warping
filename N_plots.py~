import os

import math
import numpy as np
import matplotlib.pyplot as plt

from affinewarp import SpikeData
from affinewarp import ShiftWarping, PiecewiseWarping
from affinewarp.crossval import heldout_transform

import pickle


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


# load dataset to be analysed
path_datasets_folder = "datasets"
# [spike position, trial number, spikeID]
# NOTE: if version is changed, adjust start and end of trials?
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
    tmin=0,  # start of trials
    tmax=150,  # end of trials
)
neurons_to_plot = [i for i in new_spike_IDs if spike_IDs.count(i) > 15]
neurons_to_plot = neurons_to_plot[:5]

for label in ['shift', 'linear', 'pwise-1', 'pwise-2']:
    saves_folder = "saves"
    filename = os.path.join(saves_folder,"validated_alignments_" + str(label))
    pickle_in = open(filename, 'rb')
    validated_alignments = pickle.load(pickle_in)

    # NOTE: various preprocessing and normalizations schemes (z-scoring,
    # square-root-transforming the spike counts, etc.) could be tried here.

    # Create figure.

    fig, axes = plt.subplots(len(neurons_to_plot), 2, figsize=(9.5, 6))

    # First column, raw data.
    plot_column(
        axes[:, 0], data
    )

    # Third column, shifted alignment.
    plot_column(
        axes[:, 1],
        validated_alignments
    )

    fig.suptitle("Data")
    axes[0, 0].set_title("raw data")
    # axes[0, 1].set_title("sorted by warp (" + label + " warp)")
    axes[0, 1].set_title("aligned by model (" + label + " warp)")

    for index, axis in enumerate(axes[:, 0]):
        axis.set_ylabel("n. " + str(neurons_to_plot[index]))
    fig.tight_layout()
    fig.subplots_adjust(hspace=.9, top=0.9)
    
    # save plots
    path_plots_folder = "plots"
    plt.savefig(os.path.join(path_plots_folder, "n_n0-4_"+str(label)))
