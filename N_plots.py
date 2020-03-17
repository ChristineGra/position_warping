import os

import math
import numpy as np
import matplotlib.pyplot as plt

from affinewarp import SpikeData
from affinewarp import ShiftWarping, PiecewiseWarping
from affinewarp.crossval import heldout_transform

import pickle


def plot_column(axes1, axes2, spike_data, neurons_to_plot):
    raster_kws = dict(s=4, c='k', lw=0)
    limit = math.ceil(len(neurons_to_plot)/2)
    for n, ax in zip(neurons_to_plot[:limit], axes1):
        ax.scatter(
            spike_data.spiketimes[spike_data.neurons == n],
            spike_data.trials[spike_data.neurons == n],
            **raster_kws,
        )
        ax.set_xlabel("position")
        ax.set_ylabel("trial")

    for n, ax in zip(neurons_to_plot[limit:], axes2):
        ax.scatter(
            spike_data.spiketimes[spike_data.neurons == n],
            spike_data.trials[spike_data.neurons == n],
            **raster_kws,
        )
        ax.set_xlabel("position")
        ax.set_ylabel("trial")


# load dataset to be analysed
path_datasets_folder = "datasets"
# [neuron, trial, position]
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

new_spike_IDs = list(set(spike_IDs))
slices = [i for i in new_spike_IDs if i%10 == 0]
slices.append(len(new_spike_IDs))
print(slices)

# get list of all files

# for each file extract parameters

# plot information

for label in ['shift', 'linear', 'pwise-1', 'pwise-2', 'pwise-3']:
    saves_folder = "saves"
    filename = os.path.join(saves_folder,"validated_alignments_" + str(label))
    pickle_in = open(filename, 'rb')
    validated_alignments = pickle.load(pickle_in)

    for slice1, slice2 in zip(slices[:-1], slices[1:]):
        neurons_to_plot = list(set(spike_IDs))[slice1: slice2]
        print(neurons_to_plot)
        
        # NOTE: various preprocessing and normalizations schemes (z-scoring,
        # square-root-transforming the spike counts, etc.) could be tried here.
        
        # Create figure.
        
        fig, axes = plt.subplots(math.ceil(len(neurons_to_plot)/2), 4, figsize=(9.5, 6))

        # First column, raw data.
        plot_column(
            axes[:, 0], axes[:, 2], data, neurons_to_plot
        )

        # Third column, shifted alignment.
        plot_column(
            axes[:, 1], axes[:, 3],
            validated_alignments, neurons_to_plot
        )

        fig.suptitle("Data")
        axes[0, 0].set_title("raw data")
        axes[0, 1].set_title("aligned by model (" + label + " warp)")
        
        # for index, axis in enumerate(axes[:, 0]):
            # axis.set_ylabel("n. " + str(neurons_to_plot[index]))
            # fig.subplots_adjust(hspace=.9, top=0.9)
        
        # save plots
        path_plots_folder = "plots"
        plt.savefig(os.path.join(path_plots_folder, label, "n_neuron" + str(slice1) + "-" + str(slice2)))

# plt.show(block=True)
