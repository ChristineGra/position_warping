import os

import math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from affinewarp import SpikeData
from affinewarp import ShiftWarping, PiecewiseWarping
from affinewarp.crossval import heldout_transform

import pickle


def compute_mean_FR(spike_data, neuron):
    int_spikes = [int(x) for x in spike_data.spiketimes[spike_data.neurons == neuron]]
    occurrances = Counter(int_spikes)
    values = [float(x) for x in occurrances.values()]
    keys = [float(x) for x in occurrances.keys()]
    keys_sorted, values_sorted = zip(*sorted(zip(keys, values)))
    return keys_sorted, values_sorted

    
def map_over_boundaries(spike_data):
    # account for negative or positive shift over boundary
    new_spikepositions = []
    for spkpos in spike_data.spiketimes:
        new_spkpos = spkpos
        while new_spkpos < spike_data.tmin:
            new_spkpos = new_spkpos + spike_data.tmax
        while new_spkpos > spike_data.tmax:
            new_spkpos = new_spkpos - spike_data.tmax
        new_spikepositions.append(int(new_spkpos))
    return np.asarray(new_spikepositions)



def plot_column(axes1, axes2, spike_data, neurons_to_plot):
    raster_kws = dict(s=4, c='k', lw=0)
    limit = math.ceil(len(neurons_to_plot)/2)
    xlim = 100
    # use this line to map shifts over boundaries to beginning/ end
    # spike_data.spiketimes = map_over_boundaries(spike_data)

    # plot first half of neurons in left part
    for n, ax in zip(neurons_to_plot[:limit], axes1):
        FRkeys, FRvalues = compute_mean_FR(spike_data, n)
        ax.scatter(
            spike_data.spiketimes[spike_data.neurons == n],
            spike_data.trials[spike_data.neurons == n],
            **raster_kws,
        )
        ax.set_xlim(left=0, right=xlim)
        ax.set_ylim(ymin=0, ymax=60)
        ax.set_yticks([30])
        ax2 = ax.twinx()
        ax2.plot(FRkeys, FRvalues, c='r')
        ax2.set_ylim(ymin=0, ymax=130)

    # plot second half of neurons in right part
    for n, ax in zip(neurons_to_plot[limit:], axes2):
        FRkeys, FRvalues = compute_mean_FR(spike_data, n)
        ax.scatter(
            spike_data.spiketimes[spike_data.neurons == n],
            spike_data.trials[spike_data.neurons == n],
            **raster_kws,
        )
        ax.set_xlim(left=0, right=xlim)
        ax.set_ylim(ymin=0, ymax=60)
        ax.set_yticks([30])
        ax2 = ax.twinx()
        ax2.plot(FRkeys, FRvalues, c='r')
        ax2.set_ylim(ymin=0, ymax=130)


#######################################################################################
# load dataset to be analysed
path_datasets_folder = "datasets"
# dataset format: [neuron, trial, position]
dataset = np.load(os.path.join(path_datasets_folder, "dataset_NP46_2019-12-02_18-47-02.npy"))
print(dataset.shape)

trials = [int(x) for x in dataset[1]]
spike_positions = [int(x) for x in dataset[2]]
spike_IDs = [int(x) for x in dataset[0]]

spike_IDs_set = list(set(spike_IDs))
spike_IDs_set.sort()
new_spike_IDs = range(len(spike_IDs_set))
print(spike_IDs_set)

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

# determine spikes to plot together
new_spike_IDs = list(set(spike_IDs))
slices = [i for i in new_spike_IDs if i%10 == 0]
slices.append(len(new_spike_IDs))
print(slices)

# get list of all files

# for each file extract parameters

# plot information

for label in ['shift', 'linear', 'pwise-1', 'pwise-2']:
    saves_folder = "saves"
    filename = os.path.join(saves_folder, label,"heldout_validated_alignments_" + str(label) + "_warpreg0_nbins50_iterations50_l2reg1e-07_smoothreg8")
    if label == 'shift':
        filename = filename + "_maxlag0.4"
        
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

        # Second column, shifted alignment.
        plot_column(
            axes[:, 1], axes[:, 3],
            validated_alignments, neurons_to_plot
        )

        # annotate plots
        fig.suptitle("Data for " + label + " warp")
        axes[0, 0].set_title("raw data")
        axes[0, 2].set_title("raw data")
        axes[0, 1].set_title("aligned by model")
        axes[0, 3].set_title("aligned by model")

        for ax in axes[:, 0]:
            ax.set_ylabel("trial")
        for  ax in axes[-1, :]:
            ax.set_xlabel("position")
        
        # for index, axis in enumerate(axes[:, 0]):
            # axis.set_ylabel("n. " + str(neurons_to_plot[index]))
        fig.subplots_adjust(hspace=.3, wspace=.4, top=0.9)
        
        # save plots
        path_plots_folder = "plots"
        plt.savefig(os.path.join(path_plots_folder, label, "binned_warp0_nbins50", "FR_n_neuron" + str(slice1) + "-" + str(slice2) + "warp0nbins50_binned"))

# plt.show(block=True)
