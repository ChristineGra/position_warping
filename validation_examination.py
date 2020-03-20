import os

import math
import numpy as np
import matplotlib.pyplot as plt
import statistics
from os import walk
from affinewarp import SpikeData
import pickle

# load original dataset for comparison
path_datasets_folder = "datasets"
# [neuron, trial, position]
dataset = np.load(os.path.join(path_datasets_folder, "dataset_NP46_2019-12-02_18-47-02.npy"))

trials = [int(x) for x in dataset[1]]
spike_positions = [x for x in dataset[2]]
spike_IDs = [int(x) for x in dataset[0]]

spike_IDs_set = list(set(spike_IDs))
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

########################################################################################################

# find all datasets
pathname = "saves"

files = []
for (dirpath, dirnames, filenames) in walk(pathname):
    files.extend(filenames)    

# select a neuron ID (representative for all others)
selected_neuronID = 31

# create lists: [nbins, smooth, warp, l2 (, lag)]
shift_model = [ [[],[]] for i in range(4)]
shift_model.append([[],[],[]])
linear_model = [ [[],[]] for i in range(4)]
p1_model =  [ [[],[]] for i in range(4)]
p2_model =  [ [[],[]] for i in range(4)]
model_list = [shift_model, linear_model, p1_model, p2_model]

# parameter values used for alignments
nbins_list = [50, 120]
iterations_list = [50]
smooth_list = [3, 8]
warp_list = [0.0, 0.6]
l2_list = [1e-7, 1e-5]
lag_list = [0.1, 0.25, 0.4]
param_list = [nbins_list, smooth_list, warp_list, l2_list, lag_list]


# for every dataset:

for fi in files:
    
    # extract parameters from title
    
    title_parts = fi.split("_")
    
    model_label = title_parts[3]
    
    _, warp_reg = title_parts[4].split("reg")
    warp_reg = float(warp_reg)
    if warp_reg not in warp_list:
        continue
    
    _, nbins = title_parts[5].split("bins")
    nbins = int(nbins)
    if nbins not in nbins_list:
        continue
    
    _, iterations = title_parts[6].split("iterations")
    iterations = int(iterations)
    if iterations not in iterations_list:
        continue
    
    _, l2_reg = title_parts[7].split("reg")
    l2_reg = float(l2_reg)
    if l2_reg not in l2_list:
        continue
    
    _, smooth_reg = title_parts[8].split("reg")
    smooth_reg = int(smooth_reg)
    if smooth_reg not in smooth_list: continue

    maxlag = 0
    if len(title_parts) == 10:
        _, maxlag = title_parts[9].split("lag")
        maxlag = float(maxlag)
        if maxlag not in lag_list:
            continue

    # load data
    complete_path = os.path.join(pathname, model_label, fi)
    pickle_in = open(complete_path, 'rb')
    validated_alignments = pickle.load(pickle_in)

    diffs = []
    # compute the average shift for each trial for the selected neuron
    for index, pos1, pos2 in zip(range(len(data.spiketimes)), data.spiketimes, validated_alignments.spiketimes):
        if data.neurons[index] == selected_neuronID:
            diff = abs(pos1 - pos2)
            diffs.append(diff)

    avg_diff = statistics.mean(diffs)

    # find correct places in lists to append value
    if model_label == "shift":
        list_sel = shift_model
    elif model_label == "linear":
        list_sel = linear_model
    elif model_label == "pwise-1":
        list_sel = p1_model
    elif model_label == "pwise-2":
        list_sel = p2_model

    bins_index = nbins_list.index(nbins)
    iter_index = iterations_list.index(iterations)
    smooth_index = smooth_list.index(smooth_reg)
    warp_index = warp_list.index(warp_reg)
    l2_index = l2_list.index(l2_reg)
    if maxlag is not 0:
        maxlag_index =  lag_list.index(maxlag)

    # append value
    list_sel[0][bins_index].append(avg_diff)
    list_sel[1][smooth_index].append(avg_diff)
    list_sel[2][warp_index].append(avg_diff)
    list_sel[3][l2_index].append(avg_diff)
    if maxlag is not 0:
        list_sel[4][maxlag_index].append(avg_diff)

print("sorting done")

# visualization

# iterate through models
for model, model_title in zip(model_list, ["shift", "linear", "pwise-1", "pwise-2"]):
    
    # iterate through parameters
    for parameter, param_title in zip(range(len(model)), ["number of bins", "smooth reg", "warp reg", "l2 reg", "max lag reg"]):
        
        plt.figure()
        plt.title(model_title + " warping: " + param_title)
        plt.xlabel("value of parameter")
        plt.ylabel("average shift")
        save_path = os.path.join("plots", model_title, param_title)
        
        # iterate through different values of parameter
        for ind, pos, value in zip(range(len(model[parameter])), [0.5, 1.5, 2.5], param_list[parameter]):
            data = model[parameter][ind]
            # plot boxplot
            plt.boxplot(data, positions=[pos], labels=[str(value)])
           # save figure 
        plt.savefig(save_path)
