import os

import math
import numpy as np
import matplotlib.pyplot as plt
import bisect


def rotate(center, point, angle):
    """
    helper function to rotate point on to x axis (anticlockwise)
    center: point to rotate around (x, y)
    point: point that will be rotated (x,y)
    angle: angle that the point will be rotated around the center (in radians)
    returns: x and y coordinates of rotated point
    """
    ox, oy = center
    px, py = point

    newx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    newy = oy + math.cos(angle) * (py - oy) + math.sin(angle) * (px - ox)
    return newx, newy


def plot_position_figures(data_selectionx, data_selectiony, coeffs=None, \
                            show=True, title=None):
    """
    Function to plot x and y traces separately to extract time course and
    plot the form of the maze (meaning x and y positions in relation)
    optionally plot linear fit for maze in figure
    data_selectionx: selected part of x data that will be plotted (list)
    data_selectiony: selected part of y data that will be plotted (list)
    coeffs(optional): coefficients of linear fit for maze form (list)
    show(optional): determines if program is blocked to show figures (boolean)
    title(optional): title of the x and y trace plot (string)
    """
    # plot x and y traces
    plt.figure()
    plt.plot(data_selectionx, label="x")
    plt.plot(data_selectiony, label="y")
    plt.legend()
    if title is not None:
        plt.title(title)

    # plot shape of maze: x and y in relation
    plt.figure()
    plt.xlabel("x positions")
    plt.ylabel("y positions")
    plt.scatter(data_selectionx, data_selectiony, s=2)
    if title is not None:
        plt.title(title)
    if coeffs is not None:
        plt.plot(data_selectionx, coeffs[0] * np.asarray(data_selectionx) \
                    + coeffs[1], color='red')
    if show:
        plt.show(block=True)


def plot_neuron_fireposition_figure(spike_position_and_ID):
    """
    Function to plot neuron IDs and respective firing positions (not aligned to
    x trace of mouse)
    spike_position_and_ID: list containing [spike positions, neuron IDs]
    """
    # TODO: use raster plots?
    plt.scatter(spike_position_and_ID[1,:], spike_position_and_ID[0, :], s=2)
    plt.xlabel("ID")
    plt.ylabel("position")
    plt.show(block=True)


def plot_max_min_x(new_x, start=None, end=None, spikes_different_neurons=None):
    """
    Function to plot the x trace to be used for spike annotating (cut to
    linear runs)
    optionally plot the computed start and end points for each runs
    optionally plot the spikes of different neurons after annotating (NOTE: bug!!)
    new_x: x data for runs (list)
    start: start points of runs (list)
    end: end points of runs (list)
    spikes_different_neurons: list containing [[x position, index] for
    each neuron ID]
    """
    plt.figure()
    plt.plot(new_x, label="positional trace")
    plt.scatter(start, new_x[start], marker='x', s=10, c='tab:orange', \
                label='start')
    plt.scatter(end, new_x[end], marker='x', s=10, c='tab:purple', label='end')
    plt.title("X positions and start and end point of each trial")

    if spikes_different_neurons is not None:
        for l in range(len(spikes_different_neurons)):
            if l == 0 or l ==1:
                continue
            else:
                print("number of spikes: ", len(spikes_different_neurons[l][1]))
                plt.scatter(spikes_different_neurons[l][1], spikes_different_neurons[l][0],\
                            s=5, label="neuron " + str(l))

    plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
    plt.show(block=True)


def compute_position_of_spike(spike_position_1d, trial, spikeID, start_indices_x, \
                            end_indices_x, x_data, start_distances_1d, end_distances_1d, dataset_number):
    """
    Function to compute the x position (running right and left) of a spike
    from its 1D position
    spike_pos: 1D position of spike
    trial: trial the spike is associated to
    spikeID: neuron ID the spike is associated to
    start_indices: indices in trace plot where runs begin
    end_indices: indices in trace plot where runs end
    x_data: x positions for trace plot
    start_distances: 1D position of starts
    end_distances: 1D position of ends
    returns: x position of spike, range of x positions between one start and end,
            start index of this trial
    """
    # find start and end index of this trial
    if dataset_number == 0:
        trial_start_index = start_indices_x[trial * 2]
        trial_end_index = end_indices_x[trial * 2]

        # find start and end distance of the trial
        trial_start_distance = start_distances_1d[trial * 2]
        trial_end_distance = end_distances_1d[trial * 2]

    elif dataset_number == 1:
        trial_start_index = start_indices_x[trial * 2 + 1]
        trial_end_index = end_indices_x[trial * 2 + 1]

        # find start and end distance of the trial
        trial_start_distance = start_distances_1d[trial * 2 + 1]
        trial_end_distance = end_distances_1d[trial * 2 + 1]

    # find start and end x value of the trial
    trial_start_x = x_data[trial_start_index]
    trial_end_x = x_data[trial_end_index]

    if trial_start_x < trial_end_x:
        # map spike position from distance measure to x value
        x_position_of_spike = np.interp(spike_position_1d, [trial_start_distance, trial_end_distance], [trial_start_x, trial_end_x])
    elif trial_start_x > trial_end_x:
        new_start_distance = 0
        new_end_distance = trial_end_distance - trial_start_distance
        new_spike_pos = trial_end_distance - spike_position_1d
        x_position_of_spike = np.interp(new_spike_pos, [new_start_distance, new_end_distance], [trial_end_x, trial_start_x])
    else:
        raise Exception("Error in trial start and end")

    return x_position_of_spike


def annotate_and_sort_data(spike_pos, spikeID, start_distances, end_distances, x_data, start_indices, end_indices, overall_start_distance):
    # bisect returns the index in start array the position would be inserted to
    # this position - 1 is the index of the overall run the spike belongs to
    belongs_to_overall_trial_start_number = bisect.bisect_right(start_distances, spike_pos) - 1
    belongs_to_overall_trial_end_number = bisect.bisect_left(end_distances, spike_pos)
    if belongs_to_overall_trial_start_number == -1:
        # value lies before start of experiment
        # continue
        return -1
    elif belongs_to_overall_trial_start_number != belongs_to_overall_trial_end_number:
        # value belongs between two runs -> ignore
        #continue
        return -1
    elif belongs_to_overall_trial_end_number > len(end_distances):
        # first value after end of experiment reached -> end annotating
        # break
        return -2
    else:
        # sort into right dataset dependent on overall trial number
        # (assuming they appear alternatingely)
        dataset_number = belongs_to_overall_trial_start_number % 2

        trial_in_dataset = math.floor(belongs_to_overall_trial_start_number / 2)
        x_pos = compute_position_of_spike(spike_pos, trial_in_dataset, spikeID, start_indices, \
                                    end_indices, x_data, start_distances, end_distances, dataset_number)
        if dataset_number == 0:

            final_data_0[0].append(x_pos)
            final_data_0[1].append(trial_in_dataset)
            final_data_0[2].append(spikeID)
        elif dataset_number == 1:
            final_data_1[0].append(x_pos)
            final_data_1[1].append(trial_in_dataset)
            final_data_1[2].append(spikeID)
        else:
            raise Exception("Error in sorting into datasets and annotating spikes!!")

    # return that everything worked okay
    return 0


################### paths to files needed #########################
# whl file -> positions (x1, y1, x2, y2) for two LEDs;
# either choose one or take average
path_whl = '/storage/antsiro/data/blab/kenji/ec013.752_769/ec013.752_769.whl'

# res file -> spike times
path_res = '/storage/antsiro/data/blab/kenji/ec013.752_769/ec013.752_769.res.5'

# clu file -> neuron id per spike (1st line is total number of clusters;
# clusters-2 is total number of neurons -> exclude spikes with 0 and 1)
path_clu = '/storage/antsiro/data/blab/kenji/ec013.752_769/ec013.752_769.clu.5'

# code file
path_code = '/storage2/perentos/code/python/conda/affinewarp/affinewarp/data_warping.py'


#################### global variables ###############################
# spikes are recorded with a resolution of 20000
res_spikes = 20000

# positions are recorded with a resolution of 39
res_pos = 39

# load files
whl_data = np.genfromtxt(path_whl)
res_data = np.genfromtxt(path_res)
clu_data = np.genfromtxt(path_clu)
print("data loading done")

############################################################################

# 1: extract boundaries of trials for different orientations
# TODO compute boundaries, the following are hard coded by looking at plots
lim1 = 835000  # beginning of first maze experiment (original -500000)
lim2 = 881000  # end of first experiment / beginning of second experiment (original -394000)
lim3 = 934000  # end of second/ beginning of third (original -341000)
lim4 = 979000  # end of third (original -300000)

limits = [lim1, lim2, lim3, lim4]
# print("length of whl data: ", len(whl_data[:,0]))  # = 1274616

# compute 1d positions for all trials
# extract positions
distance = 0
# collect  distance at which runs begin/ end
limit_distances = []
# last x that was used for distance computation
last_x = whl_data[0, 0]
# iterate over all x and add up difference
# to get absolute distance the animal walked
for index, x in zip(range(1, len(whl_data[1:,0])), whl_data[1:,0]):
    distance = distance + np.abs(last_x - x)
    last_x = x
    # save value if it corresponds to a beginning or end of a run
    if index in limits:
        limit_distances.append(distance)
        print("limit distance calculated")

print("limit distances: ", limit_distances)

# focus on horizontal experiment first
data_selectionx = whl_data[lim3:lim4,0]
data_selectiony = whl_data[lim3:lim4,1]

# ignore -1 values -> -1 means no measurement
data_selectionx = data_selectionx[data_selectionx > -1]
data_selectiony = data_selectiony[data_selectiony > -1]

# plot_position_figures(whl_data[979000:, 0], whl_data[979000:, 1], show=True, title="old")
################################################################################
# 2: ignore cells with less than 50 spikes

# extract number of clusters
n_clusters = clu_data[0]
print("overall clusters: " + str(n_clusters))

spikes_to_ignore = [0, 1]

# go through all clusters except 0 and 1 and count spikes
for i in range(2, int(n_clusters)):
	# print(i)
	count = (clu_data[1:] == float(i)).sum()
	# print(count)

	# ignore cluster if less than 50 spikes
	if count < 50:
		spikes_to_ignore.append(i)

# final list of spikes to ignore
print("list of clusters to ignore: " + str(spikes_to_ignore))

# merge arrays for spike times and neuron IDs
res_clu_array = np.vstack((res_data, clu_data[1:]))

# find indices to delete because cluster ID is 0, 1
# or neuron has less than 50 spikes
indices_to_delete = []
for index, elem in zip(range(res_clu_array.shape[1]), res_clu_array[1, :]):
	if elem in spikes_to_ignore:
		indices_to_delete.append(index)

spike_position_and_ID = np.delete(res_clu_array, indices_to_delete, axis=1)
# print("cleared array: ", res_clu_clear)
# print("shape of cleared array: ", spike_position_and_ID.shape)

################################################################################



# 3: convert spike times to positions

for index, time in zip(range(spike_position_and_ID.shape[1]), \
                        spike_position_and_ID[0, :]):
	pos_for_time = round((time * res_pos) / res_spikes)
	spike_position_and_ID[0, index] = pos_for_time

print("max position of a spike: ", max(spike_position_and_ID[0, :]))
# NOTE: spike_position_and_ID[0,:] contains positions in position resolution;
#            res_clu_clear[1,:] contains ID of spiking neuron

# show neurons with respective firing positions
# plot_neuron_position_figure(spike_position_and_ID)

################################################################################
# 4: linearize positions

# fit linear function on positional data
coeffs_linear_fit = np.polyfit(data_selectionx, data_selectiony, deg=1)

# test if function fits data
# plot_position_figures(data_selectionx, data_selectiony_adjusted, coeffs_linear_fit)

# compute zero crossing -> point to turn around
zero_crossing = (coeffs_linear_fit[1] * -1) / coeffs_linear_fit[0]

# compute angle
rad = math.atan(coeffs_linear_fit[0])
angle = math.degrees(rad)
print("angle to correct for: " , angle)

# compute linearization

new_x = []
new_y = []

# print(data_selectionx.shape)
for x, y in zip(data_selectionx, data_selectiony):
        newx, newy = rotate(center=(zero_crossing, 0), point=(x, y), angle=rad * -1)
        new_y.append(newy)
        new_x.append(newx)

# adjust for offset on y axis
mean_y = np.mean(new_y)
new_y_adjusted = new_y - mean_y

print("min x old: ", min(data_selectionx))
print("min x new: ", min(new_x))

#check rotation by fitting new line
coeffs_linear_fit_new = np.polyfit(new_x, new_y_adjusted, deg=1)
print("new coeffs: ", coeffs_linear_fit_new)

# NOTE: execute next lines to check alignment
# plot_position_figures(new_x, new_y_adjusted, coeffs_linear_fit_new, \
#                       show=False, title="new")
# plot_position_figures(data_selectionx, data_selectiony, coeffs_linear_fit, \
#                       show=False, title="old")
# plt.show(block=True)

##############################################################################
# 5: extract trials -> each direction is different trial

# get positional trace over time and search for minima and maxima
new_x = np.asarray(new_x)
# cut off data on ends of maze because we are interested in linear runs
upper_cutoff = 255
lower_cutoff = 50
new_x_nan = np.where((new_x > upper_cutoff) | (new_x < lower_cutoff), \
                            float("nan"), new_x)

# find minima and maxima -> trial boundaries

# set impossible indices to detect errors
start_index = -1
end_index = -1
# variable to detect whether we are in a region of nans or not
# (regions that we do not look at)
bool_nan_found = False

# search for valid start
for x, index in zip(new_x_nan, range(new_x_nan.shape[0])):
    # search for first region of nans
    if math.isnan(x):
        bool_nan_found = True
    else:
        # if region of nans is found, the forst value after is our start
        if bool_nan_found and x > lower_cutoff and x < upper_cutoff:
            start_index = index
            break
        # other case: we start at the right value -> trial begins here
        elif x == lower_cutoff or x == upper_cutoff:
            start_index = index
            break

bool_nan_found = False
# search for valid end point
for x, index in zip(reversed(new_x_nan), reversed(range(new_x_nan.shape[0]))):
    # search for first region of nans
    if math.isnan(x):
        bool_nan_found = True
    else:
        # if region of nans is found, the forst value after is our end
        if bool_nan_found and x > lower_cutoff and x < upper_cutoff:
            end_index = index
            break
        # other case: we start at the right value -> trial ends here
        elif x == lower_cutoff or x == upper_cutoff:
            end_index = index
            break

# variable to keep track of last found trial boundary
last_index = start_index

# NOTE: runs in different direction will be treated as different experiments ...
# ... (because there can be direction-sensitive place cells)
# list of start indices
start_indices = [start_index]
# list of end indices
end_indices = []

bool_nan_found = False
for x, index in zip(new_x_nan, range(new_x_nan.shape[0])):
    # start searching at beginning of trial we found before
    if index > last_index:
        # we found the first nan of a region -> end of a run
        if math.isnan(x) and not bool_nan_found:
            bool_nan_found = True
            last_index = index - 1
            end_indices.append(last_index)
        # we found the first number after a nan region -> start of a run
        elif not math.isnan(x) and bool_nan_found:
            bool_nan_found = False
            last_index = index
            start_indices.append(last_index)

    # stop searching at end of trial we found before
    if last_index == end_index:
        break

# sanity checks
bool_sorted = True
# start and end indices have to be ordered
if start_indices != sorted(start_indices) or end_indices != sorted(end_indices):
    bool_sorted = False

# equal amount of start and end indices
if len(start_indices) != len(end_indices) or not bool_sorted:
    raise Exception("ERROR in start and end indices!!!!!")

##########################################################################
# 6: Annotate dataset

# 1d distance of start of this experiment
overall_start_distance = limit_distances[2]
overall_end_distance = limit_distances[3]

# extract positions relative to overall start
distance = 0
# collect  distance at which runs begin
start_distances = []
# collect distances at which runs end
end_distances = []
# last x that was used for distance computation
last_x = new_x[0]
# iterate over all x and add up difference
# to get absolute distance the animal walked
for index, x in zip(range(1, len(new_x[1:])), new_x[1:]):
    distance = distance + np.abs(last_x - x)
    last_x = x
    # save value if it corresponds to a beginning or end of a run
    if index in start_indices:
        start_distances.append(distance)
    if index in end_indices:
        end_distances.append(distance)

print("distance: ", distance)

# sanity checks
bool_sorted = True
# start and end values have to be ordered
if start_distances != sorted(start_distances) or \
    end_distances != sorted(end_distances):
    bool_sorted = False

# equal amount of start and end values
if len(start_distances) != len(end_distances) or not bool_sorted or \
    len(start_distances) != len(start_indices):
    raise Exception("ERROR in start and end positions!!!!!")

# create two new arrays that will be final datasets (position, ID, trial)
final_data_0 = [[], [], []]  # np.zeros((count_trials_1, 3))
final_data_1 = [[], [], []]  # np.zeros((count_trials_2, 3))

# copy position an ID in correct dataset and add trial number
for spike_pos, spikeID in zip(spike_position_and_ID[0], spike_position_and_ID[1]):
    if spike_pos >= overall_start_distance and spike_pos <= overall_end_distance:
        # compute spike position relative to experiment that is being evaluated
        spike_pos_rel = spike_pos - overall_start_distance
        check = annotate_and_sort_data(spike_pos_rel, spikeID, start_distances, end_distances, new_x_nan, start_indices, end_indices, overall_start_distance)
        if check == -1:
            continue
        elif check == -2:
            break

print("Annotated data example: ")
print("Spike positions: ", final_data_0[0][:200])
print("Trial numbers: ", final_data_0[1][:200])
print("Spike IDs: ", final_data_0[2][:200])
print("\n")
print("Spike positions: ", final_data_1[0][:200])
print("Trial numbers: ", final_data_1[1][:200])
print("Spike IDs: ", final_data_1[2][:200])

# plot_max_min_x(new_x_nan, start=start_indices, end=end_indices)
###############################################################################
# 7: save datasets
path_datasets_folder = "datasets"
# TODO: also save min and max x position
np.save(os.path.join(path_datasets_folder, "dataset_ec013.752_769.5_part0.npy"), final_data_0)
np.save(os.path.join(path_datasets_folder, "dataset_ec013.752_769.5_part1.npy"), final_data_1)
print("saving done")

###############################################################################
# 8: check annotations
# extract x position
"""
# TODO: something doesn't work here!!!!! TODO
spikes_different_neurons = [[[], []] for i in range(int(n_clusters))]

for spike_pos, trial, spikeID in zip(final_data_0[0], final_data_0[1], final_data_0[2]):
    # NOTE: this only works (badly) for final_data_0 at the moment
    x_position_of_spike, x_data_range, trial_start_index = \
    compute_position_of_spike(spike_pos, trial, spikeID, start_indices, \
                        end_indices, new_x_nan, start_distances, end_distances)

    # find index to plot spike with x position
    index_of_spike = -1
    if x_position_of_spike in x_data_range:
        # either the exact value already exists in the data
        index_of_spike = x_data_range.index(x_position_of_spike)
    else:
        # or the index has to be approximated by finding index to insert spike
        index_of_spike = bisect.bisect_right(x_data_range, x_position_of_spike)

    # append data to plot to list
    spikes_different_neurons[int(spikeID)][0].append(x_position_of_spike)
    spikes_different_neurons[int(spikeID)][1].append(trial_start_index + index_of_spike)

plot_max_min_x(new_x_nan, start_indices, end_indices)
"""
# TODO: plot spikes in plot of x to get overview of spike positions and check whether annotatimng worked
# -> didn"t work too well
# TODO: code cleanup!!
# TODO: solve nonlinearities in running of mouse
