import numpy as np
import matplotlib.pyplot as plt
from affinewarp.datasets import jittered_data
from affinewarp import ShiftWarping, PiecewiseWarping
from affinewarp import SpikeData
from affinewarp.crossval import heldout_transform
import os
from scipy.io import loadmat


path_to_data = '/storage2/perentos/data/recordings/NP46/NP46_2019-12-02_18-47-02/processed/place_cell_tunings.mat'

def load_spikedata(path):
	""" 
	Load npz file and display data 
	data have to have format [trials, spiketimes, neuron_ids, tmin, tmax, sniff_onsets]
	but sniff_onset not secessary
	"""

	# load .mat file
	data = loadmat(path)
	
	# select only neurons of type 1
	index_list = []
	for index, celltype in enumerate(data['cellType'][0, :]):
		if celltype == 1:
			index_list.append(index)

	# select data from other keys that correspond to type 1 cells
	allcelltunings = data['allCellTunings'][0, :, index_list]
	cellids = data['cellID'][0, index_list]
	celltypes = data['cellType'][0, index_list]

	# create new dictionary with selected values
	data_celltype1 = {
						'allCellTunings': allcelltunings,
						'cellID': cellids,
						'cellType': celltypes
	}
	# compare entries to make sure everything is correct
	print(len(data_celltype1['allCellTunings']), len(data_celltype1['cellID']), len(data_celltype1['cellType']))
	print(len(data['allCellTunings'][0,0,:]), len(data['cellID'][0,:]), len(data['cellType'][0,:]))
	
	# data['allCellTunings'][0,:,4]) has same content as data_celltype1['allCellTunings'][4,:]) // length of both is 120 -> is this one tuning curve?

	# return data
	return data_celltype1


# load data from file
spikedata_raw = load_spikedata(path_to_data)


# plots to understand what is going on -> what exactly is "allcellstuning"?

# TODO: how to apply algorithm to this?

"""
spikedata_dict = dict(spikedata_raw)
# create SpikeData object
spikedata = SpikeData(
	trials=spikedata_dict["trials"],
	spiketimes=spikedata_dict["spiketimes"],
	neurons=spikedata_dict["neuron_ids"],
	tmin=spikedata_dict["tmin"],
	tmax=spikedata_dict["tmax"]
	)

# compute sniff onsets relative to time (between zero/ stimulus onset and 1/ trail end) 
rel_sniff_onsets = spikedata_dict["sniff_onsets"] / spikedata_dict["tmax"]

########### HYPERPARAMETERS #####################
nbins = 130			# number of timebins per trial
smooth_reg = 10		# strength of roughness penalty
warp_reg = 0		# strength of warp magnitude penalty
l2_reg = 0			# strength of L2 penalty (template magnitude)
maxlag = 0.5		# maximum amount of shift allowed

###################################################

shift_model = ShiftWarping(
	maxlag=maxlag,
	smoothness_reg_scale=smooth_reg,
	warp_reg_scale=warp_reg,
	l2_reg_scale=l2_reg
	)

# Transform each neuron's activity by holding it out of model fitting
# and applying warping functions fit to the remaining neurons.
print("transform each neuron's activity by holding it out of model fitting")
validated_alignments = heldout_transform(
	model=shift_model, binned=spikedata.bin_spikes(nbins), 
	data=spikedata, iterations=100)

# fit model to full dataset (to align sniffs)
print("model fit")
shift_model.fit(spikedata.bin_spikes(nbins))

########### possible preprocessing here #############
# TODO what kind of preprocessing possible?
##################################################

# manually specify alignment to sniff onsets
# t0: x and y positions of disired warping function for each trial
align_sniff = PiecewiseWarping()

print("manual fit")
align_sniff.manual_fit(
	data=spikedata.bin_spikes(nbins),  
	t0=np.column_stack([rel_sniff_onsets, np.full(spikedata.n_trials, 0.4)]),
	recenter=False
	)  


def plot_column(ax_list, spikedata, sniffs):
	example_neurons = [2, 6, 20, 22, 28, 9]

	# plot raster plot for each neuron
	raster_kws = dict(s=4, c='k', lw=0)
	# iterate over axes
	for n, ax in zip(example_neurons, ax_list[:-1]):
    	# create scatter plot for spikedata
		ax.scatter(
			spikedata.spiketimes[spikedata.neurons == n],
			spikedata.trials[spikedata.neurons == n],
			**raster_kws,
			)
		ax.set_ylim(-1, len(sniffs))
		ax.axis('off')

		# Plot blue dots, denoting sniffs, on rasters
		sniff_kws = dict(c='b', s=5, alpha=.55, lw=0)
		ax.scatter(sniffs, range(sniffs.size), **sniff_kws)

	# Plot histogram at bottom -> distribution of sniff onset times
	histbins = np.linspace(0, 500, 50)
	if len(np.unique(np.histogram(sniffs, histbins)[0])) == 2:
		ax_list[-1].axvline(sniffs.mean(), c='b', alpha=.7, lw=2, dashes=[2,2])
	else:
		ax_list[-1].hist(sniffs, histbins, color='blue', alpha=.65)
    
	# Format bottom subplot
	ax_list[-1].spines['right'].set_visible(False)
	ax_list[-1].spines['top'].set_visible(False)
	ax_list[-1].set_ylim(0, 15)


# Create figure.
fig, axes = plt.subplots(7, 4, figsize=(9.5, 6))


# First column, raw data.
plot_column(
	axes[:, 0], spikedata, spikedata_dict["sniff_onsets"]
)

# Second column, re-sorted trials by warping function.
plot_column(
	axes[:, 1],
	spikedata.reorder_trials(shift_model.argsort_warps()),
	spikedata_dict["sniff_onsets"][shift_model.argsort_warps()]
)

# Third column, shifted alignment.
plot_column(
	axes[:, 2],
	validated_alignments,
	shift_model.event_transform(
		range(spikedata_dict["sniff_onsets"].size), rel_sniff_onsets) * spikedata_dict["tmax"],
)

# Final column, aligned to sniff onset.
plot_column(
	axes[:, 3],
	align_sniff.transform(spikedata),
	align_sniff.event_transform(
		range(spikedata_dict["sniff_onsets"].size), rel_sniff_onsets) * spikedata_dict["tmax"],
)

# Final formatting.
for ax in axes.ravel():
	ax.set_xlim(-50, 550)
for ax in axes[-1]:
	ax.set_xlabel("time (ms)")

# TODO maybe add numbers of neurons to plot
axes[0, 0].set_title("raw data")
axes[0, 1].set_title("sorted by warp")
axes[0, 2].set_title("aligned by model")
axes[0, 3].set_title("aligned to sniff")

fig.tight_layout()
fig.subplots_adjust(hspace=.3)

plt.show(block=True)
print('done')

"""

