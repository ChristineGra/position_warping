import numpy as np
import matplotlib.pyplot as plt
from affinewarp.datasets import jittered_data
from affinewarp import ShiftWarping, PiecewiseWarping
from affinewarp import SpikeData
from affinewarp.crossval import heldout_transform
import os
import math
from scipy.io import loadmat
import pprint


path_to_data = '/storage2/perentos/data/recordings/NP46/NP46_2019-12-02_18-47-02/processed/place_cell_tunings.mat'

def plot_data(data, neuronID, axis):
        axis.imshow(np.squeeze(data))
        FR = np.mean(data, axis=0)
        axis.plot(FR, c='r')
        axis.set_ylim(bottom=0, top=60)
    

    
def load_spikedata(path):
        """
        Load npz file and configure dataset for warping
        the final data have to have format [trials, spiketimes, neuron_ids, tmin, tmax]
        path: path to data
        """

        # load .mat file
        data = loadmat(path)

        # data contains __header__, __version__, __globals__, tun
        # data[' __header__'] contains a lot of numbers
        # data['tun'] contains all the data

        # allTun = np.asarray(data['tun']['allTun'][1])
        # print(allTun)
        celltunings = np.asarray(data['tun']['cellTunings'][0])  # for each of 324 neurons: position (119)  x trial (120) (?)
        # print(celltunings[0])
        
        selected_trials = []
        # only select first 60 trials
        for index in range(len(celltunings)):
                celltunings_selected = np.asarray(celltunings[index][:,:60]) 
                selected_trials.append(celltunings_selected)

        selected_trials = np.asarray(selected_trials)  # shape (324, 119, 60) -> neurons x spike positions x trial

        
        # select neurons
        neuronIDs = np.asarray(data['tun']['cellID'][0])  # ID for each of 324 neurons

        SSI = data['tun']['SSI']
        meanFR = data['tun']['meanFR']
        muCC = data['tun']['muCC']
        neuron_types = data['tun']['cellType']
        criteria = [SSI, meanFR, muCC, neuron_types]
        print("SSI shape: ", np.asarray(SSI).shape)

        # criteria
        SSI_lower_limit = 0.2
        mean_FR_lower_limit = 0.2
        mean_FR_upper_limit = 5
        muCC_lower_limit = 0.6

        # select neurons that fulfill criteria
        selected_neurons = np.where((criteria[0] > SSI_lower_limit) & (criteria[1] > mean_FR_lower_limit) & \
                                    (criteria[1] < mean_FR_upper_limit) & (criteria[2] > muCC_lower_limit) & (criteria[3] == 2), True, False)
        print("number of selected neurons: ", len([i for i in selected_neurons[0] if i == True]))
        # print("shape of selecetd neurons: ", selected_neurons.shape)
        
        # filter trials so that only selected neurons are taken into account
        selected_trials = selected_trials[selected_neurons[0]]
        # switch axes so that data can be used in model.fit: (num trials, num timepoints, num units)
        selected_trials = np.swapaxes(selected_trials, axis1=0, axis2=2)
        print("selected trials shape: ", selected_trials.shape)

        # create model
        shift_model = ShiftWarping(maxlag=.3, smoothness_reg_scale=8.)
        linear_model = PiecewiseWarping(n_knots=0, warp_reg_scale=0, smoothness_reg_scale=8.)
        p1_model = PiecewiseWarping(n_knots=1, warp_reg_scale=0.5, smoothness_reg_scale=3.)
        p2_model = PiecewiseWarping(n_knots=3, warp_reg_scale=0, smoothness_reg_scale=3.)
        model = p1_model
        
        # fit model
        model.fit(selected_trials, iterations=30)
        # transform data based on fitted model
        data_transformed = model.transform(selected_trials)
        
        # filter neuron IDs so that only selected neurons are taken into account
        neurons = neuronIDs[selected_neurons[0]]
        
        slices = [i for i in range(len(neurons)) if i%12 == 0]
        slices.append(len(neurons) + 1)
        print(slices)

        for slice1, slice2 in zip(slices[:-1], slices[1:]):
            neurons_to_plot = neurons[slice1: slice2]
            print(neurons_to_plot)
            limit = math.ceil(len(neurons_to_plot)/2)
            print(limit)
            fig, axes = plt.subplots(limit, 4, figsize=(9.5, 6))
            
            # select neuron
            for neuronID, index in zip(neurons_to_plot, range(len(neurons_to_plot))):
                if index >= limit:
                    ax_raw = axes[index-limit, 2]
                    ax_transformed = axes[index-limit, 3]
                else:
                    ax_raw = axes[index, 0]
                    ax_transformed = axes[index, 1]
                    
                # select data
                raw_data = selected_trials[:, :, index]
                raw_data = np.expand_dims(raw_data, -1)

                # plot data before warping with FR
                plot_data(raw_data, neuronID, ax_raw)

                # plot transformed data with FR
                transformed_data = data_transformed[:, :, index]
                plot_data(transformed_data, neuronID, ax_transformed)
        plt.show(block=True)



load_spikedata(path_to_data)
