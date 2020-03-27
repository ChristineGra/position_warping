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

def plot_data(data, neuronID, axis, category):
        # plot data
        axis.imshow(np.squeeze(data))
        # compute mean firing rate
        FR = np.mean(data, axis=0)
        # plot firing rate
        axis.plot(FR, c='r')
        axis.set_ylim(bottom=0, top=60)

        title = str(neuronID)[2:-2] + " " + category
        axis.set_title(title, fontsize=6)
        axis.set_xticks([100])

        
def visualize_multiple_neurons(slices, model, label, neurons, data_transformed, selected_trials):
        # plot neurons for each slice
        for slice1, slice2 in zip(slices[:-1], slices[1:]):
            # determine which neurons belong to slice
            neurons_to_plot = neurons[slice1: slice2]
            print(neurons_to_plot)
            # compute cutoff to place plots next to each other
            limit = math.ceil(len(neurons_to_plot)/2)
            print(limit)
            # create subplots
            fig, axes = plt.subplots(limit, 4, figsize=(9.5, 6))
            fig.subplots_adjust(hspace=.4)
            fig.suptitle("Data for " + label + " warp")
            axes[0, 0].set_title("raw data")
            axes[0, 2].set_title("raw data")
            axes[0, 1].set_title("aligned by model")
            axes[0, 3].set_title("aligned by model")

            for ax in axes[:, 0]:
                ax.set_ylabel("trial")
            for  ax in axes[-1, :]:
                ax.set_xlabel("position")


            # select neuron
            for neuronID, index in zip(neurons_to_plot, range(len(neurons_to_plot))):
                if index >= limit:
                    ax_raw = axes[index-limit, 2]
                    ax_transformed = axes[index-limit, 3]
                else:
                    ax_raw = axes[index, 0]
                    ax_transformed = axes[index, 1]

                # select data
                overall_index = slice1 + index
                raw_data = selected_trials[:, :, overall_index]
                raw_data = np.expand_dims(raw_data, -1)

                # plot raw data with FR
                plot_data(raw_data, neuronID, ax_raw, "raw")

                # plot transformed data with FR
                transformed_data = data_transformed[:, :, overall_index]
                plot_data(transformed_data, neuronID, ax_transformed, "aligned")
            # save figure
            plt.savefig(os.path.join(path_save, str(label) + "neuron" + str(slice1) + "-" + str(slice2)))
        

def visualize_single_neurons(selected_trials, index, neuronID, data_transformed, label):
        # select data
        raw_data = selected_trials[:, :, index]
        raw_data = np.expand_dims(raw_data, -1)
        transformed_data = data_transformed[:, :, index]

        # plot raw data with FR
        plt.figure()
        # plot data
        plt.imshow(np.squeeze(raw_data))
        # compute FR
        FR = np.mean(raw_data, axis=0)
        # plot FR
        plt.plot(FR, c='r')
        plt.ylim(bottom=0, top=60)

        title = str(neuronID)[2:-2] + "_raw"
        plt.title(title, fontsize=6)
        plt.xticks([100])
        # save figure
        plt.savefig(os.path.join(path_save, str(label)+ "_neuron" + str(neuronID) +"_raw"))

        # plot transformed data with FR
        plt.figure()
        # plot data
        plt.imshow(np.squeeze(transformed_data))
        # compute FR
        FR = np.mean(transformed_data, axis=0)
        # plot FR
        plt.plot(FR, c='r')
        plt.ylim(bottom=0, top=60)

        title = str(neuronID)[2:-2] + "_transformed"
        plt.title(title, fontsize=6)
        plt.xticks([100])
        # save figure
        plt.savefig(os.path.join(path_save, str(label) + "_neuron" + str(neuronID) + "_transformed"))

        

def load_spikedata(path, path_save, neuron_list=None, model_selected=None, smoothreg=5., warpreg=.3, l2reg=1e-7, maxlagreg=.3, trial_selection=[0, 60]):
        """
        Load mat file, configure dataset for warping, warp and plot
        data have to be continuous
        path: path to data
        path_save: folder to save data in
        neuron_list (optional): list of preselected neurons, if ot specified, neurons will be selected based on 'SSI', 'meanFR', 'muCC' and 'cellType'
        model_selected (optional): selected model, one of "shift", "linear", "piecewise-1" or "piecewise-2", if not specified, all models will be used
        smoothreg (optional): integer between 0 and 10, default 5
        warpreg (optional): float between 0 and 1, default .3
        l2reg (optional): float between 1e-7 and 0, default 1e-7
        maxlagreg (optional): float between 0 and 0.5, default .3
        trial_selection (optional): list of indices of first and (last+1) trials to model, default [0, 60], to select all, use number of trials +1 as last index
        """

        # load .mat file
        data = loadmat(path)

        # data contains __header__, __version__, __globals__, tun
        # data[' __header__'] contains a lot of numbers
        # data['tun'] contains all the data

        celltunings = np.asarray(data['tun']['cellTunings'][0])  # for each of 324 neurons: position (119)  x trial (120) 
        
        selected_trials = []
        # only select first 60 trials since
        for index in range(len(celltunings)):
                celltunings_selected = np.asarray(celltunings[index][:,trial_selection[0]:trial_selection[1]]) 
                selected_trials.append(celltunings_selected)

        selected_trials = np.asarray(selected_trials)  # shape (324, 119, 60) -> neurons x spike positions x trial

        
        # select neurons
        neuronIDs = np.asarray(data['tun']['cellID'][0])  # ID for each of 324 neurons

        # extract parameters to select neurons
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
        if neuron_list is not None:
            # list of neurons to model is specified
            selected_neurons = np.where(np.isin(neuronIDs, neuron_list), True, False)
            selected_neurons = np.expand_dims(selected_neurons, 0)
        else:
            # neurons are not specified, choose depending on criteria
            selected_neurons = np.where((criteria[0] > SSI_lower_limit) & (criteria[1] > mean_FR_lower_limit) & \
                                        (criteria[1] < mean_FR_upper_limit) & (criteria[2] > muCC_lower_limit) & \
                                        (criteria[3] == 2), True, False)

        print("number of selected neurons: ", len([i for i in selected_neurons[0] if i == True]))
        print("shape of selecetd neurons: ", selected_neurons.shape)

        # filter trials so that only selected neurons are taken into account
        selected_trials = selected_trials[selected_neurons[0]]
        # switch axes so that data can be used in model.fit: (num trials, num timepoints, num units)
        selected_trials = np.swapaxes(selected_trials, axis1=0, axis2=2)
        print("selected trials shape: ", selected_trials.shape)

        # create model
        shift_model = ShiftWarping(maxlag=maxlagreg, smoothness_reg_scale=smoothreg, warp_reg_scale=warpreg, l2_reg_scale=l2reg)
        linear_model = PiecewiseWarping(n_knots=0, warp_reg_scale=warpreg, smoothness_reg_scale=smoothreg)
        p1_model = PiecewiseWarping(n_knots=1, warp_reg_scale=warpreg, smoothness_reg_scale=smoothreg)
        p2_model = PiecewiseWarping(n_knots=2, warp_reg_scale=warpreg, smoothness_reg_scale=smoothreg)
        if model_selected is None:
                models = [shift_model, linear_model, p1_model, p2_model]
                labels = ["shift", "linear", "piecewise-1", "piecewise-2"]
        elif model_selected is "shift":
                models = [shift_model]
                labels = ["shift"]
        elif model_selected is "linear":
                models = [linear_model]
                labels = ["linear"]
        elif model_selected is "piecewise-1":
                models = [p1_model]
                labels = ["piecewise-1"]
        elif model_selected is "piecewise-2":
                models = [p2_model]
                labels = ["piecewise-2"]
        else:
                print("Selected model is no valid option!!")
                exit()
        
        # filter neuron IDs so that only selected neurons are taken into account
        neurons = neuronIDs[selected_neurons[0]]

        # compute slices to plot neurons
        slices = [i for i in range(len(neurons)) if i%12 == 0]
        slices.append(len(neurons) + 1)
        print("slices: ", slices)

        for model, label in zip(models, labels):
            # fit model
            model.fit(selected_trials, iterations=30)
            # transform data based on fitted model
            data_transformed = model.transform(selected_trials)

            if len(neurons) <= 2:
                for neuronID, index in zip(neurons, range(len(neurons))):
                        visualize_single_neurons(selected_trials, index, neuronID, data_transformed, label, path_save)
            else:
                visualize_multiple_neurons(slices, model, label, neurons, data_transformed, selected_trials, path_save)
        


neuron_list = np.asarray([6])
path_save = os.path.join("plots", label)
load_spikedata(path_to_data, neuron_list)
