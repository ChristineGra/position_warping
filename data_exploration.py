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
        axis.imshow(np.squeeze(data))
        FR = np.mean(data, axis=0)
        axis.plot(FR, c='r')
        axis.set_ylim(bottom=0, top=60)

        title = str(neuronID)[2:-2] + " " + category
        axis.set_title(title, fontsize=6)
        axis.set_xticks([100])

        
def visualize_multiple_neurons(slices, model, label, neurons, data_transformed, selected_trials):
        for slice1, slice2 in zip(slices[:-1], slices[1:]):
            neurons_to_plot = neurons[slice1: slice2]
            print(neurons_to_plot)
            limit = math.ceil(len(neurons_to_plot)/2)
            print(limit)
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

                # plot data before warping with FR
                plot_data(raw_data, neuronID, ax_raw, "raw")

                # plot transformed data with FR
                transformed_data = data_transformed[:, :, overall_index]
                plot_data(transformed_data, neuronID, ax_transformed, "aligned")
            plt.savefig(os.path.join("plots", label, "neuron" + str(slice1) + "-" + str(slice2) + str(label) + "_maxlag_warpreg03_smooth5"))
        

def visualize_single_neurons(selected_trials, index, neuronID, data_transformed, label):
        # select data
        raw_data = selected_trials[:, :, index]
        raw_data = np.expand_dims(raw_data, -1)
        transformed_data = data_transformed[:, :, index]
        
        plt.figure()
        plt.imshow(np.squeeze(raw_data))
        FR = np.mean(raw_data, axis=0)
        plt.plot(FR, c='r')
        plt.ylim(bottom=0, top=60)

        title = str(neuronID)[2:-2] + "_raw"
        plt.title(title, fontsize=6)
        plt.xticks([100])
        plt.savefig(os.path.join("plots", label, "neuron" + str(neuronID)[2: -2] + "_raw"))

         
        plt.figure()
        plt.imshow(np.squeeze(transformed_data))
        FR = np.mean(transformed_data, axis=0)
        plt.plot(FR, c='r')
        plt.ylim(bottom=0, top=60)

        title = str(neuronID)[2:-2] + "_transformed"
        plt.title(title, fontsize=6)
        plt.xticks([100])
        plt.savefig(os.path.join("plots", label, "neuron" + str(neuronID)[2: -2] + "_transformed"))

        

def load_spikedata(path, neuron_list=None):
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
        # neuron_list = np.asarray([514])  # np.asarray([102, 137, 154, 196, 511, 716, 736,  872, 936, 950])
        
        # selected_neurons = np.where(np.isin(neuronIDs, neuron_list))

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
            selected_neurons = np.where(np.isin(neuronIDs, neuron_list), True, False)
            selected_neurons = np.expand_dims(selected_neurons, 0)
        else:
            selected_neurons = np.where((criteria[0] > SSI_lower_limit) & (criteria[1] > mean_FR_lower_limit) & \
                                        (criteria[1] < mean_FR_upper_limit) & (criteria[2] > muCC_lower_limit) & \
                                        (criteria[3] == 2), True, False)
        
        # print(selected_neurons)

        print("number of selected neurons: ", len([i for i in selected_neurons[0] if i == True]))
        print("shape of selecetd neurons: ", selected_neurons.shape)

        # filter trials so that only selected neurons are taken into account
        selected_trials = selected_trials[selected_neurons[0]]
        # switch axes so that data can be used in model.fit: (num trials, num timepoints, num units)
        selected_trials = np.swapaxes(selected_trials, axis1=0, axis2=2)
        print("selected trials shape: ", selected_trials.shape)

        # create model
        shift_model = ShiftWarping(maxlag=.3, smoothness_reg_scale=5., warp_reg_scale=0.3)
        linear_model = PiecewiseWarping(n_knots=0, warp_reg_scale=0.3, smoothness_reg_scale=5.)
        p1_model = PiecewiseWarping(n_knots=1, warp_reg_scale=0.3, smoothness_reg_scale=5.)
        p2_model = PiecewiseWarping(n_knots=2, warp_reg_scale=0.3, smoothness_reg_scale=5.)
        models = [shift_model, linear_model, p1_model, p2_model]
        labels = ["shift", "linear", "piecewise-1", "piecewise-2"]
        
        # filter neuron IDs so that only selected neurons are taken into account
        neurons = neuronIDs[selected_neurons[0]]

        
        slices = [i for i in range(len(neurons)) if i%12 == 0]
        slices.append(len(neurons) + 1)
        print(slices)

        for model, label in zip(models, labels):
            # fit model
            model.fit(selected_trials, iterations=30)
            # transform data based on fitted model
            data_transformed = model.transform(selected_trials)

            if len(neurons) <= 2:
                for neuronID, index in zip(neurons, range(len(neurons))):
                        visualize_single_neurons(selected_trials, index, neuronID, data_transformed, label)
            else:
                visualize_multiple_neurons(slices, model, label, neurons, data_transformed, selected_trials)
            # plt.show(block=True)
        


neuron_list = np.asarray([6])
load_spikedata(path_to_data, neuron_list)
