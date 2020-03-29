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
import argparse


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

        
def visualize_multiple_neurons(slices, model, label, neurons, data_transformed, selected_trials, path_save):
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
        

def visualize_single_neurons(selected_trials, index, neuronID, data_transformed, label, path_save):
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

        

def load_spikedata(path, path_save, neuron_list=None, model_selected=None, smoothreg=5., warpreg=.3, l2reg=1e-7, maxlagreg=.3, trial_selection=[0, 60], plot_knots=False):
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
        plot_knots (optional): boolean to determine whether the fitted warping functions (piecewise model) or shifts per trial (shift model) will be plotted and printed to the terminal
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

        # filter trials so that only selected neurons are taken into account
        selected_trials = selected_trials[selected_neurons[0]]
        # switch axes so that data can be used in model.fit: (num trials, num timepoints, num units)
        selected_trials = np.swapaxes(selected_trials, axis1=0, axis2=2)

        # create model
        shift_model = ShiftWarping(maxlag=maxlagreg, smoothness_reg_scale=smoothreg, warp_reg_scale=warpreg, l2_reg_scale=l2reg)
        linear_model = PiecewiseWarping(n_knots=0, warp_reg_scale=warpreg, smoothness_reg_scale=smoothreg)
        p1_model = PiecewiseWarping(n_knots=1, warp_reg_scale=warpreg, smoothness_reg_scale=smoothreg)
        p2_model = PiecewiseWarping(n_knots=2, warp_reg_scale=warpreg, smoothness_reg_scale=smoothreg)
        if model_selected == None:
                models = [shift_model, linear_model, p1_model, p2_model]
                labels = ["shift", "linear", "piecewise-1", "piecewise-2"]
        elif model_selected == "shift":
                models = [shift_model]
                labels = ["shift"]
        elif model_selected == "linear":
                models = [linear_model]
                labels = ["linear"]
        elif model_selected == "piecewise-1":
                models = [p1_model]
                labels = ["piecewise-1"]
        elif model_selected == "piecewise-2":
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

        for model, label in zip(models, labels):
            # fit model
            model.fit(selected_trials, iterations=30)

            # if knots should be displayed
            if plot_knots == True:
                    # print shifts for each trial if shift model is used
                    if label == "shift":
                        model_shifts = model.shifts
                        print("shifts for every trial: ", model_shifts)
                        
                    # plot and print warping functions for every trial
                    else:
                        # extract knot coordinated
                        x_knots = model.x_knots.T
                        y_knots = model.y_knots.T

                        # print knot coordinates
                        for ind_trial in range(len(x_knots[0])):
                                print("(x, y) coordinates of knots for trial " + str(trial_selection[0] + ind_trial) + ":")
                                for ind_knot in range(len(x_knots)):
                                        print("(" + str(x_knots[ind_knot][ind_trial]) + ", " + str(y_knots[ind_knot][ind_trial]) + ")")
                        # plot warping functions
                        plt.figure()
                        plt.plot(x_knots, y_knots)
                        plt.title("warping functions for model  " + label)
                        plt.xlabel("relative position in trial (0=start of trial, 1=end of trial)")
                        plt.ylabel("y position")
                        plt.savefig(os.path.join(path_save, str(label) + "knot_plot"))
                        
            # transform data based on fitted model
            data_transformed = model.transform(selected_trials)

            if len(neurons) <= 2:
                for neuronID, index in zip(neurons, range(len(neurons))):
                        visualize_single_neurons(selected_trials, index, neuronID, data_transformed, label, path_save)
            else:
                visualize_multiple_neurons(slices, model, label, neurons, data_transformed, selected_trials, path_save)
        


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='position warping for place cells', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('path_to_data', metavar='path_to_data', type=str, help='path to data')
        parser.add_argument('path_save', metavar='path_save', type=str, help='path to folder to save data in')
        parser.add_argument('--neuron_list', nargs="+", metavar='neuron_list', type=int, help='list of preselected neurons, if not specified, neurons will be selected based on SSI, meanFR, muCC and cellType ',default=None, required=False)
        parser.add_argument('--model_selected', metavar='model_selected', type=str, help='selected model, one of "shift", "linear", "piecewise-1" or "piecewise-2", if not specified, all models will be used',default=None, required=False)
        parser.add_argument('--smoothreg', metavar='smoothreg', type=float, help='float between 0 and 10', default=5., required=False)
        parser.add_argument('--warpreg', metavar='warpreg', type=float, help='float between 0 and 1', default=.3, required=False)
        parser.add_argument('--l2reg', metavar='l2reg', type=float, help='float between 1e-7 and 0', default=1e-7, required=False)
        parser.add_argument('--maxlagreg', metavar='maxlagreg', type=float, help='float between 0 and 0.5', default=.3, required=False)
        parser.add_argument('--trial_selection', metavar='trial_selection', nargs="+", type=int, help='list of indices of first and (last+1) trials to model, to select all use number of trials +1 as second entry', default=[0,60], required=False)
        parser.add_argument('--plot_knots', metavar='plot_knots', type=bool, help='boolean to determine whether the fitted warping functions (piecewise model) or shifts per trial (shift model) will be plotted and printed to the terminal', default=False, required=False)



        # path_to_data = '/storage2/perentos/data/recordings/NP46/NP46_2019-12-02_18-47-02/processed/place_cell_tunings.mat'
        # neuron_list = np.asarray([6, 511, 514])
        # path_save = os.path.join("plots")
        args = parser.parse_args()
        load_spikedata(args.path_to_data, args.path_save, args.neuron_list, args.model_selected, args.smoothreg, args.warpreg, args.l2reg, args.maxlagreg, args.trial_selection, args.plot_knots)
