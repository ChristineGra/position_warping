import numpy as np
import matplotlib.pyplot as plt
from affinewarp.datasets import jittered_data
from affinewarp import ShiftWarping, PiecewiseWarping
from affinewarp import SpikeData
from affinewarp.crossval import heldout_transform
import os
from scipy.io import loadmat
import pprint


path_to_data = '/storage2/perentos/data/recordings/NP46/NP46_2019-12-02_18-47-02/processed/place_cell_tunings.mat'

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

        # extract cell tunings for all neurons
        celltunings = np.asarray(data['tun']['cellTunings'][0])  # for each of 324 neurons: position (119)  x trial (120) (?)
        print(celltunings[0].shape)

        selected_trials = []
        # only select first 60 trials
        for index in range(len(celltunings)):
                celltunings_selected = np.asarray(celltunings[index][:,:60])  # TODO: do I select position or trial?????
                selected_trials.append(celltunings_selected)

        selected_trials = np.asarray(selected_trials)  # shape (324, 119, 60) -> neurons x spike positions x trial
        print(selected_trials.shape)
        
        # print(selected_trials[0, 0, :])
        
        # select neurons
        neuronIDs = np.asarray(data['tun']['cellID'][0])  # ID for each of 324 neurons

        SSI = data['tun']['SSI']
        meanFR = data['tun']['meanFR']
        muCC = data['tun']['muCC']
        criteria = [SSI, meanFR, muCC]
        print("SSI shape: ", np.asarray(SSI).shape)

        # criteria
        SSI_lower_limit = 0.2
        mean_FR_lower_limit = 0.2
        mean_FR_upper_limit = 5
        muCC_lower_limit = 0.6

        # select neurons that fulfill criteria
        selected_neurons = np.where((criteria[0] > SSI_lower_limit) & (criteria[1] > mean_FR_lower_limit) & \
                                    (criteria[1] < mean_FR_upper_limit) & (criteria[2] > muCC_lower_limit), True, False)
        print("number of selected neurons: ", len([i for i in selected_neurons[0] if i == True]))
        print("shape of selecetd neurons: ", selected_neurons.shape)
        
        # filter trials so that only selected neurons are taken into account
        selected_trials = selected_trials[selected_neurons[0]]

        # filter neuron IDs so that only selected neurons are taken into account
        neurons = neuronIDs[selected_neurons[0]]
        print("Neurons list shape: ", neurons.shape)
        
        # copy the data into final data structure
        result = [[], [], []]

        for i in range(neurons.shape[0]):
                # for every neuron
                neuronID = neurons[i][0][0]

                for trial in range(selected_trials.shape[2]):  # change to ...shape[2] if trial is in last dimension
                    # for every trial
                        trialID = trial

                        for position_index in range(selected_trials.shape[1]):  # change to ...shape[1] if trial is in last dimension
                                   # find position
                                   position = selected_trials[i, position_index, trial]  # exchange last two entries if trial is in last dimension
                                   if position == 0:
                                           continue
                                   else:
                                           # make entry in dataset
                                           result[0].append(neuronID)
                                           result[1].append(trialID)
                                           result[2].append(position)

        # check entries
        print(np.asarray(result).shape)
        # print("neurons: ", result[0][-200:])
        # print("trials: ", result[1][-200:])
        # print("positions: ", result[2][-200:])

        min_pos = min(result[2])
        max_pos = max(result[2])
        print(min_pos)
        print(max_pos)

        # save dataset
        path_datasets_folder = "datasets"
        np.save(os.path.join(path_datasets_folder, "dataset_NP46_2019-12-02_18-47-02_new.npy"), result)

# load data from file
spikedata_raw = load_spikedata(path_to_data)
