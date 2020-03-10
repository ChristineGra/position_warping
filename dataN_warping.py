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
        Load npz file and display data 
        data have to have format [trials, spiketimes, neuron_ids, tmin, tmax, sniff_onsets]
        but sniff_onset not secessary
        """
        
        # load .mat file
        data = loadmat(path)
        
        # data contains __header__, __version__, __globals__, tun
        # data[' __header__'] contains a lot of numbers
        # data['tun'] contains all the data
        
        # print(np.asarray(data['tun']['cellID']).shape)  #cellTunings: shape (1, 324)
        # print(np.asarray(data['tun']['cellTunings'][0,0]).shape) # shape (119, 120)
        # print( data['tun']['cellID'][0, 0])

        celltunings = np.asarray(data['tun']['cellTunings'][0])  # for each of 324 neurons: position (119)  x trial (120) (?)

        selected_trials = []
        # only select first 60 trials
        for index in range(len(celltunings)):
                celltunings_selected = np.asarray(celltunings[index][:, :60])
                selected_trials.append(celltunings_selected)
                if index == 0:
                        print(celltunings_selected.shape)

        print(np.asarray(selected_trials).shape)  # shape (324, 119, 60) -> neurons x spike positions x trial

        # select neurons that fulfill criteria
        neuronIDs = np.asarray(data['tun']['cellID'][0])  # ID for each of 324 neurons

        SSI = data['tun']['SSI']
        meanFR = data['tun']['meanFR']
        muCC = data['tun']['muCC']

        SSI_lower_limit = 0.2
        mean_FR_lower_limit = 0.2
        mean_FR_upper_limit = 5
        muCC_lower_limit = 0.6

        
# load data from file
spikedata_raw = load_spikedata(path_to_data)
