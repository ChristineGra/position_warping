# position_warping

This project was about aligning neuron activity from different trials to identify place cells with higher precision.

Place cells in the Hippocampus are instruments for spatial representation and self-location. Together with entorhinal grid cells, place cells form the basis for representation of places, routes and associated experiences as well as memory. Since both of these cell types are representing internal computations, investigating these cells could improve the understanding of cortical network dynamics [1].
Place cell identification therefore plays a vital role in Hippocampus and learning research. It can be deduced how animals represent the environment and react to reward [1]. 

Since the recordings are done by injecting electrodes into the region of interest and measuring single neurons in the head-fixed mice, followed by spike sorting, inaccuracies could be present in the data. Another problem is that a mouse does not necessarily run with the same speed in every trial, so the positional data could have a slight deviation. To account for this, Alex Williams et al. developed simple warping methods that are understandable and interpretable while still performing well, compared to other more complex warping methods [2]. 

In the report I explain how the warping algorithm works and show results of applying warping on place cell data, as well as evaluating what influences different regularizations have on the algorithm.

The functions that are used to align the spikes could be evaluated with respect to the experiments, e.g. do the knots of a piecewise linear function correspond with stopping of the mouse or could the slope of the function correspond with the speed of the mouse?



### Implementation

To implement position warping, I developed a Python script to fit the warping functions on experimental data. The warping functions were provided by a Python library found on GitHub (https://github.com/ahwillia/affinewarp). The library includes shift warping and piecewise warping functions that are applied to the datasets. Apart from the data, regularization parameters are passed when fitting the model. For spike data, a data structure is available to store the data in order to work with the warping functions, but for continuous data, the algorithms can be applied without preprocessing. 

My implementation includes flexible selection of the model and the respective hyperparameters, as well as the option to select a subset of neurons or a subset of trials to be taken into account when fitting the model. Furthermore, the aligned data are visualized and stored automatically.

The program is fully documented and can be called from the command line, providing additional information, however, you can find a user guide with examples at the end of the README file. The project can be found as an open source repository on GitHub (https://github.com/ChristineGra/position_warping) together with the development history.



## User Guide

1. Prepare data so that your dataset is one .mat le containing a matrix of continuous data with dimensions (spike time or position x trial x neuron ID) and a list of neuron IDs

2. Install Python (optimally in a conda environment)

3. Install the affinewarp library from GitHub (https://github.com/ahwillia/affinewarp) by following the instructions in the README file (may have to adjust the requirements file with updated library names)

4. Place script from https://github.com/ChristineGra/position warping/blob/master/data_exploration.py
in your directory

5. Call script with path to your dataset and path to folder to save in as input parameter:
    **Python data_exploration.py path_to_data path_save**
    Optional parameters are a list of neurons to t the model on, a specific model, a list of trials to select and hyperparameters for the models. Further information on these parameters can be found with typing
    **Python data_exploration.py -h**

  

  If you want to use other data formats, different models or other visualizations, the examples on the GitHub page (https://github.com/ahwillia/affinewarp) could be a good source to extend/ modify the analysis or write your own script. I highly recommend using nested crossvalidation (also implemented in the affinewarp library) when searching for the best model parameters.



#### Sources

[1] Edvard I Moser, Emilio Kropff, and May-Britt Moser. Place cells, grid cells, and the brain's
 spatial representation system. *Annu. Rev. Neurosci., 31:69-89, 2008.*

[2] Alex H Williams, Ben Poole, Niru Maheswaranathan, Ashesh K Dhawale, Tucker Fisher, Christopher D Wilson, David H Brann, Eric M Trautmann, Stephen Ryu, Roman Shusterman, et al. Discovering precise temporal patterns in large-scale neural recordings through robust and interpretable time warping. *Neuron, 105(2):246-259, 2020.*