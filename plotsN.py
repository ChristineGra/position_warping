import os

import math
import numpy as np
import matplotlib.pyplot as plt

from affinewarp import SpikeData
from affinewarp import ShiftWarping, PiecewiseWarping
from affinewarp.crossval import heldout_transform

import pickle


for model, label in zip(models, ('shift', 'linear', 'pwise-1', 'pwise-2', 'pwise-3')):

    filename = "validated_alignments" + str(label)
    pickle_in = open(filename, 'rb')
    validated_alignments = pickle.load(pickle_in)

    # NOTE: various preprocessing and normalizations schemes (z-scoring,
    # square-root-transforming the spike counts, etc.) could be tried here.

    # Create figure.

    fig, axes = plt.subplots(len(neurons_to_plot), 2, figsize=(9.5, 6))

    # First column, raw data.
    plot_column(
        axes[:, 0], data
    )

    # Second column, re-sorted trials by warping function.
    # plot_column(
    #     axes[:, 1],
    #     data.reorder_trials(model.argsort_warps())
    # )

    # Third column, shifted alignment.
    plot_column(
        axes[:, 1],
        validated_alignments
    )

    fig.suptitle("Data")
    axes[0, 0].set_title("raw data")
    # axes[0, 1].set_title("sorted by warp (" + label + " warp)")
    axes[0, 1].set_title("aligned by model (" + label + " warp)")

    for index, axis in enumerate(axes[:, 0]):
        axis.set_ylabel("n. " + str(neurons_to_plot[index]))
    fig.tight_layout()
    fig.subplots_adjust(hspace=.9, top=0.9)

# TODO save plots
plt.show(block=True)
