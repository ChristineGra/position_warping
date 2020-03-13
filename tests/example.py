import os

import math
import numpy as np
import matplotlib.pyplot as plt
import bisect


data = dict(np.load("/storage2/perentos/code/python/conda/affinewarp/affinewarp/examples/olfaction/pinene_data.npz"))
print(data["neuron_ids"][:1000])
