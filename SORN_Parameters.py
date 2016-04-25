from pylab import *
from scipy import sparse
from scipy.sparse import csc_matrix
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import normalize
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
logistic = linear_model.LogisticRegression()


############################
# Parameters for SORN code #
############################


# How many different letters exist in the sequence:
letters = [6]
Input = letters[0]


# Number of iterations:
max_iteration = 1
S = zeros((max_iteration))


# Number of timesteps with learning:
plastic_timesteps = [50000]
T_plastic = plastic_timesteps[0]


# Networksize:
networksize = [200]
N_E = networksize[0]


N_I = int(0.2 * N_E)
N_U = int(0.05 * N_E)


# Sequence length:
sequence_length = [20]
inputsize = sequence_length[0]


# Anzahl der Timesteps ohne Lernen (eta_STDP = 0):
T_train = int(0.1*T_plastic)
"""T_train = int(3*size(inputsize))"""
Timesteps = T_plastic + 2*T_train