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


letters = [20]
Input = letters[0]                                                  # Wieviele verschiedene Buchstaben


max_iteration = 1                                                  # Anzahl der Durchlaeufe
S = zeros((max_iteration))                                              #  Matrix wo die scores gespeichert werden


plastic_timesteps = [50000]                                      # Anzahl der Timesteps mit Lernen
T_plastic = plastic_timesteps[0]                                   # T_plastic = plastic_times[rank%3] jeder dritte Rank


networksize = [400]
N_E = networksize[0]


sequence_length = [1000]
inputsize = sequence_length[rank]                                     # inputsize = input_size[rank//3]  die ersten drei Ranks, dann die naechsten 3 usw


T_train = int(3*inputsize)                                            # Anzahl der Timesteps ohne Lernen (eta_STDP = 0)
Timesteps = T_plastic + 2*T_train


N_I = int(0.2 * N_E)
N_U = int(0.05 * N_E)


for k in range(max_iteration):

    ## INPUT:
    # input vector u
    u = eye(inputsize)

    # W_eu
    W_eu_pre = zeros((N_E,Input))
    for i in range(Input):
        W_eu_pre[i*N_U:(i+1)*N_U, i] = 1

    W_eu = copy(W_eu_pre[:,0][:,newaxis])

    for a in range(inputsize-1):
        stopp = False
        iter = 0
        randomnumber = rand()
        while stopp == False:
            if randomnumber < (iter+1)*1.0/Input:
                W_eu = concatenate((W_eu,W_eu_pre[:,iter][:,newaxis]),1)
                stopp = True
            iter = iter + 1

    # Plasticity values
    H_IP = 0.1
    eta_STDP = 0.001
    eta_IP = 0.001

    T_e_max = 0.5
    T_i_max = 0.5

    Lambda = 10

    # T_e und T_i
    T_e = rand(N_E) * T_e_max
    T_i = rand(N_I) * T_i_max

    # x und y
    x = rand(N_E, Timesteps + 1)
    x[x >= 0.5] = 1
    x[x < 0.5] = 0

    y = rand(N_I)
    y[y >= 0.5] = 1
    y[y < 0.5] = 0

    # Matrix bei der die Diagonalelemente 0 sind und alle anderen 1
    MatrixDiag = ones((N_E,N_E))
    for i in range(N_E):
        for j in range(N_E):
            if i == j:
                MatrixDiag[i,j] = 0


    ## W_ee:
    lam = 1 - float(Lambda)/N_E

    cc = 1
    while cc > 0:
        cc = 0
        W_ee = rand(N_E, N_E)
        W_ee[W_ee < lam] = 0
        W_ee[W_ee > 0] = rand(sum(W_ee > 0))
        summe = np.sum(W_ee, axis = 1)
        for n in summe:
            if n == 0:
                cc = 1
                break

    # alle Diagonalelemente in W_ee auf null setzen
    W_ee = W_ee * MatrixDiag

    # Normalisierung W_ee
    W_ee = normalize(W_ee , norm='l1' ,  axis = 1 , copy=False)

    ## W_ei und W_ie:
    W_ei = rand(N_E, N_I)
    W_ie = rand(N_I, N_E)

    # Normalisierung W_ei
    W_ei = normalize(W_ei , norm='l1' ,  axis = 1 , copy=False)

    # Normalisierung W_ie
    W_ie = normalize(W_ie , norm='l1' ,  axis = 1 , copy=False)

    # Sparse Matrix
    W_ee = sparse.csc_matrix(W_ee)
    row = W_ee.indices
    col = np.repeat(arange(N_E),np.diff(W_ee.indptr))
    data = W_ee.data


    #SORN
    for t in range(Timesteps):

        if t >= T_plastic:
            eta_STDP = 0

        Rx = W_ee.dot(x[:,t]) - np.dot(W_ei, y) - T_e + np.dot(W_eu, u[:,t%inputsize])

        Rx[Rx >= 0] = 1
        Rx[Rx < 0] = 0
        x[:,t + 1] = Rx

        # STDP:
        data += eta_STDP * (x[row,t + 1] * x[col,t] - x[row,t] * x[col,t + 1])
        data[data < 0] = 0

        # IP:
        T_e = T_e + eta_IP * (x[:,t] - H_IP)

        # SN:
        row_sums = np.array(W_ee.sum(1))[:,0]
        data /= row_sums[row]

        Ry = np.dot(W_ie, x[:,t + 1]) - T_i

        Ry[Ry >= 0] = 1
        Ry[Ry < 0] = 0
        y = Ry


    #plt.matshow(x, aspect='auto',cmap=plt.get_cmap('Greys'))

    Y = zeros((Timesteps))
    for n in range(Timesteps):
        Y[n] = find(u[:,n%inputsize] == 1)[0]


    # Prediction
    X_train = x[:,T_plastic:T_plastic + T_train - 1].T
    y_train = Y[T_plastic:T_plastic + T_train - 1]
    X_test = x[:,T_plastic + T_train + 1:T_plastic + 2*T_train].T
    y_test = Y[T_plastic + T_train + 1:T_plastic + 2*T_train]

    S[k] = logistic.fit(X_train, y_train).score(X_test, y_test)


result = comm.gather(S)
if rank==0:
    print(result)

