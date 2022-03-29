import numpy as np
import matplotlib.pyplot as plt
from math import *
from random import *
from sklearn.datasets import make_blobs

##donné d'entrainement
x,y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))
data1 = x
data2 = y

def produit_matriciel(data1, data2):
    count_line_data1, count_line_data2, n = 0, 0, 0
    c = np.zeros((data1.shape[0], data2.shape[1]))
    list = []
    # n = nombre de colene de premier matrice et ligne de seconde
    ## condition si le produit matriciel est possible
    if data1.shape[1] == data2.shape[0]:
        line_col = data1.shape[1]
        while count_line_data1 < data1.shape[0]:
            while count_line_data2 < data2.shape[1]:
                while n < line_col:
                    c[count_line_data1, count_line_data2] = data1[count_line_data1, n] * data2[n, count_line_data2]
                    list.append(c[count_line_data1, count_line_data2])
                    n += 1
                if n == line_col:
                    count_line_data2 += 1
                    n = 0
            if count_line_data2 == data2.shape[1]:
                count_line_data1 += 1
                count_line_data2 = 0
                n = 0

        index = 0
        line_tabl = 0
        s = int(len(list) / 2)
        list1 = []  ##liste des resultat de chaque produit
        while line_tabl <= s - 1:
            index = 0
            first = 0
            while index < data1.shape[1]:
                first = first + list[0]
                list.remove(list[0])
                index += 1
            # print(first)
            list1.append(first)
            line_tabl += 1
            list2 = []  ## liste classer par ligne

        for colone in range(data1.shape[0]):
            list2.append(list1[:data2.shape[1]])
            list1 = list1[data2.shape[1]:]
        # print(list2)
        tabl_matrice = np.array(list2)  ##resultat finale Matrice
        # print(tabl_matrice)

    else:
        print("Error")
    # print(list1)
    return tabl_matrice

data1_multiply_poids = produit_matriciel(data1, poids) ##calcule produit matriciel w * x
#print(data1_multiply_poids)

##iniialisation
def initialisation(data1):
    poids = np.random.randn(data1.shape[1], 1)
    biais = np.random.randn(1)
    return (poids, biais)


##fonction
def fonction(data1_multiply_poids,biais):
    agregation = data1_multiply_poids + biais
    activation = np.zeros((agregation.shape[0], 1))
    for count in len(agregation):
        activation[count, 0] = (1 / (1 + exp(-agregation[count, 0])))
    return activation

def loss_fonction(data2, activation): ##fonction coût
    log_loss = 0
    for count in range(len(data2)):
        log_loss = log_loss + (- (1 / len(data2)) * (y[count, 0] * log(activation[count, 0]) + (1 - y[count, 0]) * log(1 - activation[count, 0])))
    return log_loss

##np.dot(data1.T, activation - data2)
##Transposer d'une matrice
trans_data1 = np.array((data1.shape[1],data1.shape[0]))
for line_trans in range(data1.shape[0]):
    for col_trans in range(data1.shape[1]):
        trans_data1[col_trans, line_trans] = data1[line_trans, col_trans]
data1_x_atcv_data2 = produit_matriciel(trans_data1, activation - data2)

##variation des parametres
def params_variation(data2, activation):
    d_biais = 0
    d_poids = (1 / len(data1)) * (data1_x_atcv_data2)
    for count in range(len(data2)):
        d_biais = d_biais + (1 / len(data1)) * (activation[count, 0] - data2[count, 0])
    return (d_poids, d_biais)

def update(biais, poids):
    biais = bias + d_biais






















##fonction cout
""" #l

##variation des w,b
###

#mis à jour des valeur de w et b  """








