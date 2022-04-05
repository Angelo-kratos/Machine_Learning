import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

##DATASETS
x, y = make_regression(n_samples=100, n_features=1, noise=10)
y = y.reshape(y.shape[0], 1)
data2 = y
data1 = np.hstack((x, np.ones(x.shape)))
plt.scatter(x,y)
plt.show()

def matrice_produit(data1, data2):
    col_A = data1.shape[1]
    line_B = data2.shape[0]
    c = np.zeros((data1.shape[0], data2.shape[1]))

    if col_A == line_B:
        pro_list = []
        for line_A in range(data1.shape[0]):
            for col_B in range(data2.shape[1]):
                for col_line in range(col_A):
                    c = np.sum(data1[line_A, col_line] * data2[col_line, col_B])
                    pro_list.append(c)
        rev = data1.shape[0] * data2.shape[1]
        res_list = []

        ##resultat des operations multiplication
        for index in range(rev - 1):
            somme = 0
            for compter in range(col_A):
                somme = somme + pro_list[compter]
                pro_list.remove(pro_list[compter])
            res_list.append(somme)

        ###somme des operations
        somme2 = 0
        for compter in range(col_A):
            somme2 = somme2 + pro_list[compter]
        res_list.append(somme2)
        c = np.zeros((data1.shape[0], data2.shape[1]))

        ##Classemant des resultat dans un tableau numpy
        for a in range(data1.shape[0]):
            for b in range(data2.shape[1]):
                c[a, b] = res_list[0]
                res_list.remove(res_list[0])

        return c
    else:
        print("Error")

##INITIALISATION
poids = np.random.randn(2,1)
print(poids, poids.shape)

#MODEL
def fonction(data1, poids):
    return data1.dot(poids)
print(fonction(data1, poids))
plt.figure()
plt.plot(x,fonction(data1,poids))
plt.scatter(x,y)
plt.show()

#fonction = fonction(data1, poids)


##FONCTION COUT
def loss_fonction(data1, data2, poids):
    m = len(data2)
    loss = 0
    for compter in range(len(data2)):
        loss = loss + (1 / 2*m) * (fonction(data1, poids)[compter, 0] - data2[compter, 0])**2
    return loss
print(loss_fonction(data1,data2, poids))
#print(loss)

##TRASPOSEE D'UNE MATRICE
def tranposition(data1):
    matrice_trans = np.zeros((data1.shape[1], data1.shape[0]))
    for line in range(data1.shape[0]):
        for col in range(data1.shape[1]):
            matrice_trans[col, line] = data1[line, col]
    return matrice_trans
transpose = tranposition(data1)

def grad(data1, data2, poids):
    m = len(data2)
    return (1 / m)*(transpose.dot(fonction(data1, poids)-data2)) #((1 / m) * (matrice_produit(transpose, fonction(data1,poids) - data2)))
#grad = grad(transpose,fonction, data2)
#print(grad)

def descent_gradient(data2, poids,grad,learning_rate, traitement):

    for repet in range(traitement):
        poids = poids - (learning_rate * grad(data1, data2, poids))
    return poids

poids_final = descent_gradient(data2, poids,grad,learning_rate=0.01, traitement=1000)
#print(poids)

predict = fonction(data1, poids_final)
plt.figure()
plt.scatter(x,y)
plt.plot(x, predict, c='r')
plt.show()