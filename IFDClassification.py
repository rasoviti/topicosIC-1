# k nearest neighbors
# git: @carolinepsantos
# git: @rasoviti

#iris dataset: Conjunto de dados consistem em 3 tipos diferentes de pétalas de íris (Setosa, Versicolour e Virginica) 
#e comprimento da sépala, armazenados em um numpy 150x4.

# dados
from sklearn import datasets

# plot
import matplotlib.pyplot as plt

# treinamento
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# classificação
from sklearn.neighbors import KNeighborsClassifier

# validação
from sklearn.metrics import classification_report, confusion_matrix 

iris = datasets.load_iris()
X = iris.data[:,2:]  # as duas ultimas caracteristicas
y = iris.target #classificacao

#0 Comprimento da sépala; 1 Largura da sépala; 2 comprimento da pétala; Largura da pétala 
#setosa, versicolor, virginica  
plt.subplots()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Comprimento Petala')
plt.ylabel('Largura Petala')
plt.grid(True)
plt.show()

scaler = StandardScaler()
X = iris.data[:,2:]
#iris dataset: Conjunto de dados consistem em 3 tipos diferentes de pétalas de íris (Setosa, Versicolour e Virginica) 
#e comprimento da sépala, armazenados em um numpy 150x4.
              
#dados de treinamento 'até 40' de cada classe
yt=numpy.concatenate([y[:40], y[51:90], y[101:140]])
xt = numpy.concatenate([X[:40,:], X[51:90,:], X[101:140,:]])
scaler.fit(xt, yt)  

#validacão com o restante dos dados
yv=numpy.concatenate([y[40:50], y[90:100], y[140:150]])
xv = numpy.concatenate([X[40:50,:], X[90:100,:], X[140:150,:]])

xt=scaler.transform(xt)
xv=scaler.transform(xv)


classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(xt, yt)

yp = classifier.predict(xv)

print(yp)
print(yv)

 
print(confusion_matrix(yv,yp))  
print(classification_report(yv,yp))