"""
@author: @carolinepsantos
         @rasoviti
"""

from scipy.optimize import differential_evolution
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import numpy as np

def rastrigin(X):
    return 10*len(X) + sum([(x**2 - 10 * np.cos(2 * math.pi * x)) for x in X])

limites = [(-5.12,5.12),(-5.12,5.12), (-5.12,5.12), (-5.12,5.12)]
resultado = differential_evolution(rastrigin, limites)
print(resultado.x)
print(resultado.fun)

print('Teste')
limites = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
resultado = differential_evolution(rastrigin, limites)
print(resultado.x)
print(resultado.fun)

