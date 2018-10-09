
# coding: utf-8

# In[2]:


print('Aula 04 - Regressao')


# In[3]:


# Exploração dos dados da Base de Dados: Diabetes
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
diabetes.keys()
print(diabetes.DESCR)


# In[5]:


import pandas as pd

tabela = pd.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela.head(10)


# In[6]:


# Taxa
tabela['Taxa'] = diabetes.target
tabela.head(10)


# In[8]:


# Escolha da melhor caracteristica que representa a taxa
import matplotlib.pyplot as plt

# Gráficos
# 1) Idade X Taxa 
plt.scatter(tabela.age, tabela.Taxa)
plt.xlabel('Idade')
plt.ylabel('Taxa')
plt.show()


# In[9]:


# 2) Sexo X Taxa 
plt.scatter(tabela.sex, tabela.Taxa)
plt.xlabel('Sexo')
plt.ylabel('Taxa')
plt.show()


# In[11]:


# 3) Indice de Massa Corporal X Taxa 
plt.scatter(tabela.bmi, tabela.Taxa)
plt.xlabel('Indice de Massa Corporal')
plt.ylabel('Taxa')
plt.show()


# In[12]:


# 4) Pressao Arterial Sanguinea X Taxa 
plt.scatter(tabela.bp, tabela.Taxa)
plt.xlabel('Pressao Arterial Sanguinea')
plt.ylabel('Taxa')
plt.show()


# In[13]:


# 5) S1 X Taxa 
plt.scatter(tabela.s1, tabela.Taxa)
plt.xlabel('S1')
plt.ylabel('Taxa')
plt.show()


# In[17]:


# 6) S2 X Taxa 
plt.scatter(tabela.s2, tabela.Taxa)
plt.xlabel('S2')
plt.ylabel('Taxa')
plt.show()


# In[18]:


# 6) S3 X Taxa 
plt.scatter(tabela.s3, tabela.Taxa)
plt.xlabel('S3')
plt.ylabel('Taxa')
plt.show()


# In[19]:


# 7) S4 X Taxa 
plt.scatter(tabela.s4, tabela.Taxa)
plt.xlabel('S4')
plt.ylabel('Taxa')
plt.show()


# In[20]:


# 8) S5 X Taxa 
plt.scatter(tabela.s5, tabela.Taxa)
plt.xlabel('S5')
plt.ylabel('Taxa')
plt.show()


# In[21]:


# 9) S6 X Taxa 
plt.scatter(tabela.s6, tabela.Taxa)
plt.xlabel('S6')
plt.ylabel('Taxa')
plt.show()


# In[22]:


# Correlação

tabela.corr()


# In[ ]:


# Com base nos gráficos e na tabela de correlação, foram analisados dois casos:
# - utilizando as duas características cujo o resultado da correlação foi positiva e próxima a 1
# - utilizando duas caracteristicas onde uma tem seu valor positivo próxima a 1 e a outra o valor negativo próximo a 1

