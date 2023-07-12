#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import random


# In[32]:


# Generates and visualizes a scatter plot of two sets of 50 random data points each. [randn=from normal distribution.]
# x1 and x2 represent the 2 axes
x1 = np.random.randn(50)
x2 = np.random.randn(50)
print(x1)
print(x2)
plt.scatter(x1,x2)
plt.xlabel("X1")
plt.ylabel("X2")
plt.text(-2, 2.8, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')

plt.title('Scatter Plot for Random Values')
plt.ylim(-3,3)
plt.xlim(-3,3)
plt.savefig('Scatter Plot for Random Values', bbox_inches='tight')


# In[33]:


# Mean and standard deviation
mean1 = np.mean(x1)
mean2 = np.mean(x2)
sd1 = np.std(x1)
sd2 = np.std(x2)
print(mean1,sd1)
print(mean2,sd2)


# In[34]:


# 2x2 random matrix with elements sampled from a uniform distribution (range [0, 1]).
random_matrix = np.random.rand(2,2)
print(random_matrix)


# In[35]:


# Converts the lists x1 and x2 into NumPy arrays, then creates a data matrix by column stacking the arrays.
x1_array = np.array(x1)
x2_array = np.array(x2)

data_matrix = np.column_stack((x1_array, x2_array))


# In[36]:


print(data_matrix.shape)


# In[37]:


print(data_matrix)


# **STEP 1: Reshape the data by multiplication with a random matrix.** To visualize the linear transformation, click on the link : https://www.geogebra.org/m/YCZa8TAH?fbclid=IwAR3LK2VOQooxb-qzH-LLhlyzVEhi8Qsb8Of_V1zeiqWb3FcjYibVPbL0NGo

# In[38]:


transformed_data_by_random_matrix = data_matrix @ random_matrix
print(transformed_data_by_random_matrix.shape)

# first column [:,0] as x-axis value. similarly 2nd column as y-axis
plt.scatter(transformed_data_by_random_matrix[:,0], transformed_data_by_random_matrix[:,1])
plt.xlabel("X1-transformed")
plt.ylabel("X2-transformed")
plt.ylim(-3,3)
plt.xlim(-3,3)
plt.text(-1.5, 2.8, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')

plt.title('Scatter Plot for transformed Random Values')
plt.savefig('transformed Random Values', bbox_inches='tight')


# **STEP 2: Calculate covariance matrix** The covariance matrix provides insights into the relationships between different dimensions of the transformed_data_by_random_matrix.

# In[39]:


# See if elements in main diagonal are max and other diagonal are min. If not, we have to proceed further .
covariance_matrix = np.cov(transformed_data_by_random_matrix.T)
covariance_matrix


# In[40]:


#same as above
covariance_matrix = np.cov(np.transpose(transformed_data_by_random_matrix))
print(covariance_matrix)


# **STEP 3: Eigen values and vector calculation**

# In[41]:


# The eigenvalues represent the variance explained by each principal component (magnitude of new feature space)
# The eigenvectors define the directions of the principal components. (direction of new feature space)
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
print(eigen_values)
print(eigen_vectors)


# In[42]:


# Just To check if the eigenvalues really represent the variance, lets find out the proportion of variance for each components
proportion1 = eigen_values[0] / np.sum(eigen_values)
print(proportion1*100)
proportion2 = eigen_values[1] / np.sum(eigen_values)
print(proportion2*100)


# **STEP 4: Transform the data with the selected eigen vector.**

# In[43]:


#first write the eigenvectors row wise (do the transpose, from column wise to row wise)
arranged_eigen_vector = np.transpose(eigen_vectors)
print(arranged_eigen_vector.shape)
print(arranged_eigen_vector)

#also transpose the (50,2) data matrix to (2,50)
transposed_data_matrix = transformed_data_by_random_matrix.T
print(transposed_data_matrix.shape)


# In[44]:


#Project data into first principal component (by matrix multiplication)
new_data = np.dot(arranged_eigen_vector,transposed_data_matrix)
print(new_data.shape)
#print(new_data)


# In[45]:


#transpose back to (50,2)
trans_new_data=new_data.T
print(trans_new_data.shape)


# In[46]:


# again see if the elements in main diagonal are max
covariance_of_new_data = np.cov(new_data)
print(covariance_of_new_data)


# In[47]:


#lets select the best eigen vectors as specified in the step 4

best_eigen_vector = np.transpose(eigen_vectors[:,0])
print(best_eigen_vector.shape)
print(best_eigen_vector)

new_new_data = np.dot(best_eigen_vector,transposed_data_matrix)
print(new_new_data.shape)
print(new_new_data)


# **STEP 6: plot the new data**

# In[63]:


plt.scatter(new_data[0,:], new_data[1,:] )
plt.ylim(-3,3)


# In[64]:


zeros = np.zeros(50)
plt.scatter(new_data[1,:],zeros)
plt.xlabel("X1_after_PCA")
plt.ylabel("X2_after_PCA")
plt.title('Scatter Plot After PCA')
plt.text(-0.8, 0.9, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.grid()
plt.savefig('scatter plot after PCA', bbox_inches='tight')


# In[50]:


print(new_data.shape)
print(new_data)


# In[ ]:





# In[ ]:




