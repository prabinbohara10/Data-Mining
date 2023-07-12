#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


# **STEP 1 : Load the dataset into a dataframe and analyze the data**

# In[2]:


#specify path of csv file relative to "/content/drive" directory.
file_path = './dataset/Cancer_Data.csv'
df = pd.read_csv(file_path)


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[5]:


# Print the column names
print(df.columns)


# In[6]:


# statistics computed for each column in the df
summary_stats = df.describe()
print(summary_stats)


# In[7]:


# see the unique values in diagnosis column
unique_diagnosis = df['diagnosis'].unique()
print(unique_diagnosis)


# Looks like diagnosis is our target column. Lets assign X and y accordingly

# In[8]:


# Separate the data and target variables
X = df.drop(columns=['id', 'diagnosis', 'Unnamed: 32'])
y = df['diagnosis']

print("Data:")
print(X.shape)
print(X.head())
print()
print("Target:")
print(y.shape)
print(y.head())


# In[9]:


X.columns


# In[10]:


# Create a heatmap of the attribute correlations
# High correlation (+ve or -ve) by bright colors, low correlation by dull colors.
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), cmap='coolwarm', annot=True, annot_kws={'size': 5})
plt.title('Attribute Correlation Heatmap')
plt.text(8, -1.2, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.savefig('./plots/Corelation Matrix', bbox_inches='tight')
plt.show()


# **STEP 2: Preprocessing**

# In[11]:


# Perform label encoding on the 'diagnosis' target variable
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(y_encoded)


# In[12]:


# change the dataframe's diagnosis column
df['diagnosis'] = y_encoded


# In[13]:


df.head()


# In[14]:


# Standardizing everything to have a common ground so that magnitude of some feature's value doesn't affect the PCA.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)


# In[15]:


# Scaling of the whole dataframe.
df_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
# Define new X and y
X_train = df_scaled             
y_train = df['diagnosis']
print("shape of new input X",X_train.shape)
print("shape of new output Y",y_train.shape)


# In[16]:


df_scaled.head()


# In[17]:


# Assign different colors to the data points based on the 'diagnosis' column
# Used to visually distinguish 2 different categories in that column
colors = ['blue' if t == 0 else 'red' for t in df['diagnosis'].map({0: 'B', 1: 'M'})]




# In[18]:


X_train.cov()


# **STEP 3:CoVariance Matrix Calculation**

# In[19]:


# covariance matrix will be 30*30 since X=(569,30). [each 30 features has covariance with each 30. so 30*30 matrix]

covariance_matrix = np.cov(X_train, rowvar=False)                     
print("Covariance matrix shape:", covariance_matrix.shape)
attribute_labels = X.columns 
covariance_matrix = pd.DataFrame(covariance_matrix, columns=attribute_labels, index=attribute_labels)
print(covariance_matrix.head())


# In[20]:


# Visualize covariance matrix

plt.figure(figsize=(12, 8))
sns.heatmap(covariance_matrix, cmap='coolwarm', annot=True, annot_kws={'size': 5})
plt.title('Covariance Matrix on Processed Cancer Data')
plt.text(8, -1.2, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.savefig('./plots/Covariance Matrix Processed Cancer Data', bbox_inches='tight')
plt.show()


# ##### **STEP 4: Eigen vectors/values calculation**

# In[21]:


eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print("shape of eigenvalue:",eigenvalues.shape)
print("shape of eigen vectors:",eigenvectors.shape)
print("eigen values are",eigenvalues)
print("eigen vectors are",eigenvectors)


# **STEP 5: See the variance explained by each eigen values.**

# In[22]:


# Element wise array division to obtain a new array
# proportion_of_variance array provides insights into relative importance of each PC in capturing the overall variance in the dataset
proportion_of_variance = eigenvalues/sum(eigenvalues)
proportion_of_variance


# In[23]:


# Plot a scree plot to visualize how much variance is captured by which PC
cumulative_variance = np.cumsum(proportion_of_variance)
  
# range(1 to 30) on x axis, cummulative pov on y axis
plt.plot(range(1, len(proportion_of_variance) + 1), proportion_of_variance, marker='x', color='blue')
plt.plot(range(1, len(proportion_of_variance) + 1), cumulative_variance, marker='o', color='orange')                                                             
plt.xlabel('Principal Components(Eigen Vector)')
plt.ylabel('Proportion of Variance')
plt.title('Scree Plot')
legend_class = ["Explained Variance", "Cummulative Explained Variance"]
plt.legend(labels = legend_class)
plt.text(12, 0.9, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.savefig('./plots/Scree Plot', bbox_inches='tight')
plt.show()


# **STEP 6: Select the eigen vectors as needed to compute the final output** Formula: New data (Y) = Row feature vector * Row data

# In[24]:


# Transpose data into row wise
transposed_X_train = np.transpose(X_train)
print("shape of transposed x train is",transposed_X_train.shape)
# Transpose eigen vectors into row wise for row feature vector
eigenvectors_transposed=np.transpose(eigenvectors)


# In[25]:


# Dictionary to store the final Y(because we want to have multiple Y, each obtained by selecting our choice of combination of PC).
Y = {}
selected_components = [[0], [0, 1], [0,3], [2,3], [0,29]]                       # best PC, 2 best PCs, best+medium, medium+medium, best+worst
count=len(selected_components)

# Iterate over the selected components
for i in range (0,count):
  selected_eigenvectors = eigenvectors_transposed[selected_components[i]]       # select the combination of components from above list
  Y[i] = np.transpose(selected_eigenvectors @ transposed_X_train)               # projection
  print(Y[i].shape)


# In[26]:


# checking the nature of Y
print(type(Y))
print(Y.keys())
Y[1].shape
print(type(Y[1]))


# **STEP 7: Computing and visualizing covariance for the new data.**

# In[27]:


# Dictionary to store all covariance matrices(obtained by selecting our choice of combination of PC).
new_covariance_matrices_dictionary = {}
for i in range(0, count):
  new_covariance_matrix = np.cov(Y[i], rowvar=False)
  new_covariance_matrices_dictionary[i] = new_covariance_matrix                               # Combination:
  print("*Covariance matrix shape for combination",i,"is", new_covariance_matrix.shape)       # 0=best PC, 1=2 best PCs, 2= best+medium PCs
  print("-->Covariance matrix:",new_covariance_matrices_dictionary[i])                        # 3=medium+medium PCs, 4= best+worst PCs


# In[28]:


# Heatmap to visualize covariance matrix combination.
plt.imshow(new_covariance_matrices_dictionary[1], cmap='Reds', interpolation='nearest')  #instead of 'nearest', we can use 'bilinear','bicubic','spline16', 'spline36', 'hanning', 'hamming', 'hermite',etc..
plt.colorbar()
plt.title("New Covariance Matrix Combination 1 (2 best PCs)")
plt.show()


# In[29]:


#lets try bilinear,pixel coloring is smoother(linear interpolation between 4 nearest data points)
plt.imshow(new_covariance_matrices_dictionary[2], cmap='Reds', interpolation='bilinear')
plt.colorbar()
plt.title("New Covariance Matrix Combination 2 (Best and medium PCs)")
plt.show()


# In[30]:


#lets try bicubic,pixel coloring is again smoother(linear interpolation based on 16 nearest data points)
plt.imshow(new_covariance_matrices_dictionary[3], cmap='Reds', interpolation='bicubic')
plt.colorbar()
plt.title("New Covariance Matrix Combination 3 (Two medium PCs)")
plt.show()


# In[31]:


#lets try spline16,pixel coloring is even smoother(16 degree of polynomial used for interpolation, for flexibility)
plt.imshow(new_covariance_matrices_dictionary[4], cmap='Reds', interpolation='spline16')
plt.colorbar()
plt.title("New Covariance Matrix Combination 4 (Best and worst PCs)")
plt.show()


# **STEP 8:Visualize the final output**

# In[32]:


#Combination 0 = Y[0] = Best PC
#plt.ylim(-10,10)
#plt.xlim(-10,10)
varlegend= "Roll: 26,27,41"
plt.text(5,0.04,varlegend,color='red')
plt.title("1 best PC")
plt.xlabel('PC1')
plt.scatter(Y[0], np.zeros_like(Y[0]), c=y_train)   # colors = based on y_train


# In[33]:


# Combination 1 = Y[1] = 2 best PCs
#plt.ylim(-3,3)
#plt.xlim(-3,3)
varlegend= "Roll: 26,27,41"
plt.text(5,4,varlegend,color='red')
plt.title("2 best PCs")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(Y[2].iloc[:, 0], Y[2].iloc[:, 1], c=y_train)    # iloc used to access data


# In[34]:


# Combination 3 = Y[3] = 2 medium PCs
#plt.ylim(-3,3)
#plt.xlim(-3,3)
varlegend= "Roll: 26,27,41"
plt.text(2,4.5,varlegend,color='red')
plt.title("2 medium PCs")
plt.xlabel('PC3')
plt.ylabel('PC4')
plt.scatter(Y[3].iloc[:,0],Y[3].iloc[:,1],c=y_train)


# In[35]:


# Combination 4 = Y[4] = Best and worst PCs
plt.ylim(-3,3)
#plt.xlim(-3,3)
varlegend= "Roll: 26,27,41"
plt.text(5,2,varlegend,color='red')
plt.title("Best and worst PCs")
plt.xlabel('PC1')
plt.ylabel('PC30')
plt.scatter(Y[4].iloc[:,0],Y[4].iloc[:,1],c=y_train)


# **STEP 9: PCA using Library**

# In[36]:


from sklearn.decomposition import PCA
# Create instances of PCA with the desired number of components. (pca1 = 2components. pca2=so as to retain 95% info)
pca1 = PCA(n_components=2)
pca2 = PCA(0.95)

# Fit the PCA model to the data and transform the data
reduced_data1 = pca1.fit_transform(X_train)
reduced_data2 = pca2.fit_transform(X_train)

# Print the shape of the reduced data
print("Shape of reduced data:", reduced_data1.shape)
print("reduced through first",reduced_data1)
print("Shape of reduced data:", reduced_data2.shape)
print("reduced through second",reduced_data2)


# Here it can be seen that to retain 95% info, 10 principle components are required

# In[37]:


# Visualize the output for the first PCA done by library
#plt.ylim(-3,3)
#plt.xlim(-3,3)
varlegend= "Roll: 26,27,41"
plt.text(10,10,varlegend,color='red')
plt.title("Two best PCs-Using Library")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(reduced_data1.T[0],reduced_data1.T[1], c= y_train) # After applying transform method of PCA,Each row was represented as a sample. So need to transpose.


# In[38]:


# For the 2nd PCA done by library(to retain 95% info), 10 components were required! We will try to plot 3d scatter plot to visualize.
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
varlegend= "Roll: 26,27,41"
ax.text(-5,12.5,10,varlegend,color='red')                                       # 3d coordinates required to specify the legend position
ax.scatter(reduced_data2.T[0],reduced_data2.T[1],reduced_data2.T[2],c=y_train)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D Scatter Plot of 3 PCs')
plt.show()


# For the next step, we can train a model and compare the performance to see the impact of PCA we did so far.
