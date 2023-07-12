import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import random
import sklearn

# **STEP 1: Load the iris flower dataset**

from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data    # X is 2d array, where row=samples, column=features
y = iris.target  # y is 1d array, representing class label for all samples
print(X.shape)
print(y.shape)
print(iris.feature_names)
print("Target names:", iris.target_names)
print(X)

# **STEP 2: Create a dataframe and see the data statistics**

# assign meaningful columns(feature_names) to feature values for 150 samples(iris.data) in the dataframe.
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
# add a new target column.
df['target']=iris.target
df.head(2)   # view 1st 2 data, both are setosa(numbering starts from 0)

#view 49th to 52nd dataset to see 2 classes(setosa and vesicolor)
subset = df.iloc[48:52]
print(subset)

# statistics computed for each column
summary_stats = df.describe()
print(summary_stats)

# data visualization, box plot for each feature
df.plot(kind='box', figsize=(10, 6))
plt.title('Box Plot of Iris Dataset')
plt.text(1.5, 8, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')

plt.savefig('Box Plot of Iris Dataset', bbox_inches='tight')
plt.show()

import seaborn as sns
# Set the size of the figure
plt.figure(figsize=(12, 7))
df['target'] = df['target'].replace({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
# Create a pair plot with hue based on the 'target' column in the dataframe
g=sns.pairplot(df, hue='target',plot_kws={'legend': False})
g._legend.set_title('')
# # Set the title of the plot
plt.suptitle('Attribute Correlation Pair Plot', fontsize=16, y=0.98)

# Set the style and color of the text box
textbox_props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)

# Add a text box with the highlighted feature names
plt.text(0.1, 8, "THA076BCT026\nTHA076BCT027\nTHA076BCT041", fontsize=10, color='red', ha='center', bbox=textbox_props)
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Setosa', markerfacecolor='royalblue', markersize=8),
                   plt.Line2D([0], [0], marker='o', color='w', label='Versicolor', markerfacecolor='g', markersize=8),
                   plt.Line2D([0], [0], marker='o', color='w', label='Virginica', markerfacecolor='orange', markersize=8)]

# Add the legend to the plot
plt.legend(handles=legend_elements, title='target')

# Adjust the spacing between subplots
plt.subplots_adjust(top=0.9)
# Set custom tick labels for the legend
# Save the plot to a file
plt.savefig('./plots/pair_plot.png', bbox_inches='tight')

# Display the plot
plt.show()

# Visualization of data points based on the target class (purple=setosa,green=Versicolour,yellow=Virginica )
df['target']=iris.target
plt.figure(figsize=(10, 6))
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=df['target'], label='Sepal')
plt.scatter(df['petal length (cm)'], df['petal width (cm)'], c=df['target'], marker='x', label='Petal')
plt.xlabel('Length (cm)')
plt.ylabel('Width (cm)')
plt.title('Scatter Plot of Sepal and Petal Features')
plt.colorbar(label='Species')
plt.text(1, 4.3, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.legend()
plt.savefig('./plots/Sepal and Petal Features', bbox_inches='tight')
plt.show()

# **STEP 3: Data preprocessing**

# Handling missing values:
# df.isna() returns a boolean, where True means missing values(NaN) and .sum() sums the values along each column
missing_values = df.isna().sum()
print("Columns with missing values:")
print(missing_values[missing_values > 0])

# Standardizing to eliminate the chance of PCA being influenced by magnitude of some features' values.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = df.iloc[:,1:].values
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# IMPORTANT STEP!
# Create new input and output after preprocessing X and y.
X_train = df_scaled.drop('target', axis=1).values
y_train = df['target']
print("shape of new input X",X_train.shape)
print("shape of new output Y",y_train.shape)
print(X_train)
print(y_train)

df_plot=df_scaled.copy()
df_plot['target'] = df['target'].replace({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
plt.figure(figsize=(12, 8))
# Create a pair plot with hue based on the 'target' column in the dataframe
g=sns.pairplot(df_plot, hue='target',plot_kws={'legend': False})
g._legend.set_title('')

plt.suptitle('Attribute Correlation Pair Plot of Transformed Data', fontsize=16, y=0.98)

# Set the style and color of the text box
textbox_props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8)

# Add a text box with the highlighted feature names
plt.text(1.5, 6, "THA076BCT026\nTHA076BCT027\nTHA076BCT041", fontsize=10, color='red', ha='center', bbox=textbox_props)

# Adjust the spacing between subplots
plt.subplots_adjust(top=0.9)

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Setosa', markerfacecolor='royalblue', markersize=8),
                   plt.Line2D([0], [0], marker='o', color='w', label='Versicolor', markerfacecolor='g', markersize=8),
                   plt.Line2D([0], [0], marker='o', color='w', label='Virginica', markerfacecolor='orange', markersize=8)]

# Add the legend to the plot
plt.legend(handles=legend_elements, title='target')
# Save the plot to a file
plt.savefig('./plots/pair_plot_transformed.png', bbox_inches='tight')

# Display the plot
plt.show()

# lets see the new plot after standardization
plt.figure(figsize=(10, 6))
plt.scatter(df_scaled['sepal length (cm)'], df_scaled['sepal width (cm)'], c=df_scaled['target'], label='Sepal')
plt.scatter(df_scaled['petal length (cm)'], df_scaled['petal width (cm)'], c=df_scaled['target'], marker='x', label='Petal')
plt.xlabel('Length (cm)')
plt.ylabel('Width (cm)')
plt.title('Scatter Plot of Sepal and Petal Features')
plt.colorbar(label='Species')
plt.ylim(-3,4)
plt.xlim(-3,3)
plt.text(-1, 3.5, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.legend()
plt.savefig('./plots/processed Sepal and Petal Feature', bbox_inches='tight')
plt.show()

# **STEP 4: Covariance matrix calculation.**

# X_train is already an array so no need to do data_matrix = np.array(X)
data_matrix = X_train
data_matrix.shape

# covariance matrix will be 4*4 since x=(150,4)
covariance_matrix = np.cov(X_train, rowvar=False)             # row of X_train != variable(so column of X_train means variable, and rows means observation)
print("Covariance matrix shape:", covariance_matrix.shape)
print(covariance_matrix)

# visualize covariance matrix
plt.imshow(covariance_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title('Covariance Matrix')
plt.text(0, -0.8, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.xticks(np.arange(4), np.arange(4))   #tick labels on axes changed from 0 to 4
plt.yticks(np.arange(4), np.arange(4))
plt.savefig('Iris Covariance Matrix', bbox_inches='tight')
plt.show()

# **STEP 5: Eigen values and eigen vector calculation**

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
#descending sort
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
print("eigen values are",eigenvalues)
print("eigen vectors are",eigenvectors)
print(eigenvalues.shape)
print(eigenvectors.shape)

# **STEP 6: See the variance explained by each eigen values.**

# Proportion of variance for each eigen values
pov_list=[]
for i in range (0,4):
  pov=eigenvalues[i]/sum(eigenvalues)
  pov_list.append(pov)
  print("Proportion of variance for", eigenvalues[i],"eigen value is", pov)

# Plot the scree plot
plt.plot(range(1, len(pov_list) + 1), pov_list, marker='o',color='blue')
varlegend= "'THA076BCT026','THA076BCT027','THA076BCT041'"
plt.text(2.5,0.7,varlegend,color='red')   #plt.text(x_position, y_position, varlegend)
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance')
plt.title('Scree Plot')
plt.show()


# **STEP 7: Select the eigen vectors as needed to compute the final output**

print("Shape of x train is",X_train.shape)
transposed_X_train = np.transpose(X_train)
transposed_X_train.shape
eigenvectors_transposed=np.transpose(eigenvectors)

final_data = {}       # Dictionary to store the final projected data

# Define the combinations of selected components
selected_components = [[0], [0, 1], [0,2], [0, 3], [1], [1,2], [1,3], [0, 1, 2, 3]]
count=len(selected_components)

# Iterate over the selected components
for i in range (0,count):
  selected_eigenvectors = eigenvectors_transposed[selected_components[i]]              #select the combination of components from above list
  final_data[i] = np.transpose(selected_eigenvectors @ transposed_X_train)  #projection
  print(final_data[i].shape)

eigenvectors_transposed[selected_components[2]]

# **STEP 8: Computing and visualizing covariance for the new data.**

new_covariance_matrices_dictionary = {}
# Compute covariance matrix for each projected data
for i in range(0, count):
    print(" * Covariance matrix for combination no", i)
    new_covariance_matrix = np.cov(final_data[i], rowvar=False)
    new_covariance_matrices_dictionary[i] = new_covariance_matrix
    # Print the shape of the covariance matrix
    print("  --->Covariance matrix shape:", new_covariance_matrix.shape)

#Lets access the first combination(combination 0), which is (eigen[0])                    (best component)
new_covariance_matrices_dictionary[0]

#Lets access combination 1, which is eigen[0] and eigen[1]              (2 best components)
new_covariance_matrices_dictionary[1]

#Lets access combination 3, which is eigen[0] and eigen[3]           (best and worst component)
new_covariance_matrices_dictionary[3]

#Lets access combination 5, which is eigen[1] and eigen[2]        (2 medium medium best components)
new_covariance_matrices_dictionary[5]

#Lets access combination 7, which is eigen[0],eigen[1],eigen[2] and eigen[3]      (all components combined)
new_covariance_matrices_dictionary[7]

# Heatmap plot, darker shade indicate higher values. (combination starts from 0 to 7)
plt.imshow(new_covariance_matrices_dictionary[1], cmap='Reds', interpolation='nearest')  #for smoother interpolation of color on pixel values, instead of 'nearest', we can use 'bilinear','bicubic','spline16', 'spline36', 'hanning', 'hamming', 'hermite',etc..
plt.colorbar()
plt.title("New Covariance Matrix Combination 1 (2 best PC)")
plt.text(-0.3, -0.66, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.savefig('Covariance Matrix for 2 best PC', bbox_inches='tight')
plt.show()

plt.imshow(new_covariance_matrices_dictionary[3], cmap='Reds', interpolation='nearest')
plt.colorbar()
plt.title("New Covariance Matrix Combination 3 (best and worst PC)")
plt.text(-0.3, -0.66, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.savefig('Covariance Matrix for best and worst PC', bbox_inches='tight')
plt.show()

plt.imshow(new_covariance_matrices_dictionary[5], cmap='Reds', interpolation='nearest')
plt.colorbar()
plt.title("Covariance Matrix for two medium PCs")
plt.text(-0.3, -0.66, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.savefig('Covariance Matrix for two medium PCs', bbox_inches='tight')
plt.show()

plt.imshow(new_covariance_matrices_dictionary[7], cmap='Reds', interpolation='nearest')
plt.colorbar()
plt.title("New Covariance Matrix Combination 7 (All Components)")
plt.show()

# **STEP 9: Obtain final data and visualize**

# Two best PCs
final_data[1].shape
#final_data[1]

plt.ylim(-3,3)
plt.xlim(-3,3)
# plt.title("2 best PCs")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title("Scatter Plot after PCA for two best PCs on IRIS Dataset")
plt.text(-1.8, 2.8, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.scatter(final_data[1][:,0],final_data[1][:,1],c=y)
plt.savefig('Scatter Plot after PCA for two best PCs', bbox_inches='tight')

#Best and worst PCs
final_data[3].shape
#final_data[3]

plt.ylim(-3,3)
plt.xlim(-3,3)
plt.xlabel('PC1')
plt.ylabel('PC4')
plt.title("Scatter Plot after PCA for Best and worst PCs on IRIS Dataset")
plt.text(-1.8, 2.8, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.scatter(final_data[3][:,0],final_data[3][:,1],c=y)
plt.savefig('Scatter Plot after PCA for Best and worst PCs', bbox_inches='tight')

#Best and mediumly worse PCs
final_data[5].shape
#final_data[5]

plt.ylim(-3,3)
plt.xlim(-3,3)
plt.title("Scatter Plot after PCA for two Medium PCs on IRIS Dataset")
plt.text(-1.8, 2.8, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.xlabel('PC2')
plt.ylabel('PC3')
plt.scatter(final_data[5][:,0],final_data[5][:,1],c=y)
plt.savefig('Scatter Plot after PCA for two Medium PCs', bbox_inches='tight')

#All components
final_data[7].shape
#final_data[7]

plt.ylim(-3,3)
plt.xlim(-3,3)
plt.title("All PCs")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.scatter(final_data[7][:,0],final_data[7][:,1],c=y)

# Create a 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(final_data[7][:,0],final_data[7][:,1],final_data[7][:,2],c=y, marker='o')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D Scatter Plot of 3 PCs')
ax.text(-6, 2.8,1, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
plt.tight_layout()
plt.savefig('3D Scatter Plot of 3 PCs', bbox_inches='tight',dpi=300)
plt.show()


# **STEP 10: PCA with library.**

from sklearn.decomposition import PCA
# Create instances of PCA with the desired number of components. (pca1 = 2components. pca2=so as to retain 95% info)
pca1 = PCA(n_components=2)
pca2 = PCA(0.95)

# Fit the PCA model to the data and transform the data
reduced_data1 = pca1.fit_transform(X)
reduced_data2 = pca2.fit_transform(X)

# Print the shape of the reduced data
print("Shape of reduced data:", reduced_data1.shape)
print("reduced through first",reduced_data1)

print("Shape of reduced data:", reduced_data2.shape)
print("reduced through second",reduced_data2)


# **STEP 11: Lets train 4 models and visualize the results.** 2 models are implemented after applying PCA using library. The third model is implemented after applying PCA using Math(using all the earlier above steps). The fourth model is trained without PCA.

# Create 4 sets of test-train for 4 models.
from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1 = train_test_split(reduced_data1,y,test_size=0.2)   # using library, 2 components
x_train2,x_test2,y_train2,y_test2 = train_test_split(reduced_data2,y,test_size=0.2)   # using library, 0.95 info
x_train3,x_test3,y_train3,y_test3 = train_test_split(final_data[1],y,test_size=0.2)   # from scratch, 2 components
x_train4,x_test4,y_train4,y_test4 = train_test_split(X_train,y,test_size=0.2)         # no PCA

#training of the model
# model 1: using library, 2 components
from sklearn.svm import SVC
svm1 = SVC()
svm1.fit(x_train1, y_train1)

#library, 0.95 info
svm2=SVC()
svm2.fit(x_train2, y_train2)

#from scratch, 2 components
svm3=SVC()
svm3.fit(x_train3, y_train3)

#No PCA
svm4=SVC()
svm4.fit(x_train4, y_train4)

# Accuracy check for 4 models
from sklearn.metrics import accuracy_score

y_pred1 = svm1.predict(x_test1)
accuracy1 = accuracy_score(y_test1, y_pred1)
print("Accuracy1:", accuracy1)

y_pred2 = svm2.predict(x_test2)
accuracy2 = accuracy_score(y_test2, y_pred2)
print("Accuracy2:", accuracy2)

y_pred3 = svm3.predict(x_test3)
accuracy3 = accuracy_score(y_test3, y_pred3)
print("Accuracy3:", accuracy3)

y_pred4 = svm4.predict(x_test4)
accuracy4 = accuracy_score(y_test4, y_pred4)
print("Accuracy4:", accuracy4)

#prediciton from 4 models
predictions1 = svm1.predict(x_test1)
predictions2 = svm2.predict(x_test2)
predictions3 = svm3.predict(x_test3)
predictions4 = svm4.predict(x_test4)
predictions1.shape

# Lets define a function to plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
def plot_confusion_matrix(y_true, y_pred, model_name,text):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.text(0.2, -0.5, "'THA076BCT026','THA076BCT027','THA076BCT041'", fontsize=8,color='red')
    plt.savefig(f'./plots/{model_name}.png', bbox_inches='tight',dpi=300)
    plt.show()

# Plot confusion matrix for model 1
plot_confusion_matrix(y_test1, predictions1, "UsingLibrary, 2 Components")

# for model 2
plot_confusion_matrix(y_test2, predictions2, "Using Library, 0.95 Info retention")

#for model 3
plot_confusion_matrix(y_test3, predictions3, "From Scratch, 2 Components")

#for model4
plot_confusion_matrix(y_test4, predictions4, "No PCA")

