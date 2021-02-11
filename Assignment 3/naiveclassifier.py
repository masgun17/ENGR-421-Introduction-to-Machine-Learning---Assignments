import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(421)

data_set = pd.read_csv("hw02_data_set_images.csv", header=None).to_numpy()
label_set = pd.read_csv("hw03_data_set_labels.csv", header=None).to_numpy()

# I divided the data_set and label_set for each character.
xA, yA = data_set[0:39, :], label_set[0:39]
xB, yB = data_set[39:78, :], label_set[39:78]
xC, yC = data_set[78:117, :], label_set[78:117]
xD, yD = data_set[117:156, :], label_set[117:156]
xE, yE = data_set[156:195, :], label_set[156:195]

# For each character, I separated them into training set and test set.
# I used sklearn library for this. I suppose this would not be a problem, since no one told us not to do.
xAtrain, xAtest, yAtrain, yAtest = train_test_split(xA, yA, test_size=14 / 39, shuffle=False)
xBtrain, xBtest, yBtrain, yBtest = train_test_split(xB, yB, test_size=14 / 39, shuffle=False)
xCtrain, xCtest, yCtrain, yCtest = train_test_split(xC, yC, test_size=14 / 39, shuffle=False)
xDtrain, xDtest, yDtrain, yDtest = train_test_split(xD, yD, test_size=14 / 39, shuffle=False)
xEtrain, xEtest, yEtrain, yEtest = train_test_split(xE, yE, test_size=14 / 39, shuffle=False)

xtrain = np.vstack((xAtrain, xBtrain, xCtrain, xDtrain, xEtrain))
ytrain = np.vstack((yAtrain, yBtrain, yCtrain, yDtrain, yEtrain))
xtest = np.vstack((xAtest, xBtest, xCtest, xDtest, xEtest))
ytest = np.vstack((yAtest, yBtest, yCtest, yDtest, yEtest))

# Printing some sets to check the data.
# print("xtrain.shape, ytrain.shape, xtest.shape, ytest.shape:")
# print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
# print("xtrain.shape[1]:", xtrain.shape[1])


# Since I read the labels as string values,to convert them into integer values, I followed the following method:
# For "A": 1 0 0 0 0, for "B": 0 1 0 0 0 ...
newlbl= np.array([0,0,0,0,0])
for label in ytrain:
    if label == 'A':
        newlbl= np.vstack((newlbl,np.array([1, 0, 0, 0, 0])))
    if label == 'B':
        newlbl= np.vstack((newlbl,np.array([0, 1, 0, 0, 0])))
    if label == 'C':
        newlbl= np.vstack((newlbl,np.array([0, 0, 1, 0, 0])))
    if label == 'D':
        newlbl= np.vstack((newlbl,np.array([0, 0, 0, 1, 0])))
    if label == 'E':
        newlbl= np.vstack((newlbl,np.array([0, 0, 0, 0, 1])))

y_truth= newlbl[1:,:]   # Getting rid of the first row

# Number of classes
K = 5

# For each class, creating corresponding pcd, and stack them vertically in a matrix.
pcd = np.zeros((1,320))
for c in range(K):
    pcd = np.vstack((pcd, np.mean(xtrain[(y_truth == 1)[:,c]], axis=0)))

pcd = pcd[1:,].T

# For aesthetic purposes: print all pcd's below, in one line.
np.set_printoptions(linewidth=np.inf)

# print(pcd)
print("pcd[:,0]:")
print(pcd[:,0])
print("pcd[:,1]:")
print(pcd[:,1])
print("pcd[:,2]:")
print(pcd[:,2])
print("pcd[:,3]:")
print(pcd[:,3])
print("pcd[:,4]:")
print(pcd[:,4])

# Class Priors
class_priors = [np.mean((y_truth == 1)[:,c]) for c in range(K)]
print("Class priors:")
print(class_priors)

# Reshaping for plotting the images
imarr0= np.reshape(pcd[:,0], (16,20))
imarr1= np.reshape(pcd[:,1], (16,20))
imarr2= np.reshape(pcd[:,2], (16,20))
imarr3= np.reshape(pcd[:,3], (16,20))
imarr4= np.reshape(pcd[:,4], (16,20))

f, port = plt.subplots(1,5)

# Plotting the images
port[0].imshow(imarr0.T, cmap = "binary")
port[1].imshow(imarr1.T, cmap = "binary")
port[2].imshow(imarr2.T, cmap = "binary")
port[3].imshow(imarr3.T, cmap = "binary")
port[4].imshow(imarr4.T, cmap = "binary")

plt.show()

# Defining a safe log function to avoid -inf
def safelog(x):
    return np.log(x + 1e-100)

# Discriminant Function
# Since I will get the argmax later, I did not include the log(P(Ci)) part, which is same for each class.
def disc_func(x):
    return np.matmul(x, safelog(pcd)) + np.matmul((1-x), safelog(1-pcd))

# Applying discriminant function on training set
ytrain_pred = np.argmax(disc_func(xtrain), axis= 1 ) + 1
confusion_matrix = pd.crosstab(ytrain_pred, ytrain[:, 0], rownames=['y_predicted'], colnames=['y_train'])
print(confusion_matrix)

print("--------------------------------------")

# Applying discriminant function on test set
ytest_predicted = disc_func(xtest)
ytest_predicted = np.argmax(ytest_predicted, axis=1) + 1
confusion_matrix_test = pd.crosstab(ytest_predicted, ytest[:,0], rownames=['y_predicted'], colnames=['y_test'])
print(confusion_matrix_test)



