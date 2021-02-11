import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(421)

# I wanted to read labels as "A,B,C,D,E" and I couldn't do it with numpy. So I used pandas and turned the dataframe
# into numpy array.

# data_set = np.genfromtxt("hw02_data_set_images.csv", delimiter = ",")
# label_set = np.genfromtxt("hw02_data_set_labels.csv", delimiter = ",",dtype=str)
data_set = pd.read_csv("hw02_data_set_images.csv", header=None).to_numpy()
label_set = pd.read_csv("hw02_data_set_labels.csv", header=None).to_numpy()

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

# I combine the training datas and test datas. First I used np.array, but np.vstack was more useful for me in the end.
# xtrain = np.array([xAtrain,xBtrain,xCtrain,xDtrain,xEtrain])
xtrain = np.vstack((xAtrain, xBtrain, xCtrain, xDtrain, xEtrain))
# ytrain = np.array([yAtrain,yBtrain,yCtrain,yDtrain,yEtrain])
ytrain = np.vstack((yAtrain, yBtrain, yCtrain, yDtrain, yEtrain))
# xtest = np.array([xAtest,xBtest,xCtest,xDtest,xEtest])
xtest = np.vstack((xAtest, xBtest, xCtest, xDtest, xEtest))
# ytest = np.array([yAtest,yBtest,yCtest,yDtest,yEtest])
ytest = np.vstack((yAtest, yBtest, yCtest, yDtest, yEtest))

# Printing some sets to check the data.
print("xtrain.shape, ytrain.shape, xtest.shape, ytest.shape:")
print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)
print("xtrain.shape[1]:", xtrain.shape[1])


# Since I read the labels as string values,to convert them into integer values, I followed the following method:
# For "A": 1 0 0 0 0, for "B": 0 1 0 0 0 ...
# The main reason for this part is to put the label data in the same format as y_prediction.
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
# print(newlbl.shape)
y_truth= newlbl[1:,:]   # Getting rid of the first row, which consist of 0 0 0 0 0, which was just initialized to
# print(y_truth.shape)  # perform np.vstack operations


# The commented part below, was the code I first submitted. However, later I realized that, If I were given
# any other label set, this code would not work, moreover, this part assumes that I know the contents of
# label files, which I did not want to use actually. So this part is replaced by the part above.
# zeros = np.zeros(25)
# ones = np.ones(25)
# y_train1 = np.vstack((ones, zeros, zeros, zeros, zeros)).T
# y_train2 = np.vstack((zeros, ones, zeros, zeros, zeros)).T
# y_train3 = np.vstack((zeros, zeros, ones, zeros, zeros)).T
# y_train4 = np.vstack((zeros, zeros, zeros, ones, zeros)).T
# y_train5 = np.vstack((zeros, zeros, zeros, zeros, ones)).T
# y_truth = np.vstack([y_train1, y_train2, y_train3, y_train4, y_train5])
# # print("y_truth.shape:", y_truth.shape)

# Sigmoid function from lab.
def sigmoid(X, W, w0):
    return 1 / (1 + np.exp(- (np.matmul(X, W) + w0)))


# Define the gradient functions.
def gradient_w(X, y_truth, y_predicted):
    return (np.asarray(
        [-np.sum(np.repeat(((y_truth[:,c] - y_predicted[:,c])*y_predicted[:,c]*(1-y_predicted[:,c]))[:, None], X.shape[1], axis=1) * X, axis=0) for c in
         range(5)]).transpose())

def gradient_w0(y_truth, y_predicted):
    return (-np.sum( (y_truth - y_predicted)*y_predicted*(1-y_predicted) , axis=0))

# Learning parameters.
eta = 0.01
epsilon = 1e-3

# Initialization of w and w0.
w = np.random.uniform(low=-0.01, high=0.01, size=(320, 5))
w0 = np.random.uniform(low=-0.01, high=0.01, size=(1, 5))

# Printed some values during coding to check w and w0.
# print("w.shape:", w.shape)
# print("w0.shape:", w0.shape)
#
print("Initial w:\n", w)
print("Initial w0:", w0)
#
# print(sigmoid(xtrain, w, w0).shape)  # Assuming, each row is one dataset and each column represents the corresponding
                                        # score. Max of them is the classification made by sigmoid.

# Defined safelog operation to prevent log(0).
def safelog(x):
    return np.log(x + 1e-100)

# Gradient descent. Updated the w and w0 at each turn.
iteration = 1
objective_values = []
while 1:
    y_predicted = sigmoid(xtrain, w, w0)
    objective_values = np.append(objective_values,
                                 0.5 * np.sum((y_truth-y_predicted)**2) )

    w_old = w
    w0_old = w0
    w = w - eta * gradient_w(xtrain, y_truth, y_predicted)
    w0 = w0 - eta * gradient_w0(y_truth, y_predicted)

    if np.sqrt(np.sum((w0 - w0_old)) ** 2 + np.sum((w - w_old) ** 2)) < epsilon:
        break
    iteration = iteration + 1

# Printed w and w0 to compare the results.
print("After loop:")
print("w:",w)
print("w0:",w0)

# plot objective function during iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

# print("y_predicted:\n",y_predicted.shape)
# print("y_truth:\n",y_truth.shape)

# Calculate confusion matrix for training data
Y_predicted = np.argmax(y_predicted, axis=1) + 1
confusion_matrix = pd.crosstab(Y_predicted, ytrain[:, 0], rownames=['y_predicted'], colnames=['y_train'])
print(confusion_matrix)

print("--------------------------------------")

# Apply the calculated w and w0 on test data and calculate the confusion matrix.
ytest_predicted = sigmoid(xtest,w,w0)
ytest_predicted = np.argmax(ytest_predicted, axis=1) + 1
confusion_matrix_test = pd.crosstab(ytest_predicted, ytest[:,0], rownames=['y_predicted'], colnames=['y_test'])
print(confusion_matrix_test)

print("Iteration:", iteration)
