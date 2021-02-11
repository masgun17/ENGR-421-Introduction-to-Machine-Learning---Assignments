import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(421)
# bad 251 426 427 541 //visually
# nice 521
# average 428 1

# Mean parameters
class_means = np.array([[0.0, 2.5],
                        [-2.5,-2.0],
                        [2.5, -2.0]])
# Standard deviation parameters
class_covariance = np.array([ [ [3.2, 0.0],
                                [0.0, 1.2] ],
                              [ [1.2, -0.8],
                                [-0.8, 1.2] ],
                              [ [1.2, 0.8],
                                [0.8, 1.2] ] ])
# sample sizes
class_sizes = np.array([120, 90, 90])

# Sample generation
points1= np.random.multivariate_normal(class_means[0,:], class_covariance[0,:,:], class_sizes[0])
points2= np.random.multivariate_normal(class_means[1,:],class_covariance[1,:,:],class_sizes[1])
points3= np.random.multivariate_normal(class_means[2,:],class_covariance[2,:,:],class_sizes[2])

# Plot data points generated
plt.figure(figsize = (5, 5))
plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
plt.plot(points3[:,0], points3[:,1], "b." , markersize = 10)
plt.xlabel("x1")
plt.ylabel("x2")
# plt.show()

# Concatenate the data points and label them
X = np.vstack((points1, points2, points3))
points = np.concatenate( (points1,points2,points3))
label= np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))

# Exporting and importing - Not Necessary for us, so skipped
# np.savetxt("hw1_data_set.csv", np.hstack((X, label[:, None])), fmt = "%f,%f,%d")
# data_set = np.genfromtxt("hw1_data_set.csv", delimiter = ",")
# X = data_set[:, [0,1]]
# y = data_set[:,2].astype(int)
y = label.astype(int)
N = X.shape[0]
K= np.max(label)

sample_means = [np.mean(X[y == (c + 1)], axis=0) for c in range(K)]
# sample_means = np.array([[np.mean(X[:,0][y == (c + 1)]) for c in range(K)],
#                          [np.mean(X[:,1][y == (c + 1)]) for c in range(K)]])
print("Sample Means:")
print(sample_means)

sample_covariance= [np.matmul(np.transpose(X[y == (c + 1)] - sample_means[c]),
                                         X[y == (c + 1)] - sample_means[c])
                                        /np.sum([y == (c + 1)]) for c in range(K)]
print("\nSample Covariance:")
print(sample_covariance)

class_priors = [np.mean(y == (c + 1)) for c in range(K)]
print("\nClass Priors:")
print(class_priors)

# W,w,w0 parameters
W = [-0.5*(np.linalg.inv(sample_covariance[c])) for c in range(K)]
w= [np.matmul( np.linalg.inv(sample_covariance[c]), sample_means[c]) for c in range(K)]
w0= [-0.5*(np.matmul(np.matmul(np.transpose(sample_means[c]), np.linalg.inv(sample_covariance[c])), sample_means[c]))\
     -0.5*np.log(np.linalg.det(sample_covariance[c]))+ np.log(class_priors[c]) for c in range(K)]

# Score Function
def gfunc(x,W,w,w0,c):
    return np.matmul(x,np.matmul(W[c],np.transpose(x))) + np.matmul(np.transpose(w[c]),np.transpose(x)) + w0[c]

# Method for predicting labels
def prediction(x,W,w,w0):
    return np.argmax([gfunc(x,W,w,w0,c) for c in range(K)]) + 1

# Prediction
y_pred=np.array([])
for i in range(N):
    y_pred = np.append( y_pred,[prediction(X[i],W,w,w0)])

confusion_matrix = pd.crosstab(y_pred, y, rownames = ['y_pred'], colnames = ['y_truth'])
print("\nConfusion matrix:\n", confusion_matrix)

# evaluate discriminant function on a grid
x1_interval = np.linspace(-6, +6, 1201)
x2_interval = np.linspace(-6, +6, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))
for c in range(K):
    discriminant_values[:,:,c] = W[c][0][0]*(x1_grid**2) + W[c][0][1]*x1_grid*x2_grid \
                                 + W[c][1][0]*x2_grid*x1_grid + W[c][1][1]*(x2_grid**2) \
                                 + w[c][0]*x1_grid + w[c][1]*x2_grid + w0[c]

A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:,:,0] = A
discriminant_values[:,:,1] = B
discriminant_values[:,:,2] = C

plt.figure(figsize = (10, 10))
plt.plot(X[y == 1, 0], X[y == 1, 1], "r.", markersize = 10)
plt.plot(X[y == 2, 0], X[y == 2, 1], "g.", markersize = 10)
plt.plot(X[y == 3, 0], X[y == 3, 1], "b.", markersize = 10)
plt.plot(X[y_pred != y, 0], X[y_pred != y, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,2], levels = 0, colors = "k")

plt.contourf(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels = 0, colors = ["palegreen","lightcoral"])
plt.contourf(x1_grid, x2_grid, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels = 0, colors = ["cornflowerblue","lightcoral"])
plt.contourf(x1_grid, x2_grid, discriminant_values[:,:,1] - discriminant_values[:,:,2], levels = 0, colors = ["cornflowerblue","palegreen"])

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

