import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa

np.random.seed(521)

# Part 1
# Mean parameters
class_means = np.array([[2.5, 2.5],
                        [-2.5, 2.5],
                        [-2.5, -2.5],
                        [2.5, -2.5],
                        [0.0, 0.0]])
# Standard deviation parameters
class_covariance = np.array([ [ [0.8, -0.6],
                                [-0.6, 0.8] ],
                              [ [0.8, 0.6],
                                [0.6, 0.8] ],
                              [ [0.8, -0.6],
                                [-0.6, 0.8] ],
                              [ [0.8, 0.6],
                                [0.6, 0.8] ],
                              [ [1.6, 0.0],
                                [0.0, 1.6]]])
# sample sizes
class_sizes = np.array([50, 50, 50, 50, 100])

# Sample generation
points1= np.random.multivariate_normal(class_means[0,:],class_covariance[0,:,:],class_sizes[0])
points2= np.random.multivariate_normal(class_means[1,:],class_covariance[1,:,:],class_sizes[1])
points3= np.random.multivariate_normal(class_means[2,:],class_covariance[2,:,:],class_sizes[2])
points4= np.random.multivariate_normal(class_means[3,:],class_covariance[3,:,:],class_sizes[3])
points5= np.random.multivariate_normal(class_means[4,:],class_covariance[4,:,:],class_sizes[4])
X = np.vstack((points1, points2, points3, points4, points5))

# Plot data points generated
# plt.figure(figsize = (5,5))
# plt.plot(points1[:,0], points1[:,1], "k.", markersize = 10)
# plt.plot(points2[:,0], points2[:,1], "k.", markersize = 10)
# plt.plot(points3[:,0], points3[:,1], "k." , markersize = 10)
# plt.plot(points4[:,0], points4[:,1], "k." , markersize = 10)
# plt.plot(points5[:,0], points5[:,1], "k." , markersize = 10)
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()

# Part 2
K = 5
N = np.sum(class_sizes)

def update_centroids(memberships, X):
    if memberships is None:
        # initialize centroids
        centroids = X[np.random.choice(range(N), K),:]
    else:
        # update centroids
        centroids = np.vstack([np.mean(X[memberships == k,], axis = 0) for k in range(K)])
    return (centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return (memberships)

centroids = None
memberships = None
centroids = update_centroids(memberships, X)
memberships = update_memberships(centroids, X)
centroids = update_centroids(memberships, X)
memberships = update_memberships(centroids, X)

# Part 3
centroids = update_centroids(memberships, X)  # To update centroids with latest memberships

priors = [np.sum(memberships == c) for c in range(K)]/N
# print("Initial Priors =\n",priors)

means = centroids
# print("Initial Means =\n",means)

covariances = [np.matmul(np.transpose(X[memberships == c] - means[c]),
                                         X[memberships == c] - means[c])
                                        /np.sum([memberships == c]) for c in range(K)]
covariances = np.array(covariances)
# print("Initial Covariances =\n",covariances)

def getCov(index):
    return covariances[index,:,:]

# Part 4
for i in range(100):
    def h_nom(i, c):
        return (priors[c] * (np.linalg.det(getCov(c)) ** (-0.5)) *
                np.exp(-0.5 * np.dot(np.transpose(X[i, :] - means[c, :]),
                                        np.dot(np.linalg.inv(getCov(c)), (X[i, :] - means[c, :])))))
        # return (priors[c] * (np.linalg.det(getCov(c)) ** (-0.5)) *
        #         np.exp( -0.5 * np.matmul( (X[i,:] - means[c,:]) , np.matmul(np.transpose(X[i,:] - means[c,:]) ,np.linalg.inv(getCov(c))) ) ))


    hnom = np.zeros((N,K))
    for i in range(N):
        for c in range(K):
            hnom[i,c] = h_nom(i, c)

    hdenom = np.zeros((N,1))
    for i in range(N):
        sum = 0
        for c in range(K):
            sum += hnom[i,c]
        hdenom[i] = sum


    def h_func(i,c):
        return hnom[i,c]/hdenom[i]


    # Update Means
    for c in range(K):
        nom = 0
        denom = 0
        for i in range(N):
            h = h_func(i,c)
            nom += h * X[i]
            denom += h
        means[c] = nom/denom

    # Update Priors
    for c in range(K):
        nom = 0
        for i in range(N):
            h = h_func(i, c)
            nom += h
        priors[c] = nom / N

    # Update Covariances
    for c in range(K):
        nom = np.zeros((2,2))
        denom = 0.0
        for i in range(N):
            h = h_func(i, c)
            a = X[i, :] - means[c, :]
            a = a.reshape((2, 1))
            b = a.reshape((1, 2))
            nom += np.dot( a, b) * h
            denom += h
        covariances[c,:] = nom/denom


memberships = update_memberships(centroids, X)
print("Means after 100 iterations:\n",means)
# print("Covs after 100 iterations:\n",covariances)
# print("Priors after 100 iterations:\n",priors)

# Part 5
# Plot data points generated
interval = 200
x1_interval = np.linspace(-6, +6, interval )
x2_interval = np.linspace(-6, +6, interval)
estimated = np.zeros((len(x1_interval), len(x2_interval), K))
real = np.zeros((len(x1_interval), len(x2_interval), K))

def estimate(X,cov,mean,prior):
    sq = 1/(2*np.linalg.det(cov))**0.5
    ex = np.exp(-0.5 * np.dot( np.dot( np.transpose(X-mean), np.linalg.inv(cov)) , (X-mean) ))
    return ex*sq

for c in range(K):
    for i in range(interval):
        for j in range(interval):
            estimated[i, j, c] = estimate(np.array([x1_interval[i], x2_interval[j]]), covariances[c], means[c], priors[c])
            real[i, j, c] = estimate(np.array([x1_interval[i], x2_interval[j]]), class_covariance[c], class_means[c],
                                          class_sizes[c]/N)

plt.figure(figsize = (6,6))
colors = ["b.","g.","y.","r.","m."]
for c in range(K):
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], colors[c], markersize=10)
    plt.contour(x1_interval, x2_interval, estimated[:, :, c], levels= [0.05], colors="k")
    plt.contour(x1_interval, x2_interval, real[:, :, c], levels=[0.05], colors="k", linestyles = "dashed")

plt.show()


