import matplotlib.pyplot as plt
import numpy as np
import math

data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",", skip_header = 1)
# Splitting the data into training and test set
x = data_set[:,0]
y = data_set[:,1]
xtrain = x[:100]
xtest = x[100:]
ytrain = y[:100]
ytest = y[100:]

N = xtrain.shape[0]
Ntest = xtest.shape[0]
# Assigning minimum and maximum values for x's
minimum_value = 0
maximum_value = 60


# Part 1: Regressogram
bin_width = 3
left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)
p_hat1_sum = np.asarray([np.sum( ((left_borders[b] < xtrain) & (xtrain <= right_borders[b])) * ytrain ) for b in range(len(left_borders))])
p_hat1_denom = np.asarray([np.sum(  ((left_borders[b] < xtrain) & (xtrain <= right_borders[b])) ) for b in range(len(left_borders))])
p_hat1 = p_hat1_sum / (p_hat1_denom+1e-10)  # To prevent dividing with 0

# Plotting
plt.figure(figsize = (10, 6))
plt.plot(xtrain, ytrain, "b.", markersize = 10)
plt.plot(xtest, ytest, "r.", markersize = 10)
plt.xlabel("x")
plt.ylabel("y")
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat1[b], p_hat1[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat1[b], p_hat1[b + 1]], "k-")
plt.show()

# Root mean squared error (RMSE) of regressogram for test set
cumsum = 0
for i in range(Ntest):
    index = int(xtest[i]/bin_width)
    base = ytest[i] - p_hat1[index]
    cumsum += base ** 2
regRMSE = np.sqrt(cumsum/Ntest)
print("Regressogram => RMSE is",regRMSE,"when h is",bin_width )


# Part 2: Running Mean Smoother
interval_length = (maximum_value-minimum_value)*100+1
data_interval = np.linspace(minimum_value, maximum_value, interval_length)
bin_width = 3
p_hat2_sum = np.asarray([np.sum( (((x - 0.5*bin_width) < xtrain) & (xtrain <= (x + 0.5*bin_width))) * ytrain ) for x in data_interval])
p_hat2_denom = np.asarray([np.sum( (((x - 0.5 * bin_width) < xtrain) & (xtrain <= (x + 0.5 * bin_width)))) for x in data_interval])
p_hat2 = p_hat2_sum / (p_hat2_denom+1e-10)  # To prevent dividing with 0

# Plotting
plt.figure(figsize = (10, 6))
plt.plot(xtrain, ytrain, "b.", markersize = 10)
plt.plot(xtest, ytest, "r.", markersize = 10)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(data_interval, p_hat2, "k-")
plt.show()

# RMSE of running mean smoother for test set
cumsum = 0
for i in range(Ntest):
    index = int(xtest[i] * interval_length/(maximum_value-minimum_value))
    base = ytest[i] - p_hat2[index]
    cumsum += base ** 2
meanRMSE = np.sqrt(cumsum/Ntest)
print("Running Mean Smoother => RMSE is",meanRMSE,"when h is",bin_width )


# Part 3: Kernel Smoother
interval_length = (maximum_value-minimum_value)*100+1
bin_width = 1
data_interval = np.linspace(minimum_value, maximum_value, interval_length)
p_hat3_sum = np.asarray([np.sum((1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - xtrain)**2 / bin_width**2) * ytrain )) for x in data_interval])
p_hat3_denom = np.asarray([(np.sum(1 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - xtrain)**2 / bin_width**2))) for x in data_interval])
p_hat3 = p_hat3_sum / (p_hat3_denom+1e-10)  # To prevent dividing with 0

# Plotting
plt.figure(figsize = (10, 6))
plt.plot(xtrain, ytrain, "b.", markersize = 10)
plt.plot(xtest, ytest, "r.", markersize = 10)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(data_interval, p_hat3, "k-")
plt.show()

# RMSE of kernel smoother for test set
cumsum = 0
for i in range(Ntest):
    index = int(xtest[i] * interval_length/(maximum_value-minimum_value))
    base = ytest[i] - p_hat3[index]
    cumsum += base ** 2
kerRMSE = np.sqrt(cumsum/Ntest)
print("Kernel Smoother => RMSE is",kerRMSE,"when h is",bin_width )

