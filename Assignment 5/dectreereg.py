import numpy as np
import matplotlib.pyplot as plt

# Given .csv file was named as "hw04", that is why I open such a file below
data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",", skip_header = 1)
# Splitting the data into training and test set
x = data_set[:,0]
y = data_set[:,1]
x_train = x[:100]
x_test = x[100:]
y_train = y[:100]
y_test = y[100:]

N_train = len(y_train)
N_test = len(y_test)
# Setting the parameter P
P = 15


# Decision Tree Regression function, most of it directly copied from lab
# Defined as a method, since we will use it more than once
def decision_tree_reg(P, N_train, x_train, y_train):
  node_indices = {}
  is_terminal = {}
  need_split = {}
  node_splits = {}
  # Root initialization
  node_indices[1] = np.array(range(N_train))
  is_terminal[1] = False
  need_split[1] = True
  while True:
    split_nodes = [key for key, value in need_split.items() if value == True]
    if len(split_nodes) == 0:
      break
    for split_node in split_nodes:
      data_indices = node_indices[split_node]
      need_split[split_node] = False
      if len(data_indices) <= P:
        node_splits[split_node] = np.mean(y_train[data_indices])
        is_terminal[split_node] = True
      else:
        is_terminal[split_node] = False
        unique_values = np.sort(np.unique(x_train[data_indices]))
        split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
        split_scores = np.repeat(0.0, len(split_positions))
        for s in range(len(split_positions)):
          left_indices = data_indices[x_train[data_indices] <= split_positions[s]]
          right_indices = data_indices[x_train[data_indices] > split_positions[s]]
          error=0
          mean_left = np.mean(y_train[left_indices])
          error = error + np.sum((y_train[left_indices]-mean_left)**2)
          mean_right = np.mean(y_train[right_indices])
          error = error + np.sum((y_train[right_indices] - mean_right) ** 2)
          split_scores[s] = error / (len(left_indices) + len(right_indices))
        best_splits = split_positions[np.argmin(split_scores)]
        node_splits[split_node] = best_splits
        left_indices = data_indices[x_train[data_indices] <= best_splits]
        node_indices[2 * split_node] = left_indices
        is_terminal[2 * split_node] = False
        need_split[2 * split_node] = True
        right_indices = data_indices[x_train[data_indices] > best_splits]
        node_indices[2 * split_node + 1] = right_indices
        is_terminal[2 * split_node + 1] = False
        need_split[2 * split_node + 1] = True
  return is_terminal, node_splits


# Method for getting predicted values from data
# Defined as a method, since we will use it more than once
def get_ypred (data,node_splits,is_terminal):
    N = data.shape[0]
    y_predicted = np.repeat(0, N)
    for i in range(N):
      index = 1
      while True:
        if is_terminal[index] == True:
          y_predicted[i] = node_splits[index]
          break
        else:
          if data[i] <= node_splits[index]:
            index = index * 2
          else:
            index = index * 2 + 1
    return y_predicted


# Use above 2 methods to predict the values for training data with parameter P = 15
# And set is_terminal and node_splits for graphing
is_terminal = decision_tree_reg(P, N_train, x_train, y_train)[0]
node_splits = decision_tree_reg(P, N_train, x_train, y_train)[1]

# For each point in interval_length, defined below, find the predicted values
minimum_value = 0
maximum_value = 60
interval_length = (maximum_value-minimum_value) * 100 + 1
data_int = np.linspace(minimum_value,maximum_value,interval_length)
ypred = get_ypred(data_int, node_splits, is_terminal)

# Plotting first graph
plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, 'b.', label='training', markersize=10)
plt.plot(x_test, y_test, 'r.', label='test',markersize=10)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(data_int, ypred, "k-")
plt.show()


# RMSE for test set, same approach I followed in Assignment 4,
# Defined as a method, since we will use it more than once
def rmse(x_test,y_test):
  y_test_pred = get_ypred(x_test, node_splits, is_terminal)
  cumSum = 0
  for c in range(x_test.shape[0]):
    cumSum += (y_test_pred[c] - y_test[c]) ** 2
  rmse = np.sqrt(cumSum/x_test.shape[0])
  return rmse


print("RMSE is ", rmse(x_test,y_test), "when P is",P)

# Initialize an array to hold different RMSE values for different P values
minP=5;
maxP=55;
diff=5;
rmse_arr = [None] * (int)((maxP-minP)/diff) # To initialize an empty array, to be filled below
rmse_p_range=np.arange(minP,maxP,diff)
# Find each RMSE value
for c in range((int)((maxP-minP)/diff)):
    is_terminal = decision_tree_reg(5*(c+1), N_train, x_train, y_train)[0]
    node_splits  = decision_tree_reg(5*(c+1), N_train, x_train, y_train)[1]
    rmse_arr[c] = rmse(x_test,y_test)

# Plot RMSE values
plt.figure(figsize=(10, 6))
plt.plot(rmse_p_range, rmse_arr, "k-")
plt.plot(rmse_p_range, rmse_arr, "k.", markersize=15)
plt.xlabel("Pre-pruning size (P)")
plt.show()