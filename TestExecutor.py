from NaiveBayesRunner import NaiveBayesRunner
import numpy as np

data_x = np.loadtxt('test_data_x.txt')
predicted_data_y = NaiveBayesRunner().predict(data_x)
np.savetxt('test_data_predicted.txt', predicted_data_y)
