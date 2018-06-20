from NaiveBayesRunner import NaiveBayesRunner
import numpy as np

class TestExecutor:

    def __init__(self,data_x_path):
        self.data_x_path=data_x_path


    def execute(self,):
        data_x=np.loadtxt(self.data_x_path)
        predicted_data_y=NaiveBayesRunner(data_x).predict()
        np.savetxt('test_data_predicted.txt',predicted_data_y)