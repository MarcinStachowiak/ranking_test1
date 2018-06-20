from AbstractRunner import AbstractRunner
import numpy as np

class NaiveBayesRunner(AbstractRunner):

    def predict(self,data_x):
        return np.random.randint(1,4,data_x.shape[0])