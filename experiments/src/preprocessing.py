import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

def create_sequences(data: np.ndarray, time_steps: int, sliding_window: int):
    sequences, targets = [], []
    for i in range(len(data) - time_steps - sliding_window + 1):
        sequences.append(data[i:i + time_steps, :])
        targets.append(data[i + time_steps + sliding_window-1, -1]) 
    return np.array(sequences), np.array(targets)


def load_data(file_ids, time_steps: int=100, sliding_window: int=1, folder_path: str="../data/MESWE-38-processed/"):
    """
    Load CSV files into separate DataFrames for training, validation and test.
    """

    numpy_data_x = []
    numpy_data_y = []

    for id in file_ids:
        file_path = f"{folder_path}meswe_event_{id}.csv"
        df = pd.read_csv(file_path)
        cols = df.columns
        
        if "Timestamp" in cols:
            df = df.drop(["Timestamp"], axis=1) 

        # move dst to last col
        df['Dst'] = df.pop('Dst')
        
        df = df.dropna()
        
        #-----savgol smoothing of Q-data ------------------------------------------------
        # - savitzky golay filter -> https://eigenvector.com/wp-content/uploads/2020/01/SavitzkyGolay.pdf
        # - using savgol filter on Q-data can be left out, but we figured out that it gives small performance boost, so we kept it here
        columns_to_smooth = ['Q1', 'Q2', 'Q3', 'Q4']
        smoothed_values = savgol_filter(df[columns_to_smooth], window_length=51, polyorder=1, axis=0)

        smoothed_data_savgol = df.copy()
        smoothed_data_savgol[columns_to_smooth] = smoothed_values
        
        np_data = smoothed_data_savgol.to_numpy()
        #-----------------------------------------------------
        #np_data = df.to_numpy()
        
        data_X, data_y = create_sequences(np_data, time_steps, sliding_window)
        numpy_data_x.append(data_X)
        numpy_data_y.append(data_y)
    
    
    data_X_concated = np.concatenate(numpy_data_x, axis=0)
    data_y_concated = np.concatenate(numpy_data_y, axis=0)

    return data_X_concated, data_y_concated


def get_k_fold(k_fold: int):
    """function returns event number for train, validation and test set. 
    This numbers are then processed and DataFrame is created out of them.
    """

    if k_fold == 1:
        file_numbers_train = [6, 7, 9, 12, 13, 19, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37, 1, 22, 5, 13, 10, 17, 38, 15, 23]
        file_numbers_val = [4, 21, 11]
        file_numbers_test = [8, 14, 28]


    elif k_fold == 2:
        file_numbers_train = [4, 6, 7, 8, 9, 11, 12, 13, 14, 29, 30, 31, 32, 33, 34, 35, 36, 37, 5, 13, 10, 17, 38, 15, 23]
        file_numbers_val = [22, 19, 28]
        file_numbers_test = [25, 21, 1] 

    
    elif k_fold == 3:
        file_numbers_train = [6, 7, 8, 9, 11, 12, 13, 14, 19, 21, 25, 29, 30, 32, 33, 35, 36, 37, 22, 5, 13, 10, 17, 38, 23]
        file_numbers_val = [4, 31, 1]
        file_numbers_test = [34, 15, 28]


    elif k_fold == 4:
        file_numbers_train = [4, 6, 8, 9, 11, 12, 14, 19, 25, 29, 30, 31, 32, 33, 34, 36, 37, 22, 5, 10, 17, 28, 38, 23]
        file_numbers_val = [7, 15, 13]
        file_numbers_test = [35, 21, 1]

    elif k_fold == 5:
        file_numbers_train = [4, 6, 8, 9, 11, 12, 13, 14, 25, 30, 31, 32, 33, 34, 36, 37, 1, 22, 5, 13, 10, 17, 28, 38, 15]
        file_numbers_val = [7, 21, 29]
        file_numbers_test = [35, 19, 23]
        
    # these are random created folds - not used
        
    elif k_fold == 6:
        file_numbers_train = [4, 6, 8, 9, 11, 12, 13, 14, 25, 30, 31, 32, 33, 34, 36, 37, 1, 22, 5, 13, 10, 17, 28, 38, 15]
        file_numbers_val = [7, 10, 13, 17]  
        file_numbers_test = [9, 22] 
    
    else:
        raise Exception("wrong k-fold selected")
    
    return file_numbers_train, file_numbers_val, file_numbers_test


def get_torch_data(data_X: np.ndarray, data_y: np.ndarray):
    data_X: torch.Tensor = torch.from_numpy(data_X)
    data_y: torch.Tensor = torch.from_numpy(data_y)
    # this is to get same dimensions for X and y data
    data_y = data_y.unsqueeze(1).unsqueeze(1)
    
    data_X = data_X.float()
    data_y = data_y.float()
    return data_X, data_y


class StandardScaler():
    def __init__(self, train_X, train_y):

        self.data_mean = np.mean(train_X, axis=(0, 1), keepdims=True)
        self.data_std = np.std(train_X, axis=(0, 1), keepdims=True)

        self.y_mean = np.mean(train_y, keepdims=True)
        self.y_std = np.std(train_y, keepdims=True)

    def standardize_X(self, data_X: np.ndarray) -> np.ndarray:
        """ standardizes input data with mean and std from train set"""
        data_X_scaled = (data_X - self.data_mean) / self.data_std
        return data_X_scaled
        
    def standardize_y(self, data_y: np.ndarray) -> np.ndarray:
        """ standardizes target data with mean and std from train set"""
        data_y_scaled = (data_y - self.y_mean) / self.y_std
        return data_y_scaled
    
    

class MinMaxScaler():
    def __init__(self, train_X, train_y):
        """
        Initializes the scaler with the min and max values of the training data.
        """
        self.data_min = np.min(train_X, axis=(0, 1), keepdims=True)
        self.data_max = np.max(train_X, axis=(0, 1), keepdims=True)
        self.y_min = np.min(train_y, keepdims=True)
        self.y_max = np.max(train_y, keepdims=True)

    def scale_X(self, data_X: np.ndarray) -> np.ndarray:
        """
        Scales input data to the range [-1, 1] using min and max from the training set.
        """
        data_X_scaled = 2 * (data_X - self.data_min) / (self.data_max - self.data_min) - 1
        return data_X_scaled

    def scale_y(self, data_y: np.ndarray) -> np.ndarray:
        """
        Scales target data to the range [-1, 1] using min and max from the training set.
        """
        data_y_scaled = 2 * (data_y - self.y_min) / (self.y_max - self.y_min) - 1
        return data_y_scaled





