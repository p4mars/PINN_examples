# Imported modules
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
import pandas as pd
import os
import numpy as np
import torch
import os
import shutil
import time as m_time
import itertools

np.random.seed(42)


def move_datasets_for_loo_cv(source_dir, test_dir):
    """
    Perform Leave-One-Out Cross-Validation by moving one dataset at a time
    from source_dir to test_dir, ensuring each dataset is used as a test set once.
    """
    # List all files in the source directory
    datasets = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    if not datasets:
        print("No datasets found in the source directory.")
        return

    for test_file in datasets:
        # Move the current test file to the test directory
        test_file_path = os.path.join(source_dir, test_file)
        target_test_path = os.path.join(test_dir, test_file)

        print(f"Moving {test_file} to test directory...")
        shutil.move(test_file_path, target_test_path)

        # Yield used to put files back after use
        yield test_file

        # After processing, move the test file back to the source directory
        print(f"Moving {test_file} back to source directory...")
        shutil.move(target_test_path, test_file_path)

class surrogate_testing():
    @staticmethod
    def get_group_folders(base_path, selected_folders):
        """
        Retrieves specified group folders from the base path.
        """
        all_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        return [os.path.join(base_path, f) for f in all_folders if f in selected_folders]
    
    @staticmethod
    def generate_train_test_splits(group_folders, train_group_count):
        """
        Generates train-validation splits ensuring each group is used for validation at least once.
        """
        all_splits = []
        for train_groups in itertools.combinations(group_folders, train_group_count):
            val_groups = [g for g in group_folders if g not in train_groups]
            all_splits.append((list(train_groups), val_groups))
        return all_splits

    @staticmethod
    def copy_files(src_folders, dest_folder):
        """
        Copies all files from source folders to the destination folder.
        """
        os.makedirs(dest_folder, exist_ok=True)  # Ensure the folder exists

        for folder in src_folders:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):  # Ensure it's a file
                    shutil.copy(file_path, os.path.join(dest_folder, file))
    
    @staticmethod
    def get_file_paths(folders):
        """
        Returns a list of full file paths from a list of folders.
        """
        file_paths = []
        for folder in folders:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    file_paths.append(file_path)
        return file_paths
    
    @staticmethod
    def delete_files(folder):
        """
        Deletes all files in a folder and removes the folder.
        """
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Remove each file




class prep_data():
    "Class to perform validation on model"
    def __init__(self, data_directory, scaled=False, n_outputs=2, n_samples=10, start_time=None, end_time=None):
        self.data_dir = data_directory
        self.scaled = scaled
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.start_time = start_time
        self.end_time = end_time
        
        
    # Function to load CSV files
    def load_csv_files(self):
        """Loads all CSV files from the given directory."""
        all_data = []
        
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.data_dir, file_name)
                data = pd.read_csv(file_path)
                all_data.append(data)
        return all_data
    
    def data_X_y(self):
        all_data_files = self.load_csv_files()
        
        input_lst = []
        output_lst = []
        
        time_column = 'time'
        
        for data in all_data_files:
            # Filter data to include only rows within the specified time frame
            if self.start_time is not None and self.end_time is not None:
                # Ensure the time_column exists and filter based on it
                if time_column in data.columns:
                    data = data[(data[time_column] >= self.start_time) & (data[time_column] <= self.end_time)]
                    
            
                
            
            # Randomly sample n_samples rows from the filtered data
            if len(data) >= self.n_samples:
                #sampled_data = data.sample(n=self.n_samples, random_state=42)
                sampled_data = data.iloc[np.linspace(0, len(data) - 1, self.n_samples).astype(int)]
            else:
                # If there aren't enough samples after filtering, use all available rows
                sampled_data = data
                
            # Choose the correct output columns based on n_outputs
            if self.n_outputs == 1:
                outputs = sampled_data.iloc[:, 1]
            else:
                outputs = sampled_data.iloc[:, 1:]
            
            # Scale inputs and outputs if required
            if self.scaled:
                inputs = np.log1p(sampled_data.iloc[:, :1])
            else:
                inputs = sampled_data.iloc[:, :1]
            
            
                
            input_lst.append(inputs)
            output_lst.append(outputs)
        
        return np.concatenate(input_lst), np.concatenate(output_lst)
        
    
    def train_val_split(self, val_size=0.2):
        "Split data into train and validation data"    
        X, y = self.data_X_y()
        
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
        
        # Print the shapes of the datasets
        print('Training data')
        print("Training data shape (X_train):", X_train.shape)
        print("Training labels shape (y_train):", y_train.shape)
        print("Validation data")
        print("Validation data shape (X_val):", X_val.shape)
        print("Validation labels shape (y_val):", y_val.shape)
        
        return X_train, y_train, X_val, y_val
    
    def compare_with_data(self, PINN=None):
        
        # Get all data points
        X_all, y_all = self.data_X_y()
        
        # Obtain predictions from PINN
        if PINN is not None:
            Predictions = PINN(X_all)
        else:
            Predictions = None
        
        # Compare predictions and provide average error
        if PINN is not None:
        
            if self.n_outputs == 1:
                error_average = np.sqrt(np.mean((Predictions.reshape(-1) - y_all)**2))
                error_max = np.sqrt(np.max((Predictions.reshape(-1) - y_all)**2))
                
                # Print avergae error
                print(f"Average error pressure: {error_average:f} [bar]")
    
                # Print max error
                print(f"Max error pressure: {error_max:f} [bar]")
                
            
            else:
                error_average_pressure = np.sqrt(np.mean((Predictions[:, 0].reshape(-1) - y_all[:, 0])**2))
                error_max_pressure = np.sqrt(np.max((Predictions[:, 0].reshape(-1) - y_all[:, 0])**2))
                
                error_average_thrust = np.sqrt(np.mean((Predictions[:, 1].reshape(-1) - y_all[:, 1])**2))
                error_max_thrust = np.sqrt(np.max((Predictions[:, 1].reshape(-1) - y_all[:, 1])**2))
                
                error_average = [error_average_pressure, error_average_thrust]
                error_max = [error_max_pressure, error_max_thrust]
                
                # Print avergae error
                print(f"Average error pressure: {error_average_pressure:f} [bar]")
                print(f"Average error thrust: {error_average_thrust:f} [N]")
    
                # Print max error
                print(f"Max error pressure: {error_max_pressure:f} [bar]")
                print(f"Max error thrust: {error_max_thrust:f} [N]")
            
        else:
            error_average = None
            error_max = None

        return X_all, y_all, Predictions, error_average, error_max
    
    
class prep_data_surrogate():
    "Class to perform validation on model"
    def __init__(self, data_directory, scaled=False, n_outputs=2, n_samples=10, start_time=None, end_time=None):
        self.data_dir = data_directory
        self.scaled = scaled
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.start_time = start_time
        self.end_time = end_time
        
        
    # Function to load CSV files
    def load_csv_files(self):
        """Loads all CSV files from the given directory."""
        all_data = []
        
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.data_dir, file_name)
                data = pd.read_csv(file_path)
                all_data.append(data)
        return all_data
    
    def data_X_y(self):
        all_data_files = self.load_csv_files()
        
        input_lst = []
        output_lst = []
        
        time_column = 'time'
        
        for data in all_data_files:
            # Filter data to include only rows within the specified time frame
            if self.start_time is not None and self.end_time is not None:
                # Ensure the time_column exists and filter based on it
                if time_column in data.columns:
                    data = data[(data[time_column] >= self.start_time) & (data[time_column] <= self.end_time)]
                    
            
                
            
            # Randomly sample n_samples rows from the filtered data
            if len(data) >= self.n_samples:
                #sampled_data = data.sample(n=self.n_samples, random_state=42)
                sampled_data = data.iloc[np.linspace(0, len(data) - 1, self.n_samples).astype(int)]
            else:
                # If there aren't enough samples after filtering, use all available rows
                sampled_data = data
                
            # Choose the correct output columns based on n_outputs
            if self.n_outputs == 1:
                outputs = sampled_data.iloc[:, 20]
            else:
                outputs = sampled_data.iloc[:, 20:]
            
            # Scale inputs and outputs if required
            if self.scaled:
                inputs = np.log1p(sampled_data.iloc[:, :20])
            else:
                inputs = sampled_data.iloc[:, :20]
            
            
                
            input_lst.append(inputs)
            output_lst.append(outputs)
        
        return np.concatenate(input_lst), np.concatenate(output_lst)
        
    
    def train_val_split(self, val_size=0.2, comb_data=False):
        "Split data into train and validation data"    
        X, y = self.data_X_y()
        
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
        
        if comb_data:
            # Combine the training and validation data
            X_train = np.concatenate((X_train, X_val), axis=0)
            y_train = np.concatenate((y_train, y_val), axis=0)
        
        # Print the shapes of the datasets
        #print('Training data')
        #print("Training data shape (X_train):", X_train.shape)
        #print("Training labels shape (y_train):", y_train.shape)
        #print("Validation data")
        #print("Validation data shape (X_val):", X_val.shape)
        #print("Validation labels shape (y_val):", y_val.shape)
        
        return X_train, y_train, X_val, y_val
    
    def compare_with_data(self, PINN=None):
        
        # Get all data points
        X_all, y_all = self.data_X_y()
        
        # Obtain predictions from PINN
        if PINN is not None:
            # Start time
            start_time = m_time.time()

            # Obtain predictions
            Predictions = PINN(X_all)
            
            # End time
            end_time = m_time.time()

            # Calculate the elapsed time
            execution_time = end_time - start_time
        else:
            Predictions = None
            execution_time = None
        
        # Compare predictions and provide average error
        if PINN is not None:
        
            if self.n_outputs == 1:
                error_average = np.sqrt(np.mean((Predictions.reshape(-1) - y_all)**2))
                error_max = np.sqrt(np.max((Predictions.reshape(-1) - y_all)**2))
                
                # Print avergae error
                print(f"Average error pressure: {error_average:f} [bar]")
    
                # Print max error
                print(f"Max error pressure: {error_max:f} [bar]")
                
            
            else:
                error_average_pressure = np.sqrt(np.mean((Predictions[:, 0].reshape(-1) - y_all[:, 0])**2))
                error_max_pressure = np.sqrt(np.max((Predictions[:, 0].reshape(-1) - y_all[:, 0])**2))
                error_peak_pressure = np.sqrt((np.max(Predictions[:, 0].reshape(-1)) - np.max(y_all[:, 0]))**2) 
                
                error_average_thrust = np.sqrt(np.mean((Predictions[:, 1].reshape(-1) - y_all[:, 1])**2))
                error_max_thrust = np.sqrt(np.max((Predictions[:, 1].reshape(-1) - y_all[:, 1])**2))
                error_peak_thrust = np.sqrt((np.max(Predictions[:, 1].reshape(-1)) - np.max(y_all[:, 1]))**2)
                
                error_average = [error_average_pressure, error_average_thrust]
                error_max = [error_max_pressure, error_max_thrust]
                error_peak = [error_peak_pressure, error_peak_thrust]
                
                # Print avergae error
                print(f"Average error pressure: {error_average_pressure:f} [bar]")
                print(f"Average error thrust: {error_average_thrust:f} [N]")
    
                # Print max error
                print(f"Max error pressure: {error_max_pressure:f} [bar]")
                print(f"Max error thrust: {error_max_thrust:f} [N]")
                
                # Print peak errors
                print(f"Peak error pressure: {error_peak_pressure:f} [bar]")
                print(f"Peak error thrust: {error_peak_thrust:f} [N]")
            
        else:
            error_average = None
            error_max = None
            error_peak = None

        return X_all, y_all, Predictions, error_average, error_max, error_peak, execution_time
    
    

   
    
class prep_data_cooling():
    "Class to perform validation on model"
    def __init__(self, data_directory, scaled_inputs=False, scaled_outputs=False, n_outputs=1, n_samples=10, start_time=None, end_time=None):
        self.data_dir = data_directory
        self.scaled_inputs = scaled_inputs
        self.scaled_outputs = scaled_outputs
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.start_time = start_time
        self.end_time = end_time
        
    # Function to load CSV files
    def load_csv_files(self):
        """Loads all CSV files from the given directory."""
        all_data = []
        
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.data_dir, file_name)
                data = pd.read_csv(file_path)
                all_data.append(data)
        return all_data
    
    def data_X_y(self, all_data=False):
        all_data_files = self.load_csv_files()
        
        input_lst = []
        output_lst = []
        
        
        time_column = 'time'
        
        for data in all_data_files:
            # Filter data to include only rows within the specified time frame
            if self.start_time is not None and self.end_time is not None:
                # Ensure the time_column exists and filter based on it
                if time_column in data.columns:
                    data = data[(data[time_column] >= self.start_time) & (data[time_column] <= self.end_time)]
                    
            
                
            
            # Randomly sample n_samples rows from the filtered data
            if (len(data) >= self.n_samples) and (all_data is False):
                #sampled_data = data.sample(n=self.n_samples, random_state=42)
                sampled_data = data.iloc[np.linspace(0, len(data) - 1, self.n_samples).astype(int)]
            else:
                # If there aren't enough samples after filtering, use all available rows
                sampled_data = data
            
            
            
            if self.scaled_outputs:
                outputs = np.log1p(sampled_data.iloc[:,1])
            else:
                outputs = sampled_data.iloc[:,1]
            
            
            '''
            # Choose the correct output columns based on n_outputs
            if self.n_outputs == 1:
                outputs = sampled_data.iloc[:, 20]
            else:
                outputs = sampled_data.iloc[:, 20:]
            '''
            
            # Scale inputs and outputs if required
            if self.scaled_inputs:
                inputs = np.log1p(sampled_data.iloc[:, 0])
            else:
                inputs = sampled_data.iloc[:, 0]
            
            
                
            input_lst.append(inputs)
            output_lst.append(outputs)
        
        return np.concatenate(input_lst), np.concatenate(output_lst)
        
    
    def train_val_split(self, val_size=0.2, comb_data=False):
        "Split data into train and validation data"    
        X, y = self.data_X_y()
        
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
        
        if comb_data:
            # Combine the training and validation data
            X_train = np.concatenate((X_train, X_val), axis=0)
            y_train = np.concatenate((y_train, y_val), axis=0)
            
        
        # Print the shapes of the datasets
        print('Training data')
        print("Training data shape (X_train):", X_train.shape)
        print("Training labels shape (y_train):", y_train.shape)
        print("Validation data")
        print("Validation data shape (X_val):", X_val.shape)
        print("Validation labels shape (y_val):", y_val.shape)
        
        return X_train, y_train, X_val, y_val
    
    def compare_with_data(self, PINN=None):
        
        # Get all data points
        X_all, y_all = self.data_X_y(all_data=True)
        
        # Obtain predictions from PINN
        if PINN is not None:
            Predictions = PINN(X_all)
        else:
            Predictions = None
        
        # Compare predictions and provide average error
        if PINN is not None:
            error_average = np.sqrt(np.mean((Predictions.reshape(-1) - y_all)**2))
            error_max = np.sqrt(np.max((Predictions.reshape(-1) - y_all)**2))
        else:
            error_average = None
            error_max = None

        return X_all, y_all, Predictions, error_average, error_max     
        
    
   
    
class prep_data_cooling_surrogate():
    "Class to perform validation on model"
    def __init__(self, data_directory, scaled=False, n_outputs=1, n_samples=10, start_time=None, end_time=None, long_time=None):
        self.data_dir = data_directory
        self.scaled = scaled
        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.start_time = start_time
        self.end_time = end_time
        self.long_time = long_time
        
    # Function to load CSV files
    def load_csv_files(self):
        """Loads all CSV files from the given directory."""
        all_data = []
        
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.csv'):
                file_path = os.path.join(self.data_dir, file_name)
                data = pd.read_csv(file_path)
                all_data.append(data)
        return all_data
    
    def data_X_y(self):
        all_data_files = self.load_csv_files()
        
        input_lst = []
        output_lst = []
        
        time_column = 'Time'
        
        for data in all_data_files:
            # Filter data to include only rows within the specified time frame
            if self.start_time is not None and self.end_time is not None:
                # Ensure the time_column exists and filter based on it
                if time_column in data.columns:
                    data = data[(data[time_column] >= self.start_time) & (data[time_column] <= self.end_time)]
                    
            
                
            
            # Randomly sample n_samples rows from the filtered data
            if len(data) >= self.n_samples:
                #sampled_data = data.sample(n=self.n_samples, random_state=42)
                sampled_data = data.iloc[np.linspace(0, len(data) - 1, self.n_samples).astype(int)]
            else:
                # If there aren't enough samples after filtering, use all available rows
                sampled_data = data
            
            outputs = sampled_data.iloc[:,7]
            
            '''
            # Choose the correct output columns based on n_outputs
            if self.n_outputs == 1:
                outputs = sampled_data.iloc[:, 20]
            else:
                outputs = sampled_data.iloc[:, 20:]
            '''
            
            # Scale inputs and outputs if required
            if self.scaled:
                inputs = np.log1p(sampled_data.iloc[:, :7])
            else:
                inputs = sampled_data.iloc[:, :7]
            
                
            input_lst.append(inputs)
            output_lst.append(outputs)
        
        return np.concatenate(input_lst), np.concatenate(output_lst)
        
    
    def train_val_split(self, val_size=0.2, comb_data=False):
        "Split data into train and validation data"    
        X, y = self.data_X_y()
        
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
        
        if comb_data:
            # Combine the training and validation data
            X_train = np.concatenate((X_train, X_val), axis=0)
            y_train = np.concatenate((y_train, y_val), axis=0)
        
        # Print the shapes of the datasets
        #print('Training data')
        #print("Training data shape (X_train):", X_train.shape)
        #print("Training labels shape (y_train):", y_train.shape)
        #print("Validation data")
        #print("Validation data shape (X_val):", X_val.shape)
        #print("Validation labels shape (y_val):", y_val.shape)
        
        return X_train, y_train, X_val, y_val
    
    def compare_with_data(self, PINN=None):
        
        # Get all data points
        X_all, y_all = self.data_X_y()
        
        if self.long_time is not None:
            time = np.arange(0, self.long_time, 1).reshape(-1, 1)
            
            inputs = X_all[0][1:]
            
            # Repeat the constants for each time step
            constant_features_repeated = np.tile(inputs, (time.shape[0], 1))  # Repeat inputs for each time step
            
            # Concatenate the constant features with the time feature along the second axis
            X_all = np.concatenate((time, constant_features_repeated), axis=1)

        else:
            X_all = X_all
        
        # Obtain predictions from PINN
        if PINN is not None:
            # Start time
            start_time = m_time.time()

            # Obtain predictions
            Predictions = PINN(X_all)
            
            # End time
            end_time = m_time.time()

            # Calculate the elapsed time
            execution_time = end_time - start_time
        else:
            Predictions = None
            execution_time = None
        
        # Compare predictions and provide average error
        if PINN is not None and self.long_time is None:
            error_average = np.sqrt(np.mean((Predictions.reshape(-1) - y_all)**2))
            error_max = np.sqrt(np.max((Predictions.reshape(-1) - y_all)**2))
            
            # Print avergae error
            print(f"Average error: {error_average:f} [deg C]")

            # Print max error
            print(f"Max error: {error_max:f} [deg C]")
            
        else:
            error_average = None
            error_max = None

        return X_all, y_all, Predictions, error_average, error_max, execution_time
    
   
    
   
    
'''
# Test code
X, y = datasets.load_iris(return_X_y=True)
test_model = DecisionTreeClassifier(random_state=42)

Validation_runs(test_model, None).K_fold_function(X, y, 20)

'''

'''
 
    def K_fold_function(self, X, y, n_split=0):
        # Setup Kfold validation and determine number of splits
        scores = cross_val_score(self.model, X, y, cv = KFold(n_splits=n_split))
    
        print("Cross Validation Scores: ", scores)
        print("Average CV Score: ", scores.mean())
        print("Number of CV Scores used in Average: ", len(scores))
        
        return scores, scores.mean()

'''