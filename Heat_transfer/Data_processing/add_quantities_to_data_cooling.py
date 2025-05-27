# Imported modules
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Open the TDMS file and read all data
save_folder_filtered_data = 'stored_folder'

file_path = "file_path"
file_name = os.path.splitext(os.path.basename(file_path))[0]


# Read the CSV file
data = pd.read_csv(file_path, header=0)

# Extract columns
'''
time = data.iloc[:, 0]
T = data.iloc[:, 1]
T_env = np.round(np.average(data.iloc[:, 2]), 1)
'''

time = data.iloc[:, 0]
L = data.iloc[:, 1]
R = data.iloc[:, 2]
m = data.iloc[:, 3]
c_p = data.iloc[:, 4]
T_env = data.iloc[:, 5]
T = data.iloc[:, 7]
T_max = np.max(T)


# Add additional data to all data files for input purposes
# Constants
g_0 = 9.80665 # Gravitational acceleration
R_a = 8.3144 # Gas constant
P_a = 101325/1e5 # Standard pressure at sea level

'''
# Cup
L = 70.8e-3
R = 65.2e-3/2
m = 192e-3
c_p = 0.84
material = 1
'''


# Save to CSV
df = pd.DataFrame({
    "Time": time,
    'Length': L ,
    'Radius': R,
    'Wet mass' : m,
    'Cp' : c_p,
    "Ambient temperature": T_env,
    "Starting temperature": T_max,
    "Object temperature": T
    
})




filt_data_filename = os.path.join(save_folder_filtered_data, f"{file_name}.csv")
#filt_data_filename = os.path.join(save_folder_filtered_data, f"{file_name}_prepared.csv")
output_csv_path = filt_data_filename
df.to_csv(output_csv_path, index=False)
