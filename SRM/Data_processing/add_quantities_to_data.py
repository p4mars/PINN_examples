# Imported modules
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Open the TDMS file and read all data
save_folder_filtered_data = 'save_path'

file_path = "file_path"
file_name = os.path.splitext(os.path.basename(file_path))[0]


# Read the CSV file
data = pd.read_csv(file_path, header=0)

# Extract columns
time = data.iloc[:, 0]
pressure = data.iloc[:, 1]
thrust = data.iloc[:, 2]


# Add additional data to all data files for input purposes
# Constants
g_0 = 9.80665 # Gravitational acceleration
R_a = 8.3144 # Gas constant
P_a = 101325/1e5 # Standard pressure at sea level

# Grain 
L_grain = 60e-3 # Length of grain [m]
d_grain_in = 20e-3 # Port diameter grain [m]
t_liner = 1.75e-3 # Wall thickness of the liner
d_grain_out = 43.5e-3 - 2 * t_liner # Outside diameter of grain [m]
#r_grain_init = d_grain_in / 2 # Port radius [m]
#mass_grain_init_1 = L * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2) * rho_grain # Inital mass of grain [kg]
mass_grain = 97e-3 # [kg]
#rho_grain =  mass_grain_init / ( L_grain * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2))# Density of grain [kg/m3]
N_grains = 4

N_coated_cores = 4

a = 5.13
n = 0.222


# Inhibiters
uninhibited_core = 1
uninhibited_sides = 1

# Combustion mixture (predicted using CEA)
gamma =  1.137 # Specific heat of combustion mixture[-]
W_g = 39.9e-3  # Molecular mass of mixture
R_g = R_a / W_g 
T_0 = 1600 # Temperature of combustion mixture


# Motor specifications
d_chamber = 44e-3 # Outside diamter of the chamber [m]
L_chamber = 275e-3 # Length of the chamber [m]
d_t = 11e-3 # Nozzle throat radius [m]
d_e = 22e-3 # Nozzle exit radius [m]


#A_t = np.pi * r_t**2  # Nozzle throat area [m]
#A_e = np.pi * r_e**2 # Nozzle exit area [m]
igniter_type = 3



# Save to CSV
df = pd.DataFrame({
    "time": time,
    "Length of the grain" : L_grain,
    "Inner diameter of the grain" : d_grain_in,
    "Outer diameter of the grain" : d_grain_out,
    "Liner thickness of the grain" : t_liner,
    "Density of the grain" : mass_grain,
    "Number of grains" : N_grains, 
    "Uninhibited core" : uninhibited_core,
    "Uninhibited sides" : uninhibited_sides, 
    "Number of coated cores" : N_coated_cores,
    "gamma" : gamma,
    "Molecular weight of combustion gas" : W_g,
    "Combustion temperature" : T_0,
    "Diameter of the chamber": d_chamber,
    "Lenght of the Chamber" : L_chamber,
    "Diameter of the nozzle throat" : d_t,
    "Diameter of the nozzle exit" : d_e, 
    "Type of igniter" : igniter_type,
    "a" : a,
    "n" : n,
    "pressure": pressure,
    "thrust": thrust
})




filt_data_filename = os.path.join(save_folder_filtered_data, f"{file_name}_prepared.csv")
output_csv_path = filt_data_filename
df.to_csv(output_csv_path, index=False)