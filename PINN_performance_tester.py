# Imported modules
import torch
import numpy as np
import matplotlib.pyplot as plt
from Validation_class import prep_data
from Input_generator_SRM import SRM_input
from NN_tools_thesis import NN
import time



# Open the trained model file 
file_path = "/Users/tristanhirs/Downloads/Thesis/Python code/PINN_trained_models/SRM_PINN_trained.pth"

# Load the trained PINN model
SRM_PINN_Trained = torch.load(file_path)

learn_parameters = {
    "t_shift" : 1,
    "a" : 5.13,
    "n" : 0.222
    }


# Create an instance of the model
SRM_PINN_Trained = NN(18, 1, 128, 10, epochs=400, batch_size=4, use_batch=True, lr=1e-4, loss_terms=None, learn_params=None)

# Load the state dict into the model
SRM_PINN_Trained.load_state_dict(torch.load(file_path))



# Constants
g_0 = 9.80665 # Gravitational acceleration
R_a = 8.3144 # Gas constant
P_a = 101325/1e5 # Standard pressure at sea level


# Generate inputs for network
time_plot = np.arange(0, 10, 1e-2)

# Grain 
L_grain = 107e-3 # Length of grain [m]
d_grain_in = 25e-3 # Port diameter grain [m]
t_liner = 1.6e-3 # Wall thickness of the liner
d_grain_out = 80e-3 - 2 * t_liner # Outside diameter of grain [m]
r_grain_init = d_grain_in / 2 # Port radius [m]
#mass_grain_init_1 = L * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2) * rho_grain # Inital mass of grain [kg]
mass_grain_init = 758e-3 # [kg]
rho_grain =  mass_grain_init / ( L_grain * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2))# Density of grain [kg/m3]
N_grains = 1
a = 5.13
n = 0.222


# Inhibiters
inhibited_core = 0
inhibited_sides = 1

# Combustion mixture (predicted using CEA)
gamma =  1.137 # Specific heat of combustion mixture [-]
W_g = 39.9e-3  # Molecular mass of mixture [kg]
R_g = R_a / W_g 
T_0 = 1600 # Temperature of combustion mixture [K]


# Motor specifications
d_chamber = 80e-3 # Outside diamter of the chamber [m]
L_chamber = 118.5e-3 # Length of the chamber [m]
d_t = 8.73e-3
r_t = d_t/2 # Nozzle throat radius [m]
d_e = 16.73e-3
r_e = d_e/2 # Nozzle exit radius [m]
A_t = np.pi * r_t**2  # Nozzle throat area [m]
A_e = np.pi * r_e**2 # Nozzle exit area [m]


input_plot_scaled = SRM_input(time_plot, L_grain, d_grain_in, d_grain_out, t_liner, mass_grain_init, N_grains, a, n, inhibited_core, inhibited_sides, gamma, W_g, T_0, d_chamber, L_chamber, d_t, d_e, scaled=True)




# Start time
start_time = time.time()

# Predictions from network
with torch.no_grad():
    P_0_predictions = SRM_PINN_Trained.predict(input_plot_scaled)


# Plot output
plt.figure()
plt.plot(time_plot, P_0_predictions)
plt.xlabel("Time [s]")
plt.ylabel("Chamber pressure [bar]")
plt.grid(which='both', linestyle='--')
plt.show()

# End time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:f} seconds")








