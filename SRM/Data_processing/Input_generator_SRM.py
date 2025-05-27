# Imported modules
import numpy as np
#from pycea import CEA # Try to implement this later



def SRM_input(time, L_grain, D_grain_in, D_grain_out, t_liner, mass_grain, N_grains, uninhibited_core, uninhibited_sides, N_coated_cores, Gamma, W_g, T_0, D_chamber, L_chamber, D_nozzle_throat, D_nozzle_exit, Type_of_igniter, a, n, scaled=True):
    
    # Array of constant inputs
    constant_inputs = np.array([L_grain, D_grain_in, D_grain_out, t_liner, mass_grain, N_grains, uninhibited_core, uninhibited_sides, N_coated_cores, Gamma, W_g, T_0, D_chamber, L_chamber, D_nozzle_throat, D_nozzle_exit, Type_of_igniter, a, n])
 
    # Repeat the constants for each time step
    constant_features_repeated = np.tile(constant_inputs, (len(time), 1))  # Shape will be (number of time points, 13)

    # Reshape the time feature to make it compatible for concatenation
    time_feature = time.reshape(-1, 1)  # Shape (number of time points, 1)

    # Concatenate the constant features with the time feature along columns
    input_data = np.hstack([time_feature ,constant_features_repeated])
    
    if scaled:
        # Scaled input data
        return np.log1p(input_data)
    else:
        return input_data
    

'''
    
# Test code

t = np.arange(0, 5, 1)

# Grain 
L_grain = 50e-3
d_grain_in = 25e-3 # Port diameter grain [m]
t_liner = 1.6e-3 # Wall thickness of the liner
d_grain_out = 80e-3 - 2 * t_liner # Outside diameter of grain [m]
mass_grain_init = 758e-3 # [kg]
rho_grain =  mass_grain_init / ( L_grain * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2))# Density of grain [kg/m3]
N_grains = 2
a = 5.13
n = 0.222


# Inhibiters
inhibited_core = 0
inhibited_sides = 1

# Combustion mixture (predicted using CEA)
gamma =  1.137 # Specific heat of combustion mixture [-]
W_g = 39.9e-3  # Molecular mass of mixture [kg]
T_0 = 1600 # Temperature of combustion mixture [K]


# Motor specifications
d_chamber = 80e-3 # Outside diamter of the chamber [m]
L_chamber = 118.5e-3 # Length of the chamber [m]
d_t = 8.73e-3
d_e = 16.73e-3




test = SRM_input(t, L_grain, d_grain_in, d_grain_out, t_liner, mass_grain_init, N_grains, a, n, inhibited_core, inhibited_sides, gamma, W_g, T_0, d_chamber, L_chamber, d_t, d_e, scaled=True)
'''







