# Imported modules
from NN_tools_thesis import NN, gradient
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Validation_class import prep_data_surrogate, move_datasets_for_loo_cv
from Loss_function_tools import SRM_helper_tools
import time
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, curve_fit
import pandas as pd
import os

# Constants
g_0 = 9.80665 # Gravitational acceleration
R_a = 8.3144 # Gas constant
P_a = 101325/1e5  # Standard pressure at sea level in bar

# Import training data
# Multiple file usage
data_dir = '/Users/tristanhirs/Downloads/Thesis/Motor test data/Training data/'  # Path to CSV file(s)
data_dir_val = '/Users/tristanhirs/Downloads/Thesis/Motor test data/Validation_data/'

# Lists to save performance
average_error_pressure_lst = []
max_error_pressure_lst = []
peak_error_pressure_lst = []
average_error_thrust_lst = []
max_error_thrust_lst = []
peak_error_thrust_lst = []
train_time_lst = []
execution_time_lst = []
test_file_lst = []

# - - - - - - - - - - - - - - - - - - - - PINN - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PINN Functions

# Define physics loss
def PINN_pressure_loss(NN_output: torch.nn.Module, used_inputs):
    # Collocation points time array
    start, end = 1e-16, 8
    steps = 100
    time_inputs = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    
    # Setup total loss
    total_loss = 0
    
    for inputs in used_inputs:
        inputs = torch.expm1(inputs)
        #time_shift = NN_output.t_shift
        time_shift = 0.0
        
        
        # Unzip constant values from inputs
        L_grain = inputs[0]
        d_grain_in = inputs[1]
        d_grain_out = inputs[2]
        t_liner = inputs[3]
        mass_grain = inputs[4]
        N_grains = inputs[5]
        uninhibited_core = inputs[6]
        uninhibited_sides = inputs[7]
        gamma = inputs[9]
        W_g = inputs[10]
        T_0 = inputs[11]
        d_chamber = inputs[12]
        L_chamber = inputs[13]
        d_t = inputs[14]
        a = inputs[17]
        n = inputs[18]
        

        # Perform calulcations for standard values
        R_g = R_a / W_g 
        A_t = np.pi * d_t**2/4  # Nozzle throat area [m]
        rho_grain =  mass_grain / ( L_grain * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2))# Density of grain [kg/m3]
        
        
        # Find best time shift
        time_inputs = time_inputs + time_shift

        # Repeat the constants for each time step (number of time points)
        # Assuming time_inputs is a tensor with shape (number of time points, 1)
        constant_features_repeated = inputs.unsqueeze(0).repeat(time_inputs.size(0), 1).requires_grad_(True)
    
        # Reshape the time feature to make it compatible for concatenation
        # time_feature is assumed to be of shape (number of time points, 1)
        time_feature = time_inputs.reshape(-1, 1)  # This should already be (number of time points, 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_feature, constant_features_repeated), dim=1).requires_grad_(True)
        
        # Scale
        input_data = torch.log1p(input_data)
    
        # Obtain predictions from the network (Pc, a, n)
        P_C = NN_output(input_data)[:, 0].unsqueeze(-1)

        # Obtain derivative of Pc with respect to the scaled time over the network and convert to dPc/dt
        dP_C_dt = gradient(P_C, input_data)[:, 0]
        
        # Obtain r_integrator
        radius = SRM_helper_tools.r_integrator(P_C, a, n, time_inputs, d_grain_in/2)
        
        # Check when motor is burning or not
        burning_logic = torch.logical_and(radius <= (d_grain_out/2), (radius - (d_grain_in/2)) <= L_grain/2)
        
        # Set radius to fixed values if all propellant is burned
        radius[radius >= (d_grain_out/2)] = d_grain_out/2
        radius[(radius - (d_grain_in/2)) >= L_grain/2] = L_grain/2

        # Obtain dPdt from formula
        dPdt_theory = SRM_helper_tools.dPdt_check(radius, P_C, P_a, burning_logic, a, n, rho_grain, R_g, T_0, A_t, gamma, uninhibited_core, uninhibited_sides, L_grain, d_grain_in/2, d_grain_out, N_grains, L_chamber, d_chamber, t_liner) * (1 + time_inputs)

        # Determine loss for specific case and add to total loss
        total_loss += torch.mean((dPdt_theory - dP_C_dt)**2) 

    return total_loss / len(used_inputs) 

def PINN_pressure_loss_learn(NN_output: torch.nn.Module, used_inputs):
    # Collocation points time array
    start, end = 1e-16, 8
    steps = 100
    time_inputs = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    
    # Setup total loss
    total_loss = 0
    
    for inputs in used_inputs:
        inputs = torch.expm1(inputs)
        #time_shift = NN_output.t_shift
        time_shift = 0.0
        
        
        # Unzip constant values from inputs
        L_grain = inputs[0]
        d_grain_in = inputs[1]
        d_grain_out = inputs[2]
        t_liner = inputs[3]
        mass_grain = inputs[4]
        N_grains = inputs[5]
        uninhibited_core = inputs[6]
        uninhibited_sides = inputs[7]
        gamma = inputs[9]
        W_g = inputs[10]
        T_0 = inputs[11]
        d_chamber = inputs[12]
        L_chamber = inputs[13]
        d_t = inputs[14]
        

        # Perform calulcations for standard values
        R_g = R_a / W_g 
        A_t = np.pi * d_t**2/4  # Nozzle throat area [m]
        rho_grain =  mass_grain / ( L_grain * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2))# Density of grain [kg/m3]
        
        
        # Find best time shift
        time_inputs = time_inputs + time_shift

        # Repeat the constants for each time step (number of time points)
        # Assuming time_inputs is a tensor with shape (number of time points, 1)
        constant_features_repeated = inputs.unsqueeze(0).repeat(time_inputs.size(0), 1).requires_grad_(True)
    
        # Reshape the time feature to make it compatible for concatenation
        # time_feature is assumed to be of shape (number of time points, 1)
        time_feature = time_inputs.reshape(-1, 1)  # This should already be (number of time points, 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_feature, constant_features_repeated), dim=1).requires_grad_(True)
        
        # Scale
        input_data = torch.log1p(input_data)
    
        # Obtain predictions from the network (Pc, a, n)
        P_C = NN_output(input_data)[:, 0].unsqueeze(-1)
        a = NN_output.a
        n = NN_output.n

        # Obtain derivative of Pc with respect to the scaled time over the network and convert to dPc/dt
        dP_C_dt = gradient(P_C, input_data)[:, 0]
        
        # Obtain r_integrator
        radius = SRM_helper_tools.r_integrator(P_C, a, n, time_inputs, d_grain_in/2)
        
        # Check when motor is burning or not
        burning_logic = torch.logical_and(radius <= (d_grain_out/2), (radius - (d_grain_in/2)) <= L_grain/2)
        
        # Set radius to fixed values if all propellant is burned
        radius[radius >= (d_grain_out/2)] = d_grain_out/2
        radius[(radius - (d_grain_in/2)) >= L_grain/2] = L_grain/2

        # Obtain dPdt from formula
        dPdt_theory = SRM_helper_tools.dPdt_check(radius, P_C, P_a, burning_logic, a, n, rho_grain, R_g, T_0, A_t, gamma, uninhibited_core, uninhibited_sides, L_grain, d_grain_in/2, d_grain_out, N_grains, L_chamber, d_chamber, t_liner) * (1 + time_inputs)
        
        
        # Determine loss for specific case and add to total loss
        total_loss += torch.mean((dPdt_theory - dP_C_dt)**2) 

    return total_loss / len(used_inputs) 



def PINN_thrust_loss(NN_output: torch.nn.Module, used_inputs):
    # Collocation points time array
    start, end = 1e-16, 8
    steps = 100
    time_inputs = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    
    # Setup total loss
    total_loss = 0

    for inputs in used_inputs:
        inputs = torch.expm1(inputs)
        #time_shift = NN_output.t_shift
        time_shift = 0.0
        
        
        # Unzip constant values from inputs
        gamma = inputs[9]
        d_t = inputs[14]
        d_e = inputs[15]


        # Perform calulcations for standard values
        A_t = np.pi * d_t**2/4  # Nozzle throat area [m]
        A_e = np.pi * d_e**2/4 # Nozzle exit area [m]
        
        
        
        # Find best time shift
        time_inputs = time_inputs + time_shift

        # Repeat the constants for each time step (number of time points)
        # Assuming time_inputs is a tensor with shape (number of time points, 1)
        constant_features_repeated = inputs.unsqueeze(0).repeat(time_inputs.size(0), 1)
    
        # Reshape the time feature to make it compatible for concatenation
        # time_feature is assumed to be of shape (number of time points, 1)
        time_feature = time_inputs.reshape(-1, 1)  # This should already be (number of time points, 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_feature, constant_features_repeated), dim=1)
        
        # Scaled input data
        input_data = torch.log1p(input_data)
    
        # Obtain predictions from the network (Pc, F, a, n)
        outputs_of_network = NN_output(input_data)
        P_C = outputs_of_network[:,0].unsqueeze(-1)
        F_network = outputs_of_network[:,1].unsqueeze(-1)
        
        # Obtain correct pressure ratio
        pepc = SRM_helper_tools.fsolve_PyTorch(SRM_helper_tools.AeAt_func, 0.001, args=(SRM_helper_tools.Gamma(gamma), gamma, A_e, A_t))
        
        # Calculate nozzle exit pressure
        P_e_sol = P_C*1e5 * pepc  # Units = [Pa]
        
        # Obtain thrsut coefficient
        C_f = SRM_helper_tools.Cf(SRM_helper_tools.Gamma(gamma), gamma, P_e_sol, P_C*1e5, P_a*1e5, (A_e / A_t))
        
        # Calculate theoretical thrust
        F_theory = C_f * P_C*1e5 * A_t # Units = [N]
        

        # Determine loss for specific case and add to total loss
        total_loss += torch.mean((F_theory - F_network)**2)     
    
    return total_loss / len(used_inputs)



# Define pressure boundary loss function (P_a at start)
def Boundary_loss_pressure(NN_output: torch.nn.Module, used_inputs):
    t_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)
    
    # Setup total loss
    total_loss = 0
    
    
    for inputs in used_inputs:
        inputs = torch.expm1(inputs)
        
        # Repeat the constants for each time step (number of time points)
        # Assuming time_inputs is a tensor with shape (number of time points, 1)
        constant_features_repeated = inputs.unsqueeze(0).repeat(t_boundary.size(0), 1)
    
        # Reshape the time feature to make it compatible for concatenation
        # time_feature is assumed to be of shape (number of time points, 1)
        time_feature = t_boundary.reshape(-1, 1)  # This should already be (number of time points, 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_feature, constant_features_repeated), dim=1)
        
        # Scaled input data
        input_data = torch.log1p(input_data)
    
        # Obtain predictions from the network (Pc, F, a, n)
        P_C = NN_output(input_data)[:, 0].unsqueeze(-1)
        
        total_loss += (torch.squeeze(P_C) - P_a)**2
        
    return total_loss / len(used_inputs)


# Define force boundary loss function (0 at start)
def Boundary_loss_force(NN_output: torch.nn.Module, used_inputs):
    t_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)
    
    # Setup total loss
    total_loss = 0
    
    
    for inputs in used_inputs:
        inputs = torch.expm1(inputs)
        
        # Repeat the constants for each time step (number of time points)
        # Assuming time_inputs is a tensor with shape (number of time points, 1)
        constant_features_repeated = inputs.unsqueeze(0).repeat(t_boundary.size(0), 1)
    
        # Reshape the time feature to make it compatible for concatenation
        # time_feature is assumed to be of shape (number of time points, 1)
        time_feature = t_boundary.reshape(-1, 1)  # This should already be (number of time points, 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_feature, constant_features_repeated), dim=1)
        
        # Scaled input data
        input_data = torch.log1p(input_data)
    
        # Obtain predictions from the network (Pc, F, a, n)
        F_network = NN_output(input_data)[:,1].unsqueeze(-1)
        
        total_loss += (torch.squeeze(F_network) - 0)**2
        
    return total_loss / len(used_inputs)



# Positive loss function
def Positive_loss_pressure(NN_output: torch.nn.Module, used_inputs):
    steps = 50
    time_inputs = torch.linspace(0, 8, steps).view(-1, 1).requires_grad_(True)
    
    # Setup total loss
    total_loss = 0
    
    for inputs in used_inputs:
        inputs = torch.expm1(inputs)
    
        # Repeat the constants for each time step (number of time points)
        # Assuming time_inputs is a tensor with shape (number of time points, 1)
        constant_features_repeated = inputs.unsqueeze(0).repeat(time_inputs.size(0), 1)
    
        # Reshape the time feature to make it compatible for concatenation
        # time_feature is assumed to be of shape (number of time points, 1)
        time_feature = time_inputs.reshape(-1, 1)  # This should already be (number of time points, 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_feature, constant_features_repeated), dim=1)
        
        # Scale
        input_data = torch.log1p(input_data)

        # Obtain predictions from the network (Pc, F, a, n)
        P_C = NN_output(input_data)[:, 0].unsqueeze(-1)
        
        total_loss += torch.mean(torch.clamp(-P_C, min=0)**2)

    return total_loss / len(used_inputs)


# Positive loss function
def Positive_loss_force(NN_output: torch.nn.Module, used_inputs):
    steps = 50
    time_inputs = torch.linspace(0, 8, steps).view(-1, 1).requires_grad_(True)
    
    # Setup total loss
    total_loss = 0
    
    for inputs in used_inputs:
        inputs = torch.expm1(inputs)
    
        # Repeat the constants for each time step (number of time points)
        # Assuming time_inputs is a tensor with shape (number of time points, 1)
        constant_features_repeated = inputs.unsqueeze(0).repeat(time_inputs.size(0), 1)
    
        # Reshape the time feature to make it compatible for concatenation
        # time_feature is assumed to be of shape (number of time points, 1)
        time_feature = time_inputs.reshape(-1, 1)  # This should already be (number of time points, 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_feature, constant_features_repeated), dim=1)
        
        # Scale
        input_data = torch.log1p(input_data)

        # Obtain predictions from the network (Pc, F, a, n)
        F_network = NN_output(input_data)[:,1].unsqueeze(-1)
        
        total_loss += torch.mean(torch.clamp(-F_network, min=0)**2)

    return total_loss / len(used_inputs)

# Initialize loss functions
loss_functions = {
    PINN_pressure_loss: 1e-6,
    Boundary_loss_pressure: 1,
    Positive_loss_pressure: 1,
    PINN_thrust_loss: 1e-3,
    Boundary_loss_force: 1,
    Positive_loss_force: 1,
    }

# Initialize loss functions
loss_functions_learn = {
    PINN_pressure_loss_learn: 1e-6,
    Boundary_loss_pressure: 1,
    Positive_loss_pressure: 1,
    PINN_thrust_loss: 1e-3,
    Boundary_loss_force: 1,
    Positive_loss_force: 1,
    }

# Initialize learnable paramaters
learn_parameters = {
    "a" : 5.13,
    "n" : 0.222
    }



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - Data-driven - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - Numerical Model - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Simulation module
def solid(t, y, L_grain, d_grain_in, t_liner, d_grain_out, r_grain_init, mass_grain_init, rho_grain, N_grains, a, n, uninhibited_core, uninhibited_sides, gamma, W_g, T_0, R_g, d_chamber, L_chamber, d_t, d_e, r_t, r_e, A_e, A_t):
    # Inital conditions
    P0 = y[0]
    r = y[1]
    
    
    # Burn surface area calculation and chamber volume
    A_b = ((uninhibited_core * (2 * np.pi * r * (L_grain - uninhibited_sides * 2 * (r - r_grain_init))) + uninhibited_sides * (2 * np.pi * ((d_grain_out**2 - (2 * r)**2) / 4)) )) * N_grains
    V_c = (uninhibited_core *  (np.pi * r**2 * (L_grain - uninhibited_sides * 2 * (r - r_grain_init))) + uninhibited_sides *( np.pi * (d_grain_out**2/4) * 2 * (r - r_grain_init))) * N_grains + (np.pi * (d_chamber**2/4) * L_chamber) - N_grains * (np.pi * ((d_grain_out + 2*t_liner)**2 / 4) * L_grain)
    
    
    if r >= d_grain_out/2 or (r - r_grain_init) >= L_grain/2:
        drdt = 0
        A_b = 0
        if P0 <= P_a:
            dP0dt = 0
        else:
            dP0dt = (- P0*1e5 * ( (A_t / V_c) * np.sqrt(gamma * R_g * T_0 * (2 / (gamma + 1) )**( (gamma + 1) / (gamma - 1) ))))/1e5
    else:
        dP0dt = (((A_b * (a/1e3) * (np.abs(P0)*1e5/1e6)**n)/V_c) * (rho_grain * R_g * T_0 - P0*1e5) - P0*1e5 * ( (A_t / V_c) * np.sqrt(gamma * R_g * T_0 * (2 / (gamma + 1) )**( (gamma + 1) / (gamma - 1) )) ))/1e5
        drdt = (a/1e3) * (np.abs(P0)*1e5/1e6)**n
    
    dydt = [dP0dt, drdt]
    return dydt

def Gamma(gamma):
    return np.sqrt(gamma) * (2 / (gamma + 1)) **((gamma + 1) / (2 * (gamma - 1)))

def AeAt(Gamma, gamma, Pe, Pc):
    return Gamma / np.sqrt(((2 * gamma) / (gamma - 1)) * (Pe / Pc)**(2 / gamma) *(1 - (Pe / Pc)**((gamma - 1) / gamma)))

def AeAtM(gamma, M):
    return (1 / M**2) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M**2))**((gamma + 1) / (gamma - 1))

def c_star(Gamma, R, Tc):
    return (1 / Gamma) * np.sqrt(R * Tc)
    
def Cf(Gamma, gamma, Pe, Pc, Pa, AeAt):
    return Gamma * np.sqrt(((2 * gamma) / (gamma - 1)) * (1 - (Pe / Pc)**((gamma - 1) / gamma))) + ((Pe / Pc) - (Pa / Pc)) * AeAt
    
def F(m_dot, Ue, Pe, Pa, Ae):
    return m_dot * Ue + (Pe - Pa) * Ae

def Ue(gamma, R, Tc, Pe, Pc):
    return np.sqrt(((2 * gamma) / (gamma - 1)) * R * Tc * (1 - (Pe / Pc)**((gamma - 1) / gamma)))

def AeAt_func(PePc, Gamma, gamma, Ae, At):
    return Gamma / np.sqrt(((2 * gamma) / (gamma - 1)) * (PePc)**(2 / gamma) *(1 - (PePc)**((gamma - 1) / gamma))) - (Ae / At)

def m_dot_id(Gamma, Pc, At, R, Tc):
    return (Gamma * Pc * At) / np.sqrt(R * Tc)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - MAIN PROGRAM - - - - - - - - - - - - - - - - - - - - - - - - -

# Perform Leave-One-Out Cross-Validation
for test_file in move_datasets_for_loo_cv(data_dir, data_dir_val):
    print(f"Current test dataset: {test_file}")
    
    # Obtain training and validation data
    X_train, y_train, X_val, y_val = prep_data_surrogate(data_dir, scaled=True, n_outputs=2, n_samples=100, start_time=None, end_time=None).train_val_split(comb_data=True)
    
    # PINN
    # Create network (re-init weights and biases)
    PINN_solid_rocket_motor_network = NN(20, 2, 64, 12, epochs=5000, batch_size=64, use_batch=False, lr=1e-3, loss_terms=loss_functions, learn_params=None, unique_input=True, activation_fn=nn.ELU, init_method=nn.init.kaiming_uniform_, show_plots=False)
    
    # Start time
    start_time = time.time()

    # Train network
    PINN_solid_rocket_motor_network.fit(X_train, y_train)

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time_pinn = end_time - start_time
    
    # Obtain errors
    _ , _, _, error_average_pinn, error_max_pinn, error_peak_pinn, execution_time_pinn = prep_data_surrogate(data_dir_val, scaled=True, n_outputs=2, n_samples=1e12, start_time=None, end_time=None).compare_with_data(PINN_solid_rocket_motor_network.predict)
    
    error_average_pressure_pinn = error_average_pinn[0]
    error_max_pressure_pinn = error_max_pinn[0]
    error_peak_pressure_pinn = error_peak_pinn[0]
    
    error_average_thrust_pinn = error_average_pinn[1]
    error_max_thrust_pinn = error_max_pinn[1]
    error_peak_thrust_pinn = error_peak_pinn[1]
    
    # I-PINN
    # Create network (re-init weights and biases)
    IPINN_solid_rocket_motor_network = NN(20, 2, 64, 12, epochs=5000, batch_size=64, use_batch=False, lr=1e-3, loss_terms=loss_functions, learn_params=learn_parameters, unique_input=True, activation_fn=nn.ELU, init_method=nn.init.kaiming_uniform_, show_plots=False)
    
    # Start time
    start_time = time.time()

    # Train network
    IPINN_solid_rocket_motor_network.fit(X_train, y_train)

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time_ipinn = end_time - start_time
    
    # Obtain errors
    _ , _, _, error_average_ipinn, error_max_ipinn, error_peak_ipinn, execution_time_ipinn = prep_data_surrogate(data_dir_val, scaled=True, n_outputs=2, n_samples=1e12, start_time=None, end_time=None).compare_with_data(IPINN_solid_rocket_motor_network.predict)
    
    error_average_pressure_ipinn = error_average_ipinn[0]
    error_max_pressure_ipinn = error_max_ipinn[0]
    error_peak_pressure_ipinn = error_peak_ipinn[0]
    
    error_average_thrust_ipinn = error_average_ipinn[1]
    error_max_thrust_ipinn = error_max_ipinn[1]
    error_peak_thrust_ipinn = error_peak_ipinn[1]
    
    # Obtain h for I-Numerical
    I_a = IPINN_solid_rocket_motor_network.a.data.numpy()[0]
    I_n = IPINN_solid_rocket_motor_network.n.data.numpy()[0]
    
    # Data driven
    # Create network (re-init weights and biases)
    NN_solid_rocket_motor_network = NN(20, 2, 64, 12, epochs=5000, lr=1e-3, loss_terms=None, learn_params=None, unique_input=False, activation_fn=nn.ELU, init_method=nn.init.kaiming_uniform_, show_plots=False)
    
    # Start time
    start_time = time.time()

    # Train network
    NN_solid_rocket_motor_network.fit(X_train, y_train)

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time_data_driven = end_time - start_time
    
    # Obtain errors
    _ , _, _, error_average_data_driven, error_max_data_driven, error_peak_data_driven, execution_time_data_driven = prep_data_surrogate(data_dir_val, scaled=True, n_outputs=2, n_samples=1e12, start_time=None, end_time=None).compare_with_data(NN_solid_rocket_motor_network.predict)
    
    error_average_pressure_data_driven = error_average_data_driven[0]
    error_max_pressure_data_driven = error_max_data_driven[0]
    error_peak_pressure_data_driven = error_peak_data_driven[0]
    
    error_average_thrust_data_driven = error_average_data_driven[1]
    error_max_thrust_data_driven = error_max_data_driven[1]
    error_peak_thrust_data_driven = error_peak_data_driven[1]
    
    # Numerical method
    X_test , Experimental_data, _, _, _, _, _ = prep_data_surrogate(data_dir_val, scaled=False, n_outputs=2, n_samples=1e12, start_time=None, end_time=None).compare_with_data()
    
    inputs = X_test[0][1:]

    # Grain 
    L_grain = inputs[0]
    d_grain_in = inputs[1]
    t_liner = inputs[3]
    d_grain_out = inputs[2]
    r_grain_init = d_grain_in / 2 # Port radius [m]
    mass_grain_init = inputs[4]
    rho_grain =  mass_grain_init / ( L_grain * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2))# Density of grain [kg/m3]
    N_grains = inputs[5]
    a = inputs[17]
    n = inputs[18]


    # Inhibiters
    uninhibited_core = inputs[6]
    uninhibited_sides = inputs[7]

    # Combustion mixture (predicted using CEA)
    gamma = inputs[9]
    W_g = inputs[10]
    T_0 = inputs[11]
    R_g = R_a / W_g 



    # Motor specifications
    d_chamber = inputs[12]
    L_chamber = inputs[13]
    d_t = inputs[14]
    d_e = inputs[15]
    r_t = d_t/2 # Nozzle throat radius [m]
    r_e = d_e/2 # Nozzle exit radius [m]
    A_t = np.pi * r_t**2  # Nozzle throat area [m]
    A_e = np.pi * r_e**2 # Nozzle exit area [m]
    

    # Set time of numerical simulation
    t_span =  (0, X_test[:,0][-1])

    # Create a dense set of time points for output
    t_eval = np.linspace(t_span[0], t_span[1], len(X_test[:,0]))  # Timesteps in simulation
    
    # Start time
    start_time = time.time()

    # Obtain_results
    sol = solve_ivp(solid, t_span, [P_a, r_grain_init], method='RK45', dense_output=True, t_eval=t_eval, args=(L_grain, d_grain_in, t_liner, d_grain_out, r_grain_init, mass_grain_init, rho_grain, N_grains, a, n, uninhibited_core, uninhibited_sides, gamma, W_g, T_0, R_g, d_chamber, L_chamber, d_t, d_e, r_t, r_e, A_e, A_t))

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    execution_time_num = end_time - start_time
    
    # Obtain pressure from simulation
    P_0_sol = sol.y[0]
    
    # Obtain thrust
    pepc = fsolve(AeAt_func, 0.001, args=(Gamma(gamma), gamma, A_e, A_t))
    P_e_sol = P_0_sol*1e5 * pepc
    C_f = Cf(Gamma(gamma), gamma, P_e_sol, P_0_sol*1e5, P_a*1e5, (A_e / A_t))
    Thrust1 = C_f * P_0_sol*1e5 * A_t 
    
    # Obtain errors
    error_avg_pressure_num = np.sqrt(np.mean((Experimental_data[:, 0] - P_0_sol)**2))
    error_max_pressure_num = np.sqrt(np.max((Experimental_data[:, 0] - P_0_sol)**2))
    error_peak_pressure_num = np.sqrt((np.max(Experimental_data[:, 0]) - np.max(P_0_sol))**2) 

    error_avg_thrust_num = np.sqrt(np.mean((Experimental_data[:, 1] - Thrust1)**2))
    error_max_thrust_num = np.sqrt(np.max((Experimental_data[:, 1] - Thrust1)**2))
    error_peak_thrust_num = np.sqrt((np.max(Experimental_data[:, 1]) - np.max(Thrust1))**2)
    
    
    
    # I-Numerical method
    # Start time
    start_time = time.time()

    # Obtain_results
    sol = solve_ivp(solid, t_span, [P_a, r_grain_init], method='RK45', dense_output=True, t_eval=t_eval, args=(L_grain, d_grain_in, t_liner, d_grain_out, r_grain_init, mass_grain_init, rho_grain, N_grains, I_a, I_n, uninhibited_core, uninhibited_sides, gamma, W_g, T_0, R_g, d_chamber, L_chamber, d_t, d_e, r_t, r_e, A_e, A_t))

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    execution_time_inum = end_time - start_time
    
    # Obtain pressure from simulation
    P_0_sol = sol.y[0]
    
    # Obtain thrust
    pepc = fsolve(AeAt_func, 0.001, args=(Gamma(gamma), gamma, A_e, A_t))
    P_e_sol = P_0_sol*1e5 * pepc
    C_f = Cf(Gamma(gamma), gamma, P_e_sol, P_0_sol*1e5, P_a*1e5, (A_e / A_t))
    Thrust1 = C_f * P_0_sol*1e5 * A_t 
    
    # Obtain errors
    error_avg_pressure_inum = np.sqrt(np.mean((Experimental_data[:, 0] - P_0_sol)**2))
    error_max_pressure_inum = np.sqrt(np.max((Experimental_data[:, 0] - P_0_sol)**2))
    error_peak_pressure_inum = np.sqrt((np.max(Experimental_data[:, 0]) - np.max(P_0_sol))**2) 

    error_avg_thrust_inum = np.sqrt(np.mean((Experimental_data[:, 1] - Thrust1)**2))
    error_max_thrust_inum = np.sqrt(np.max((Experimental_data[:, 1] - Thrust1)**2))
    error_peak_thrust_inum = np.sqrt((np.max(Experimental_data[:, 1]) - np.max(Thrust1))**2)
    
    
    # Store
    test_file_lst.append(test_file)
    average_error_pressure_lst.append([error_average_pressure_pinn, error_average_pressure_ipinn, error_average_pressure_data_driven, error_avg_pressure_num, error_avg_pressure_inum])
    max_error_pressure_lst.append([error_max_pressure_pinn, error_max_pressure_ipinn, error_max_pressure_data_driven, error_max_pressure_num, error_max_pressure_inum])
    peak_error_pressure_lst.append([error_peak_pressure_pinn, error_peak_pressure_ipinn, error_peak_pressure_data_driven, error_peak_pressure_num, error_peak_pressure_inum])
    average_error_thrust_lst.append([error_average_thrust_pinn, error_average_thrust_ipinn, error_average_thrust_data_driven, error_avg_thrust_num, error_avg_thrust_inum])
    max_error_thrust_lst.append([error_max_thrust_pinn, error_max_thrust_ipinn, error_max_thrust_data_driven, error_max_thrust_num, error_max_thrust_inum])
    peak_error_thrust_lst.append([error_peak_thrust_pinn, error_peak_thrust_ipinn, error_peak_thrust_data_driven, error_peak_thrust_num, error_peak_thrust_inum])
    train_time_lst.append([elapsed_time_pinn, elapsed_time_ipinn, elapsed_time_data_driven])
    execution_time_lst.append([execution_time_pinn, execution_time_ipinn, execution_time_data_driven, execution_time_num, execution_time_inum])


# Total average RMSE error
Total_average_pressure_error = np.mean(np.array(average_error_pressure_lst), axis=0)
Total_average_thrust_error = np.mean(np.array(average_error_thrust_lst), axis=0)


# Total average max error
Total_max_pressure_error = np.mean(np.array(max_error_pressure_lst), axis=0)
Total_max_thrust_error = np.mean(np.array(max_error_thrust_lst), axis=0)

# Total peak error
Total_peak_pressure_error = np.mean(np.array(peak_error_pressure_lst), axis=0)
Total_peak_thrust_error = np.mean(np.array(peak_error_thrust_lst), axis=0)

# Total average train time
Total_train_time = np.mean(np.array(train_time_lst), axis=0)

# Total average execution time
Total_execution_time = np.mean(np.array(execution_time_lst), axis=0)


# Print errors
formatted_errors_avg_pressure = " ".join(f"{error:.6f}" for error in Total_average_pressure_error)
print(f"Total average error pressure [Bar]: {formatted_errors_avg_pressure}")

formatted_errors_avg_thrust = " ".join(f"{error:.6f}" for error in Total_average_thrust_error)
print(f"Total average error thrust [N]: {formatted_errors_avg_thrust}")

formatted_errors_max_pressure = " ".join(f"{error:.6f}" for error in Total_max_pressure_error)
print(f"Total max error pressure [Bar]: {formatted_errors_max_pressure}")

formatted_errors_max_thrust = " ".join(f"{error:.6f}" for error in Total_max_thrust_error)
print(f"Total max error thrust [N]: {formatted_errors_max_thrust}")

formatted_errors_peak_pressure = " ".join(f"{error:.6f}" for error in Total_peak_pressure_error)
print(f"Total peak error pressure [Bar]: {formatted_errors_peak_pressure}")

formatted_errors_peak_thrust = " ".join(f"{error:.6f}" for error in Total_peak_thrust_error)
print(f"Total peak error thrust [N]: {formatted_errors_peak_thrust}")

formatted_errors_train = " ".join(f"{error:.6f}" for error in Total_train_time)
print(f"Total average train time [s]: {formatted_errors_train}")

formatted_errors_execution = " ".join(f"{error:.6f}" for error in Total_execution_time)
print(f"Total average execution time [s]: {formatted_errors_execution}")



save_data = int(input("Store data? [1 = yes, 0 = no]"))

if save_data == 1:
    # Save to CSV
    df = pd.DataFrame({
        "Test file": test_file_lst,
        "Average error pressure": average_error_pressure_lst,
        "Maximum error pressure": max_error_pressure_lst,
        "Peak error pressure": peak_error_pressure_lst,
        "Average error thrust": average_error_thrust_lst,
        "Maximum error thrust": max_error_thrust_lst,
        "Peak error thrust": peak_error_thrust_lst,
        "Train time": train_time_lst,
        "Execution time": execution_time_lst
    })
    
    output_csv_path = "/Users/tristanhirs/Downloads/Thesis/Motor test data/Error_file/LOO_CV_100_points_SRM.csv"  # Specify the path where you want to save the CSV
    df.to_csv(output_csv_path, index=False)

