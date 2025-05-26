# Imported modules
from NN_tools_thesis import NN, gradient
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
from Loss_function_tools import SRM_helper_tools
from Input_generator_SRM import SRM_input
from Validation_class import prep_data, prep_data_surrogate
import time
import os
import pandas as pd
import matplotlib.animation as animation
import torch.optim as optim

# Constants
g_0 = 9.80665 # Gravitational acceleration
R_a = 8.3144 # Gas constant
P_a = 101325/1e5 # Standard pressure at sea level

# Import training data
# Multiple file usage
data_dir = '/Users/tristanhirs/Downloads/Thesis/Motor test data/Training data/'  # Path to CSV file(s)
data_dir_single = '/Users/tristanhirs/Downloads/Thesis/Motor test data/Training_data_single/Only time/'
data_dir_single_data = '/Users/tristanhirs/Downloads/Thesis/Motor test data/Training_data_single/All/'
data_dir_trained_PINN = '/Users/tristanhirs/Downloads/Thesis/Motor test data/PINN_trained_models/'
data_dir_val = '/Users/tristanhirs/Downloads/Thesis/Motor test data/Validation_data/'
data_dir_sim_data = '/Users/tristanhirs/Downloads/Thesis/Motor test data/Simulated_data/reference_test_2_sim_output.csv'
data_dir_videos = "/Users/tristanhirs/Downloads/Thesis/Training_process_videos/"


# Unpack train and validation data
X_train, y_train, X_val, y_val = prep_data(data_dir_single, scaled=False, n_outputs=2, n_samples=10, start_time=None, end_time=None).train_val_split()
test_inputs , _, _, _, _ = prep_data_surrogate(data_dir_single_data, scaled=False, n_outputs=2, n_samples=5, start_time=None, end_time=None).train_val_split()

inputs = torch.tensor(np.unique(test_inputs[:, 1:], axis=0)[0], dtype=torch.float32)

# Plot training data for testing
'''
plt.plot(X_train[:,0], y_train[:, 0], 'o')
plt.plot(X_val[:,0], y_val[:, 0],'o')
plt.plot(X_train[:,0], y_train[:, 1], 'o')
plt.plot(X_val[:,0], y_val[:, 1],'o')
plt.show()

'''

# Setup of figure to show progress during training
sim_data = pd.read_csv(data_dir_sim_data)

# Define thresholds
pressure_lower = 0.8  # Set your desired lower threshold
pressure_upper = 55  # Set your desired upper threshold

# Apply threshold filtering (keep rows where all values are within range)
sim_data = sim_data[(sim_data["pressure"] >= pressure_lower) & (sim_data["pressure"] <= pressure_upper)]


sim_time = sim_data["time"].to_numpy()
sim_pressure = sim_data["pressure"].to_numpy()
sim_thrust = sim_data['thrust'].to_numpy()

executed = False
fig, axs = None, None



# PINN


# New PINN loss with correct implemtnation of formulas
def PINN_pressure_loss(NN_output: torch.nn.Module, used_inputs):
    # Collocation points time array
    start, end = 1e-16, 6
    steps = 300
    time_inputs = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    #log_space1 = torch.logspace(start=torch.log10(torch.tensor(start)),end=torch.log10(torch.tensor(0.5)),steps=200).view(-1,1).requires_grad_(True)
    
    # end space
    
    #time_end = torch.linspace(0, 1, 200)  # Uniform grid in [0,1]
    #time_end = time_end ** 0.3  # Lower power -> stronger clustering at the end

    #Rescale to [5,6]
    #time_end = time_end * (6 - 3) + 3
    
    # Concatenate with reversed version to get more density at both start and end
    #time_inputs = torch.cat((log_space1, time_inputs, time_end), dim=0)
    
    #time_shift = NN_output.t_shift
    #time_shift = 0.0
    
    
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
    #d_e = inputs[15]
    a = inputs[17]
    n = inputs[18]

    # Perform calulcations for standard values
    R_g = R_a / W_g 
    A_t = np.pi * d_t**2/4  # Nozzle throat area [m]
    rho_grain =  mass_grain / ( L_grain * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2))# Density of grain [kg/m3]
    
    
    # Find best time shift
    #time_inputs = time_inputs + time_shift


    # Obtain predictions from the network (Pc, a, n)
    P_C = NN_output(time_inputs)[:, 0].unsqueeze(-1)
    #print(P_C.max())
    #a = NN_output.a
    #n = NN_output.n

    # Obtain derivative of Pc with respect to the scaled time over the network and convert to dPc/dt
    dP_C_dt = gradient(P_C, time_inputs)
    #print(dP_C_dt.max())

    
    # Obtain r_integrator
    radius = SRM_helper_tools.r_integrator(P_C, a, n, time_inputs, d_grain_in/2)
    
    # Check when motor is burning or not
    burning_logic = torch.logical_and(radius <= (d_grain_out/2), (radius - (d_grain_in/2)) <= L_grain/2)
    
    # Set radius to fixed values if all propellant is burned
    radius[radius >= (d_grain_out/2)] = d_grain_out/2
    radius[(radius - (d_grain_in/2)) >= L_grain/2] = L_grain/2

    # Obtain dPdt from formula
    dPdt_theory = SRM_helper_tools.dPdt_check(radius, P_C, P_a, burning_logic, a, n, rho_grain, R_g, T_0, A_t, gamma, uninhibited_core, uninhibited_sides, L_grain, d_grain_in/2, d_grain_out, N_grains, L_chamber, d_chamber, t_liner)
    #print(dPdt_theory.max())
    #print(torch.abs(dPdt_theory - dP_C_dt))
    
    #print(((dPdt_theory - dP_C_dt)**2).mean())

    return ((dPdt_theory - dP_C_dt)**2).mean()



def PINN_thrust_loss(NN_output: torch.nn.Module, used_inputs):
    # Collocation points time array
    start, end = 1e-16, 6
    steps = 300
    time_inputs = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    #log_space1 = torch.logspace(start=torch.log10(torch.tensor(1e-16)),end=torch.log10(torch.tensor(1)),steps=100).view(-1,1).requires_grad_(True)
    

    # Concatenate with reversed version to get more density at both start and end
    #time_inputs = torch.cat((log_space1, time_inputs), dim=0)
    
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
    

    # Obtain predictions from the network (Pc, F, a, n)
    outputs_of_network = NN_output(time_inputs)
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
    
    return torch.mean((F_theory - F_network)**2)



# Define pressure boundary loss function (P_a at start)
def Boundary_loss_pressure(NN_output: torch.nn.Module, used_inputs):
    t_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)
    

    # Obtain predictions from the network (Pc, F, a, n)
    P_C = NN_output(t_boundary)[:, 0].unsqueeze(-1)
    
    #print(((torch.squeeze(P_C) - P_a)**2).mean())
       
    return ((torch.squeeze(P_C) - P_a)**2).mean()


# Define force boundary loss function (0 at start)
def Boundary_loss_force(NN_output: torch.nn.Module, used_inputs):
    t_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)
    

    # Obtain predictions from the network (Pc, F, a, n)
    F_network = NN_output(t_boundary)[:,1].unsqueeze(-1)
        
    return ((torch.squeeze(F_network) - 0)**2).mean()



# Positive loss function
def Positive_loss_pressure(NN_output: torch.nn.Module, used_inputs):
    steps = 50
    time_inputs = torch.linspace(0, 6, steps).view(-1, 1).requires_grad_(True)
    
    # Obtain predictions from the network (Pc, F, a, n)
    P_C = NN_output(time_inputs)[:, 0].unsqueeze(-1)
    
    #print((torch.clamp(-P_C, min=0)**2).mean())
        
    return (torch.clamp(-P_C, min=0)**2).mean()


# Positive loss function
def Positive_loss_force(NN_output: torch.nn.Module, used_inputs):
    steps = 50
    time_inputs = torch.linspace(0, 6, steps).view(-1, 1).requires_grad_(True)

    # Obtain predictions from the network (Pc, F, a, n)
    F_network = NN_output(time_inputs)[:,1].unsqueeze(-1)
        
    return torch.mean(torch.clamp(-F_network, min=0)**2)

def show_training_process(NN_output: torch.nn.Module, used_inputs):
    global executed, fig, axs
    
    if not executed:
        plt.ion()  # Turn on interactive mode
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        executed = True
    
    ax1, ax2 = axs
    
    # Collocation points time array
    start, end = 0, 6
    steps = 2000
    time_inputs = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    
    # Obtain predictions from the network (Pc, a, n)
    P_C = NN_output(time_inputs)[:, 0].unsqueeze(-1)
    F_T = NN_output(time_inputs)[:, 1].unsqueeze(-1)
    
    
    # Show predictions over time during training
    ax1.cla()
    ax2.cla()
    
    ax1.plot(X_train, y_train[:, 0], 'o', color='green', label='Train data')
    ax1.plot(sim_time, sim_pressure, color='darkorange', label='Numerical sim')
    ax1.plot(time_inputs.detach().numpy(), P_C.detach().numpy(), '-', label='Pressure PINN')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Pressure [bar]')
    ax1.set_xlim(-0.5, 6)
    ax1.set_ylim(-0.5, 55)
    ax1.grid(which='both', linestyle='--')
    ax1.legend()
    
    ax2.plot(X_train, y_train[:, 1], 'o', color='green', label='Train data')
    ax2.plot(sim_time, sim_thrust, color='darkorange', label='Numerical sim')
    ax2.plot(time_inputs.detach().numpy(), F_T.detach().numpy(), '-', label='Thrust PINN')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Thrust [N]')
    ax2.set_xlim(-0.5, 6)
    ax2.set_ylim(-40, 480)
    ax2.grid(which='both', linestyle='--')
    ax2.legend()
    
    
    plt.pause(0.05)
    
    return 0


# Initialize learnable paramaters
learn_parameters = {
    "t_shift" : 1,
    "a" : 5.13,
    "n" : 0.222
    }

'''
# Initialize loss functions
loss_functions = {
    PINN_pressure_loss: 1,
    Boundary_loss_pressure: 1,
    Positive_loss_pressure: 10,
    PINN_thrust_loss: 10,
    Boundary_loss_force: 1,
    Positive_loss_force: 1,
    #show_training_process: 1e-16
    }
'''

# Initialize loss functions
loss_functions = {
    PINN_pressure_loss: 1e-4,
    Boundary_loss_pressure: 1,
    Positive_loss_pressure: 1,
    PINN_thrust_loss: 1,
    Boundary_loss_force: 1,
    Positive_loss_force: 1,
    show_training_process: 1e-16
    }

# Create network
#PINN_solid_rocket_motor_network = NN(1, 2, 64, 2, epochs=15000, lr=8e-3, loss_terms=loss_functions, learn_params=None, activation_fn=nn.ReLU6, init_method=nn.init.kaiming_normal_, optimizer_type=optim.Adam)
PINN_solid_rocket_motor_network = NN(1, 2, 64, 2, epochs=15000, lr=1e-2, loss_terms=loss_functions, learn_params=None, activation_fn=nn.ELU, init_method=nn.init.kaiming_uniform_)


# Start time
start_time = time.time()

# Train network
PINN_solid_rocket_motor_network.fit(X_train, y_train)

# End time
end_time = time.time()

#a_data = PINN_solid_rocket_motor_network.a.data.numpy()[0]
#n_data = PINN_solid_rocket_motor_network.n.data.numpy()[0]
#print(f'Estimation of parameter a and n for regression equation: {a_data} and {n_data}')
#print(PINN_solid_rocket_motor_network.t_shift.data.numpy()[0])

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:f} seconds")



time_test, true_values, predictions, error_average, error_max, error_peak = prep_data(data_dir_single, scaled=False, n_outputs=2, n_samples=1e6, start_time=None, end_time=None).compare_with_data(PINN_solid_rocket_motor_network.predict)


# Plot the predicted vs actual values
fig2, axs2 = plt.subplots(1,2,figsize=(12, 6))

# Primary axis for Chamber Pressure
axs2[0].plot(time_test[:, 0], true_values[:, 0], 'o', label="Experimental Data Pressure", color='darkorange')
axs2[0].plot(time_test[:, 0], predictions[:, 0], '-', label="Predicted Pressure")
axs2[0].set_xlabel('Time [s]')
axs2[0].set_ylabel('Chamber Pressure [Bar]')
axs2[0].grid(which='both', linestyle='--')
axs2[0].legend()

axs2[1].plot(time_test[:, 0], true_values[:, 0], 'o', label="Experimental Data Pressure", color='darkorange')
axs2[1].plot(time_test[:, 0], predictions[:, 0], '-', label="Predicted Pressure")
axs2[1].set_xlabel('Time [s]')
axs2[1].set_ylabel('Chamber Pressure [Bar]')
axs2[1].grid(which='both', linestyle='--')
axs2[1].legend()

# Title and legend
fig2.suptitle('Predicted vs Actual Chamber Pressure and Thrust')
plt.show()



# Save entire trained model into a certain folder

#filt_data_filename = os.path.join(data_dir_trained_PINN, "SRM_PINN_trained.pth")
#torch.save(PINN_solid_rocket_motor_network.state_dict(), filt_data_filename)




