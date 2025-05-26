# Imported modules
#from NN_tools_thesis_sinlge_input import NN, gradient
from NN_tools_thesis import NN, gradient
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import itertools
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.signal import savgol_filter
from Loss_function_tools import SRM_helper_tools

# Setup constants

# Constants
g_0 = 9.80665 # Gravitational acceleration
R_a = 8.3144 # Gas constant
P_a = 101325/1e5 # Standard pressure at sea level

# Grain 
L_grain = 105e-3 # Length of grain [m]
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
uninhibited_core = 1
uninhibited_sides = 0

# Combustion mixture (predicted using CEA)
gamma =  1.137 # Specific heat of combustion mixture[-]
W_g = 39.9e-3  # Molecular mass of mixture
R_g = R_a / W_g 
T_0 = 1600 # Temperature of combustion mixture


# Motor specifications
d_chamber = 80e-3 # Outside diamter of the chamber [m]
L_chamber = 118.5e-3 # Length of the chamber [m]
r_t = 8.37e-3/2 # Nozzle throat radius [m]
#r_t = 7e-3/2
r_e = 16.73e-3/2 # Nozzle exit radius [m]
#r_e = 14e-3/2
A_t = np.pi * r_t**2  # Nozzle throat area [m]
A_e = np.pi * r_e**2 # Nozzle exit area [m]
t_b = 10 # Burn time
alpha = 0.1

V_init =  np.pi * r_grain_init**2 * L_grain + (np.pi * (d_chamber**2/4) * L_chamber) - (np.pi * ((d_grain_out + 2*t_liner)**2 / 4) * L_grain)



# Initialize scalers for input and output
'''
input_scaler = MinMaxScaler()  # For scaling inputs
output_scaler = MinMaxScaler()  # For scaling outputs 
'''
# Use standard scaler to prevent zeros in input data
input_scaler = StandardScaler() 
output_scaler = StandardScaler()


# Import training data
# Function to load CSV files
def load_csv_files(data_dir):
    """Loads all CSV files from the given directory."""
    all_data = []
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(data_dir, file_name)
            data = pd.read_csv(file_path)
            all_data.append(data)
    
    return all_data

# Interpolation for unequal time data
def interpolate_data(df, target_time):
    """Interpolates data to match a target time sequence."""
    interpolated_df = pd.DataFrame()
    interpolated_df['Time'] = target_time
    interpolated_df['Pressure'] = np.interp(target_time, df['time'], df['pressure'])
    return interpolated_df

# Function to prepare data for PINN (with 14 constant inputs and 1 time input)
def prepare_data(data_list, constant_inputs, target_time, test_size=0.2):
    """Prepares data by interpolating time and adding constant inputs. Splits into train/val sets."""
    
    # List to store all training data
    all_inputs = []
    all_outputs = []

    for data in data_list:
        # Interpolate to match the target time
        interpolated_data = interpolate_data(data, target_time)

        # Extract time and pressure
        time_data = interpolated_data['Time'].values
        pressure_data = interpolated_data['Pressure'].values

        # Create input features: 14 constant inputs + 1 time input
        inputs = np.hstack([np.repeat(constant_inputs[np.newaxis, :], len(time_data), axis=0), time_data[:, np.newaxis]])
        
        # Store inputs and pressure outputs
        all_inputs.append(inputs)
        all_outputs.append(np.abs(pressure_data))
    
    # Convert lists to numpy arrays
    all_inputs = np.vstack(all_inputs)  # Stack all inputs together
    all_inputs = all_inputs[:][:,-1]
    all_outputs = np.hstack(all_outputs)  # Stack all outputs together
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(all_inputs, all_outputs, test_size=test_size, random_state=42)
    

    # Fit and transform the input data (X_train and X_val)
    X_train_log = np.log1p(X_train)
    X_val_log = np.log1p(X_val)
    
    X_train_scaled = X_train_log
    X_val_scaled = X_val_log

    # Fit and transform the output data (y_train and y_val)
    
    y_train_scaled = output_scaler.fit_transform(y_train.reshape(-1, 1))
    
    y_train_log = np.log1p(y_train)
    y_train_scaled = y_train_log

    y_val_scaled = output_scaler.transform(y_val.reshape(-1, 1))

    
    return X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled


   

# Example usage
if __name__ == "__main__":
    data_dir = '/Users/tristanhirs/Downloads/Thesis/Motor test data/Training_data_single/Only time/'  # Path to your CSV files
    constant_inputs = np.array([L_grain, d_grain_in/2, d_grain_out/2, mass_grain_init, rho_grain, a, n, gamma, W_g, R_g, T_0, A_t, A_e])  # 14 constant inputs
    
    # Target time sequence (shared time range for interpolation)
    target_time = np.linspace(0, 6, 20)  # Adjust to your target time range
    
    # Load the CSV files
    all_data = load_csv_files(data_dir)
    
    # Prepare data (with a 20% validation split)
    X_train, X_val, y_train, y_val, X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled = prepare_data(all_data, constant_inputs, target_time, test_size=0.2)
    
    # Print the shapes of the datasets
    print("Training data shape (X_train):", X_train.shape)
    print("Validation data shape (X_val):", X_val.shape)
    print("Training labels shape (y_train):", y_train.shape)
    print("Validation labels shape (y_val):", y_val.shape)

'''
plt.plot(X_train[:,-1], y_train, 'o')
plt.plot(X_val[:,-1], y_val,'o')
plt.show()
'''

# Use same weight and bias intialization every time for Neural Network
torch.manual_seed(42)



# PINN
test_lst = []


# New PINN loss with correct implemtnation of formulas
def PINN_loss_new(NN_output: torch.nn.Module):
    # Collocation points time array
    steps = 200
    #time_shift = NN_output.t_shift.item()
    #time_shift = 0
    time_inputs = torch.linspace(1e-16, 8, steps).view(-1, 1).requires_grad_(True)    
    
    
    # Obtain predictions from the network (Pc, a, n)
    P_C = NN_output(time_inputs) 
    #a = NN_output.a
    #n = NN_output.n
    #A_b = NN_output.A_b
    #V_c = NN_output.V_c
    
    # Obtain derivative of Pc with respect to time over the network
    dP_C_dt = gradient(P_C, time_inputs) 
    
    # Obtain r_integrator
    radius = SRM_helper_tools.r_integrator(P_C, a, n, time_inputs, d_grain_in/2)
    
    # Check when motor is burning or not
    burning_logic = torch.logical_and(radius <= d_grain_out/2, (radius - d_grain_in/2) <= L_grain/2)
    
    # Obtain dPdt from formula
    dPdt_theory = SRM_helper_tools.dPdt_check(radius, P_C, P_a, burning_logic, a, n, rho_grain, R_g, T_0, A_t, gamma, uninhibited_core, uninhibited_sides, L_grain, r_grain_init, d_grain_out, N_grains, L_chamber, d_chamber, t_liner)
    

    '''
    # Obtain predictions from the network (Pc, a, n)
    P_C = NN_output(time_inputs)[:,0].unsqueeze(-1)
    #a = NN_output.a
    #n = NN_output.n

    
    # Obtain derivative of Pc with respect to time over the network
    dP_C_dt = gradient(P_C, time_inputs)[:,0].unsqueeze(-1)
    
    # Obtain r_integrator
    radius = SRM_helper_tools.r_integrator(P_C, a, n, time_inputs, d_grain_in/2)
    
    # Check when motor is burning or not
    burning_logic = torch.logical_and(radius <= (d_grain_out/2), (radius - (d_grain_in/2)) <= L_grain/2)

    # Obtain dPdt from formula
    dPdt_theory = SRM_helper_tools.dPdt_check(radius, P_C, P_a, burning_logic, a, n, rho_grain, R_g, T_0, A_t, gamma, uninhibited_core, uninhibited_sides, L_grain, d_grain_in/2, d_grain_out, N_grains, L_chamber, d_chamber, t_liner, A_b=0)
    #print(torch.abs(dPdt_theory - dP_C_dt))
    '''
    
    #print(dPdt_theory)
    
    return torch.mean((dPdt_theory - dP_C_dt)**2)



# New PINN loss with correct implemtnation of formulas
def PINN_pressure_loss(NN_output: torch.nn.Module, used_inputs):
    # Collocation points time array
    start, end = 1e-16, 8
    steps = 200
    time_inputs = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    log_space1 = torch.logspace(start=torch.log10(torch.tensor(1e-16)),end=torch.log10(torch.tensor(1)),steps=100).view(-1,1).requires_grad_(True)
    #log_space2 = torch.logspace(start=torch.log10(torch.tensor(end)),end=torch.log10(torch.tensor(start)),steps=steps).view(-1,1).requires_grad_(True)
    
    # Normalize to your range [start, end] by inverting
    #inverse_log_space = end - (end - start) * (log_space2 - log_space2.min()) / (log_space2.max() - log_space2.min())

    # Concatenate with reversed version to get more density at both start and end
    time_inputs = torch.cat((log_space1, time_inputs), dim=0)

    
    
    # Setup total loss
    total_loss = 0
    
    # test
    used_inputs_2 = np.array([[105e-3, 25e-3, 80e-3, 1.6e-3, 758e-3, 1, 1, 0, 1.137, 39.9e-3, 1600, 80e-3, 118.5e-3, 8.37e-3, 16.73e-3, 5.13, 0.222]])  # 14 constant inputs

    for scaled_inputs in used_inputs_2:
        #print(scaled_inputs)
        inputs = scaled_inputs
        #time_shift = NN_output.t_shift
        time_shift = 0
        
        
        # Unzip constant values from inputs
        L_grain = inputs[0]
        d_grain_in = inputs[1]
        d_grain_out = inputs[2]
        t_liner = inputs[3]
        mass_grain = inputs[4]
        N_grains = inputs[5]
        uninhibited_core = inputs[6]
        uninhibited_sides = inputs[7]
        gamma = inputs[8]
        W_g = inputs[9]
        T_0 = inputs[10]
        d_chamber = inputs[11]
        L_chamber = inputs[12]
        d_t = inputs[13]
        #d_e = inputs[14]
        a = inputs[15]
        n = inputs[16]
        

        # Perform calulcations for standard values
        R_g = R_a / W_g 
        A_t = np.pi * d_t**2/4  # Nozzle throat area [m]
        rho_grain =  mass_grain / ( L_grain * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2))# Density of grain [kg/m3]
        
        
        # Find best time shift
        time_inputs = time_inputs + time_shift

        
        # Obtain predictions from the network (Pc, a, n)
        P_C = NN_output(time_inputs)[:,0].unsqueeze(-1)
        #a = NN_output.a
        #n = NN_output.n

        
        # Obtain derivative of Pc with respect to time over the network
        dP_C_dt = gradient(P_C, time_inputs)[:,0].unsqueeze(-1)
        
        # Obtain r_integrator
        radius = SRM_helper_tools.r_integrator(P_C, a, n, time_inputs, d_grain_in/2)
        
        # Check when motor is burning or not
        burning_logic = torch.logical_and(radius <= (d_grain_out/2), (radius - (d_grain_in/2)) <= L_grain/2)

        # Obtain dPdt from formula
        dPdt_theory = SRM_helper_tools.dPdt_check(radius, P_C, P_a, burning_logic, a, n, rho_grain, R_g, T_0, A_t, gamma, uninhibited_core, uninhibited_sides, L_grain, d_grain_in/2, d_grain_out, N_grains, L_chamber, d_chamber, t_liner, A_b=0)
        #print(torch.abs(dPdt_theory - dP_C_dt))
        
        
        # Determine loss for specific case and add to total loss
        total_loss += torch.mean((dPdt_theory - dP_C_dt)**2)
        #print(total_loss)

    return total_loss / len(used_inputs_2)



'''
# PINN loss with ignition time delay
def PINN_loss_angle(NN_output: torch.nn.Module):
    # Collocation points time array
    steps = 100
    time_inputs = torch.linspace(1e-16, 6, steps).view(-1, 1).requires_grad_(True)    
    
    # Obtain predictions from the network (Pc, a, n)
    P_C = NN_output(time_inputs) 
    #a = NN_output.a
    #n = NN_output.n
    alpha = NN_output.alpha
    alpha = alpha * torch.pi / 180
    
    # Obtain derivative of Pc with respect to time over the network
    dP_C_dt = gradient(P_C, time_inputs) 
    
    # Obtain r_integrator
    radius = r_integrator(P_C, a, n, time_inputs, 0)
    
    # Obtain A_b regions
    A_b = A_b_calc(radius, L_grain, r_grain_init, d_grain_out, alpha)
    
    # Obtain V_c_integrator
    V_c = V_c_integrator(P_C, a, n, A_b, time_inputs, V_init)
    
    # Obtain dPdt from formula
    dPdt_theory = dPdt_new(radius, P_C, a, n, A_b, V_c)
    
    return torch.mean((dPdt_theory - dP_C_dt)**2)


# PINN los with learnable paramaters -> sort of cheating and hard to justify
def PINN_loss_learn_param(NN_output: torch.nn.Module):
    # Collocation points time array
    steps = 100
    time_inputs = torch.linspace(1e-16, 8, steps).view(-1, 1).requires_grad_(True)    
    
    # Obtain predictions from the network (Pc, a, n)
    P_C = NN_output(time_inputs) 
    a = NN_output.a
    n = NN_output.n
    A_b = NN_output.A_b
    V_c = NN_output.V_c
    
    # Obtain derivative of Pc with respect to time over the network
    dP_C_dt = gradient(P_C, time_inputs) 
    
    # Obtain dPdt from formula
    dPdt_theory = dPdt_learn_param(P_C, a, n, A_b, V_c)
    
    return torch.mean((dPdt_theory - dP_C_dt)**2)
'''



# Define boundary loss function (P_a at start)
def Boundary_loss(NN_output: torch.nn.Module):
    t_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)
    P_C = NN_output(t_boundary)
    
    return torch.mean((torch.squeeze(P_C) - P_a)**2)





# Define boundary loss function (P_a at start)
def Boundary_loss_new(NN_output: torch.nn.Module):
    t_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)
    
    # Setup total loss
    total_loss = 0
    
    used_inputs_2 = np.array([[105e-3, 25e-3, 80e-3, 1.6e-3, 758e-3, 1, 1, 0, 1.137, 39.9e-3, 1600, 80e-3, 118.5e-3, 8.37e-3, 16.73e-3, 5.13, 0.222]])  # 14 constant inputs
    
    for scaled_inputs in used_inputs_2:
        
    
        # Obtain predictions from the network (Pc, F, a, n)
        outputs_of_network = NN_output(t_boundary)
        P_C = outputs_of_network[:,0].unsqueeze(-1)
        #F_network = outputs_of_network[:,1].unsqueeze(-1)
        
        total_loss += (torch.squeeze(P_C) - P_a)**2 #+ (torch.squeeze(F_network) - 0)**2
        
    return total_loss / len(used_inputs_2)




# Positive loss function
def Positive_loss(NN_output: torch.nn.Module):
    steps = 20
    time_inputs = torch.linspace(0, 10, steps).view(-1, 1).requires_grad_(True)

    P_C = NN_output(time_inputs)

    return torch.mean(torch.clamp(-P_C, min=0)**2)


# Create network
#PINN_solid_rocket_motor_network = NN(1, 1, 128, 15, epochs=600, loss2=PINN_loss_new, loss2_weight=1, lossBC=Boundary_loss, lossBC_weight=1, lossPositive=Positive_loss, lossPositive_weight=1, lr=1e-3)


# Initialize loss functions
loss_functions = {
    #PINN_loss_new: 1,
    #PINN_thrust_loss: 1e-4,
    PINN_pressure_loss: 1,
    #Boundary_loss: 1,
    #Boundary_loss_new: 1,
    #Positive_loss: 1
    }

# Create network
PINN_solid_rocket_motor_network = NN(1, 1, 128, 15, epochs=600, batch_size=8, use_batch=False, lr=1e-3, loss_terms=loss_functions, learn_params=None)

train_loss, val_loss = PINN_solid_rocket_motor_network.fit(X_train, y_train, X_val, y_val) # Remove validation for now

# Plot losses
plt.figure()
plt.plot(train_loss)
plt.plot(val_loss)
plt.yscale('log')


time_plot = np.arange(0, 8, 1e-4)

'''
# Create a list or array of constants
constant_inputs = np.array([L_grain, r_grain_in, r_grain_out, m_grain, rho_grain, a, n, gamma, W_g, R_g, T_0, A_t, A_e])

# Repeat the constants for each time step
constant_features_repeated = np.tile(constant_inputs, (len(time_plot), 1))  # Shape will be (number of time points, 13)

# Reshape the time feature to make it compatible for concatenation
time_feature = time_plot.reshape(-1, 1)  # Shape (number of time points, 1)

# Concatenate the constant features with the time feature along the second axis (columns)
input_data = np.hstack([constant_features_repeated, time_feature])


# Scaled input data
input_plot_scaled = np.log1p(input_data)
#input_plot_scaled = input_scaler.fit_transform(input_data)


# Predict outputs using the trained model
predictions_scaled = PINN_solid_rocket_motor_network.predict(input_plot_scaled)

# Convert the scaled predictions back to the original scale
#predictions = output_scaler.inverse_transform(predictions_scaled)
predictions = np.expm1(predictions_scaled)
'''
# Plot predictions and print final values for net paramater discovery values
predictions = PINN_solid_rocket_motor_network.predict(time_plot)
#a_data = PINN_solid_rocket_motor_network.a.data.numpy()[0]
#n_data = PINN_solid_rocket_motor_network.n.data.numpy()[0]
#print(f'Estimation of parameter a and n for heat equation: {a} and {n}')

# Plot the predicted vs actual values
plt.figure(figsize=(10, 6))
plt.plot()
plt.plot(time_plot, predictions,'o', label="Predicted Pressure")
plt.plot(np.expm1(X_train_scaled), y_train, 'o', label="Train Data")
plt.plot(np.expm1(X_val_scaled), y_val, 'o', label="Validation Data")
plt.xlabel('Time [s]')
plt.ylabel('Chamber Pressure [Bar]')
plt.legend()
plt.title('Predicted vs Actual Chamber Pressure')
plt.show()





