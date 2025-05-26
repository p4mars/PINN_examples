# Imported modules
from NN_tools_thesis import NN, gradient
import numpy as np
import matplotlib.pyplot as plt
import torch
from Validation_class import prep_data_cooling
import torch.nn as nn
import time
import os


# Constants
g_0 = 9.80665 # Gravitational acceleration
R_a = 8.3144 # Gas constant
P_a = 101325/1e5 # Standard pressure at sea level
h = 5
r = 67.7e-3/2
L = 112e-3
A = 2 * np.pi * r * L
m = 454e-3
c_p = 0.84
T_0 = 70.8
T_env = 6.0
h_test = 2.5e-4
R = ((h * A) / (m * c_p*1e3))

'''
k = h_test
T_env = 27
T_0 = 250
'''

# Import training data
# Multiple file usage
data_dir = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Train_data_time_input/'  # Path to CSV file(s)
data_dir_single = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Training_data_single/'
#data_dir_trained_PINN = '/Users/tristanhirs/Downloads/Thesis/Python code/PINN_trained_models/'

X_train, y_train, X_val, y_val = prep_data_cooling(data_dir, scaled_inputs=False, scaled_outputs=False, n_outputs=1, n_samples=10, start_time=None, end_time=None).train_val_split()

'''
plt.plot(np.expm1(X_train), np.expm1(y_train), 'o')
plt.plot(np.expm1(X_val), np.expm1(y_val),'o')
plt.show()

'''


# Use same weight and bias intialization every time for Neural Network
torch.manual_seed(42)

# PINN

# Define physics loss

def Newtons_cooling_law(NN_output: torch.nn.Module, used_inputs):
    # Setup collocation points over domain
    start, end = 0, 15e3
    steps = 300
    #Collocation_temps = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    
    time_inputs = torch.linspace(start+1000, end, int((steps/10) * 5)).view(-1, 1).requires_grad_(True)
    log_space1 = torch.logspace(start=torch.log10(torch.tensor(1e-16)),end=torch.log10(torch.tensor(start+1000)),steps=int((steps/10) * 5)).view(-1,1).requires_grad_(True)
    
    
    # Concatenate with reversed version to get more density at both start and end
    Collocation_temps = torch.cat((log_space1, time_inputs), dim=0)
    
    # Obtain predictions from netwrok (outputs)
    T_predicted = NN_output(Collocation_temps)
    
    #print(T_predicted[-1])
    
    # Obtain dT/dt
    dT_dt = gradient(T_predicted, Collocation_temps)
    
    # Obtain h as learnable parameter
    #h = NN_output.h
    
    # Newton's Law of Cooling
    NLC = ((h * A) / (m * c_p*1e3)) * (T_env - T_predicted) - dT_dt
    
    print(torch.mean(NLC**2))
    print((NLC**2).mean())

    return torch.mean(NLC**2)

# Define boundary condtion starting temperature
def BC_start(NN_output: torch.nn.Module, used_inputs):
    # Setup input network (t = 0)
    time_start = torch.tensor(0.).view(-1,1).requires_grad_(True)
    
    # Obtain input temperautre prediction from network
    Temp_start = NN_output(time_start)
    
    return torch.mean((Temp_start - T_0)**2)

def BC_infinity(NN_output: torch.nn.Module, used_inputs):
    # Setup input network (t = 0)
    time_inf = torch.tensor(20e3).view(-1,1).requires_grad_(True)
    
    # Obtain input temperautre prediction from network
    Temp_inf = NN_output(time_inf)
    #print(Temp_inf)
    
    return torch.mean((Temp_inf - T_env)**2)

def Far_field_loss(NN_output: torch.nn.Module, used_inputs):
    # Setup collocation points over domain
    start, end = 0, 15e3
    steps = 300
    #Collocation_temps = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    
    time_inputs = torch.linspace(start+1000, end, int((steps/10) * 5)).view(-1, 1).requires_grad_(True)
    log_space1 = torch.logspace(start=torch.log10(torch.tensor(1e-16)),end=torch.log10(torch.tensor(start+1000)),steps=int((steps/10) * 5)).view(-1,1).requires_grad_(True)

    # Concatenate with reversed version to get more density at both start and end
    Collocation_temps = torch.cat((log_space1, time_inputs), dim=0)
    
    # Filter collocation points in the far-field domain
    far_field_points = Collocation_temps[Collocation_temps[:, 0] > 11e3]
    
    # Predict T for far-field points
    T_predicted = NN_output(far_field_points)
    
    # Penalize deviations from T_env in the far-field
    return torch.mean((T_predicted - T_env) ** 2)




'''
# Scaled losses
def Newtons_cooling_law_scaled(NN_output: torch.nn.Module, used_inputs):
    # Setup collocation points over domain
    start, end = 0, 15e3
    steps = 1000
    #Collocation_temps = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    
    
    time_inputs = torch.linspace(start+1000, end, int((steps/10) * 5)).view(-1, 1).requires_grad_(True)
    log_space1 = torch.logspace(start=torch.log10(torch.tensor(1e-16)),end=torch.log10(torch.tensor(start+1000)),steps=int((steps/10) * 5)).view(-1,1).requires_grad_(True)

    # Concatenate with reversed version to get more density at both start and end
    Collocation_temps = torch.cat((log_space1, time_inputs), dim=0)
    
    # Scale collocation temps
    Collocation_temps_scaled = torch.log1p(Collocation_temps)
    
    # Obtain predictions from netwrok (outputs)
    T_predicted = NN_output(Collocation_temps_scaled)
    
    # Obtain dT/dt
    dT_dt_scaled = gradient(T_predicted, Collocation_temps_scaled) #* (1 / (1 + Collocation_temps))
    
    # Obtain h as learnable parameter
    #c_p = NN_output.c_p
    
    
    
    # Newton's Law of Cooling
    eq = (((h * A) / (m * c_p*1e3)) * (T_env - T_predicted)) * (1 + Collocation_temps)
    
    NLC = eq - dT_dt_scaled

    return torch.mean(NLC**2)

# Define boundary condtion starting temperature
def BC_start_scaled(NN_output: torch.nn.Module, used_inputs):
    # Setup input network (t = 0)
    time_start = torch.tensor(0.).view(-1,1).requires_grad_(True)
    
    # Scaled input
    time_start_scaled = torch.log1p(time_start)
    
    # Obtain input temperautre prediction from network
    Temp_start = NN_output(time_start_scaled)
    
    return torch.mean((Temp_start - T_0)**2)
'''

# Initialize learnable paramaters
learn_parameters = {
    "c_p" : 0.84
    }



# Initialize loss functions
loss_functions = {
    Newtons_cooling_law: 1,
    BC_start: 1,
    #BC_infinity: 1e-7
    #Far_field_loss: 1e-8
    #Newtons_cooling_law_scaled: 1,
    #BC_start_scaled: 1
    }

# Create network
PINN_cooling_down_experiment = NN(1, 1, 64, 2, epochs=50000, batch_size=64, use_batch=False, lr=5e-5, loss_terms=loss_functions, learn_params=None, loss_init=None)




# Print init weights
#PINN_cooling_down_experiment.print_initial_weights()


'''
# Initialize loss functions
loss_functions = {
    Newtons_cooling_law: 115000,
    BC_start: 1
    }

# Create network
PINN_cooling_down_experiment = NN(1, 1, 64, 5, epochs=70000, batch_size=64, use_batch=False, lr=1e-3, loss_terms=None, learn_params=None)

'''

# Start time
start_time = time.time()

# Train network
PINN_cooling_down_experiment.fit(X_train, y_train)

# End time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:f} seconds")


# Plot predictions and print final values for net paramater discovery values
#h_data = PINN_cooling_down_experiment.h.data.numpy()[0]
#print(f'Estimation of parameter h: {h_data}')
#time_test_long = np.arange(0, 86.4e3, 1)
#predictions_long = PINN_cooling_down_experiment.predict(time_test_long)

time_test, true_values, predictions, error_average, error_max = prep_data_cooling(data_dir, scaled_inputs=False, scaled_outputs=False, n_outputs=1, n_samples=1e6, start_time=None, end_time=None).compare_with_data(PINN_cooling_down_experiment.predict)




# Print avergae error
print(f"Average error: {error_average:f} [deg C]")
print(f"Average error: {error_max:f} [deg C]")


'''
# Testing
time_test = np.arange(0, 10000, 0.1)
true_values = T_env + (T_0 - T_env) * np.exp(-k*time_test)
predictions = PINN_cooling_down_experiment.predict(time_test)
'''


# Plot the predicted vs actual values
plt.figure(figsize=(10, 6))

# Primary axis for Chamber Pressure
plt.plot(time_test, predictions, '-', label="Predicted Temperature")
plt.plot(time_test, true_values, '-', label="Train Data Temperature")
#plt.plot(X_train, y_train, 'o', label="Training Data Tempereature")
#plt.plot(np.expm1(time_test), predictions, '-', label="Predicted Temperature")
#plt.plot(np.expm1(time_test), true_values, '-', label="Train Data Temperature")
#plt.plot(np.expm1(X_train), y_train, 'o', label="Training Data Tempereature")
plt.plot()
plt.xlabel('Time [s]')
plt.ylabel(r'Temperature [$^\circ$ C]')

# Title and legend
plt.title('Predicted vs Actual Chamber Pressure and Thrust')
plt.legend(loc="upper right")

plt.show()



# Save entire trained model into a certain folder

#filt_data_filename = os.path.join(data_dir_trained_PINN, "Cooling_PINN_trained.pth")
#torch.save(PINN_solid_rocket_motor_network.state_dict(), filt_data_filename)






