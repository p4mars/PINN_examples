# Imported modules
from NN_tools_thesis import NN, gradient
import numpy as np
import matplotlib.pyplot as plt
import torch
from Validation_class import prep_data_cooling_surrogate
import time
import os


# Constants
h = 5 # Convective heat coefficient of air 

# Import training data
# Multiple file usage
data_dir = 'data_dir'  # Path to CSV file(s)
data_dir_single = 'data_dir'
data_dir_trained_PINN = 'PINN_model'
data_dir_val = 'data_dir_val'


X_train, y_train, X_val, y_val = prep_data_cooling_surrogate(data_dir, scaled=False, n_outputs=1, n_samples=50, start_time=None, end_time=None).train_val_split()

'''
plt.plot(np.expm1(X_train[:,0]), y_train, 'o')
plt.plot(np.expm1(X_val[:,0]), y_val,'o')
plt.show()
'''

# PINN


# Define physics loss

def Newtons_cooling_law(NN_output: torch.nn.Module, used_inputs):
    # Setup collocation points over domain
    start, end = 0, 15e3
    steps = 500
    #Collocation_temps = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    time_inputs = torch.linspace(start+1000, end, int((steps/10) * 5)).view(-1, 1).requires_grad_(True)
    log_space1 = torch.logspace(start=torch.log10(torch.tensor(1e-16)),end=torch.log10(torch.tensor(start+1000)),steps=int((steps/10) * 5)).view(-1,1).requires_grad_(True)

    # Concatenate with reversed version to get more density at both start and end
    Collocation_temps = torch.cat((log_space1, time_inputs), dim=0)
    
    # Reset total loss
    total_loss = 0
    
    for inputs in used_inputs:
        
        L = inputs[0]
        R = inputs[1]
        m = inputs[2]
        c_p = inputs[3]
        T_env = inputs[4]
        
        
        # Calculate area for formula
        A = 2 * np.pi * R * L
        
        
        # Repeat the constants for each time step (number of time points)
        constant_features_repeated = inputs.unsqueeze(0).repeat(Collocation_temps.size(0), 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((Collocation_temps, constant_features_repeated), dim=1)
        
    
        # Obtain predictions from netwrok (outputs)
        T_predicted = NN_output(input_data)
    
        # Obtain dT/dt
        dT_dt = gradient(T_predicted, input_data)[:,0].unsqueeze(-1) 
    
    
        # Obtain h as learnable parameter
        #h = NN_output.h
        #c_p = NN_output.c_p
    
        # Newton's Law of Cooling
        NLC = (((h * A) / (m * c_p * 1e3)) * (T_env - T_predicted)) - dT_dt
        
        # Determine loss for specific case and add to total loss
        total_loss += torch.mean((NLC)**2) 

    return total_loss / len(used_inputs)




# Define boundary condtion starting temperature
def BC_start(NN_output: torch.nn.Module, used_inputs):
    # Setup input network (t = 0)
    time_start = torch.tensor(0.).view(-1,1).requires_grad_(True)
    
    
    # Reset total loss
    total_loss = 0
    
    for inputs in used_inputs:
        
        # Obtain starting temperature
        T_0 = inputs[5]
        
        # Repeat the constants for each time step (number of time points)
        constant_features_repeated = inputs.unsqueeze(0).repeat(time_start.size(0), 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_start, constant_features_repeated), dim=1)
        
    
        # Obtain input temperautre prediction from network
        Temp_start = NN_output(input_data)
        
        # Determine loss for specific case and add to total loss
        total_loss += torch.mean((Temp_start - T_0)**2)
    
    return total_loss / len(used_inputs)

'''
# Define boundary condtion minimum temperature
def BC_min_temp(NN_output: torch.nn.Module, used_inputs):
    # Setup collocation points over domain
    start, end = 0, 15e3
    steps = 100
    
    Collocation_temps = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    #time_inputs = torch.linspace(start+1000, end, int((steps/10) * 5)).view(-1, 1).requires_grad_(True)
    #log_space1 = torch.logspace(start=torch.log10(torch.tensor(1e-16)),end=torch.log10(torch.tensor(start+1000)),steps=int((steps/10) * 5)).view(-1,1).requires_grad_(True)
 
    # Concatenate with reversed version to get more density at both start and end
    #Collocation_temps = torch.cat((log_space1, time_inputs), dim=0)
    
    
    # Reset total loss
    total_loss = 0
    
    for inputs in used_inputs:
        
        T_env = inputs[4]
        
        # Repeat the constants for each time step (number of time points)
        constant_features_repeated = inputs.unsqueeze(0).repeat(Collocation_temps.size(0), 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((Collocation_temps, constant_features_repeated), dim=1)
    
        # Obtain input temperautre prediction from network
        T_predicted = NN_output(input_data)
        
        # Calculate loss
        total_loss += torch.mean(torch.relu(T_env - T_predicted)**2)
    
    return total_loss / len(used_inputs)
    

# Define boundary condtion starting temperature
def BC_infinity(NN_output: torch.nn.Module, used_inputs):
    # Setup input network (t = 0)
    time_start = torch.tensor(1e6).view(-1,1).requires_grad_(True)
    
    
    # Reset total loss
    total_loss = 0
    
    for inputs in used_inputs:
        
        # Get environmental temperature from data
        T_env = inputs[4]
        
        # Repeat the constants for each time step (number of time points)
        constant_features_repeated = inputs.unsqueeze(0).repeat(time_start.size(0), 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_start, constant_features_repeated), dim=1)
    
        # Obtain input temperautre prediction from network
        Temp_infinity = NN_output(input_data)
        
        # Determine loss for specific case and add to total loss
        total_loss += torch.mean((Temp_infinity - T_env)**2)
    
    return total_loss / len(used_inputs)

# Define boundary condtion starting temperature
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
    
    
    # Reset total loss
    total_loss = 0
    
    for inputs in used_inputs:
        
        # Get environmental temperature from data
        T_env = inputs[4]
        
        # Repeat the constants for each time step (number of time points)
        constant_features_repeated = inputs.unsqueeze(0).repeat(far_field_points.size(0), 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((far_field_points, constant_features_repeated), dim=1)
    
        # Obtain input temperautre prediction from network
        Temp_infinity = NN_output(input_data)
        
        # Determine loss for specific case and add to total loss
        total_loss += torch.mean((Temp_infinity - T_env)**2)
    
    return total_loss / len(used_inputs)

'''



# Initialize learnable paramaters
learn_parameters = {
    "h" : 5.
    }



# Initialize loss functions
loss_functions = {
    Newtons_cooling_law: 1,
    BC_start: 1
    }

# Create network
PINN_cooling_down_experiment = NN(7, 1, 64, 4, epochs=50000, batch_size=64, use_batch=False, lr=1e-5, loss_terms=loss_functions, learn_params=None, unique_input=True)

'''
# Initialize loss functions
loss_functions = {
    Newtons_cooling_law: 2.9,
    BC_start: 1
    }


# Create network
PINN_cooling_down_experiment = NN(7, 1, 50, 2, epochs=30000, batch_size=64, use_batch=False, lr=1e-3, loss_terms=loss_functions, learn_params=None, unique_input=True, activation_fn=nn.Tanh, loss_init=None)
'''

'''
# Initialize loss functions
loss_functions = {
    Newtons_cooling_law: 100,
    #BC_start: 1
    }


# Create network
PINN_cooling_down_experiment = NN(7, 1, 100, 3, epochs=30000, batch_size=64, use_batch=False, lr=1e-4, loss_terms=loss_functions, learn_params=None, unique_input=True)
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


time_test, true_values, predictions, error_average, error_max, _ = prep_data_cooling_surrogate(data_dir_val, scaled=False, n_outputs=1, n_samples=1e6, start_time=None, end_time=None).compare_with_data(PINN_cooling_down_experiment.predict)

# Plot the predicted vs actual values
plt.figure(figsize=(10, 6))

# Primary axis for Chamber Pressure
plt.plot(time_test[:, 0], predictions, '-', label="Predicted Temperature")
plt.plot(time_test[:, 0], true_values, '-', label="Experimental Data Temperature")
#plt.plot(X_train[:, 0], y_train, 'o', label="Training Data Tempereature")
#plt.plot(np.expm1(time_test[:,0]), predictions, '-', label="Predicted Temperature")
#plt.plot(np.expm1(time_test[:,0]), true_values, '-', label="Input Data Temperature")
#plt.plot(np.expm1(X_train[:,0]), y_train, 'o', label="Training Data Tempereature")
plt.plot()
plt.xlabel('Time [s]')
plt.ylabel(r'Temperature [$^\circ$C]')
plt.grid(which='both', linestyle='--')

# Title and legend
plt.title('Temperature vs Time')
plt.legend(loc="upper right")

plt.show()



# Save entire trained model into a certain folder
'''
filt_data_filename = os.path.join(data_dir_trained_PINN, "Cooling_PINN_trained.pth")
torch.save(PINN_cooling_down_experiment.state_dict(), filt_data_filename)

'''




