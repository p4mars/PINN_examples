# Imported modules
from NN_tools_thesis import NN, gradient
import numpy as np
import matplotlib.pyplot as plt
import torch
from Validation_class import prep_data_cooling_surrogate
import time
import pandas as pd
import torch.nn as nn

# Constants
h = 5 # Convective heat coefficient of air 

# Import training data
# Multiple file usage
data_dir = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Training_data/'  # Path to CSV file(s)
data_dir_single = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Training_data_single/'
data_dir_trained_PINN = '/Users/tristanhirs/Downloads/Thesis/Python code/PINN_trained_models/'
data_dir_val = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Validation_data/'
data_dir_sim_data = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Num_sim_data/num_sim_test.csv'

X_train, y_train, X_val, y_val = prep_data_cooling_surrogate(data_dir_single, scaled=False, n_outputs=1, n_samples=5, start_time=0, end_time=5000).train_val_split(comb_data=True)


time_test_plot, true_values_plot, _, _, _, _ = prep_data_cooling_surrogate(data_dir_single, scaled=False, n_outputs=1, n_samples=1e6, start_time=None, end_time=None).compare_with_data()



# Setup of figure to show progress during training
sim_data = pd.read_csv(data_dir_sim_data)


sim_time = sim_data["Time"].to_numpy()
sim_temperature = sim_data["Temperature"].to_numpy()




executed = False  # Global flag
fig, axs = None, None
call_count = 0
text_annotation = None




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


def show_training_process(NN_output: torch.nn.Module, used_inputs):
    global executed, fig, axs, call_count, text_annotation
    
    call_count += 1  # Increment call counter (amount of epochs)
    
    if not executed:
        plt.ion()  # Turn on interactive mode
        fig, axs = plt.subplots(1, 1, figsize=(12, 6))
        executed = True
    
    ax1 = axs

    # Collocation points time array
    start, end = 0, 12e3
    steps = 2000
    time_inputs = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    
    
    #print(time_inputs)
    
    ax1.cla()
    
    ax1.plot(time_test_plot[:, 0], true_values_plot, color='Darkorange', label="Test data")
    ax1.plot(sim_time, sim_temperature, '--', color='red', label='Numerical sim')
    ax1.plot(X_train[:, 0], y_train, 'o', color='green', label='Train data')
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel(r'Temperature [$^\circ$C]')
    ax1.set_xlim(-100, 12.5e3)
    ax1.set_ylim(-1, 70)
    ax1.grid(which='both', linestyle='--')
    
    
    
    for inputs in used_inputs:
        # Repeat the constants for each time step (number of time points)
        constant_features_repeated = inputs.unsqueeze(0).repeat(time_inputs.size(0), 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_inputs, constant_features_repeated), dim=1)
        
    
        # Obtain predictions from netwrok (outputs)
        T_predicted = NN_output(input_data)
    
    
        # Show predictions over time during training
        ax1.plot(input_data[:, 0].detach().numpy(), T_predicted.detach().numpy(), '-', label='Temperature prediction', color="C0")
        
    
    ax1.legend(loc='upper right')
    
    # Show and update epoch count
    if text_annotation is None:
        text_annotation = fig.text(0.52, 0.01, f'Epoch: {call_count}', 
                                   ha='center', fontsize=12, color='black')
    else:
        text_annotation.set_text(f'Epoch: {call_count}')  # Update text dynamically

    
    
    plt.pause(0.05)
    
    return 0



# Initialize learnable paramaters
learn_parameters = {
    "h" : 5.
    }



# Initialize loss functions
loss_functions = {
    Newtons_cooling_law: 1000,
    BC_start: 1,
    #show_training_process: 1e-16
    }

# Create network
PINN_cooling_down_experiment = NN(7, 1, 64, 4, epochs=5000, batch_size=64, use_batch=False, lr=1e-4, loss_terms=loss_functions, learn_params=None, unique_input=True)

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


time_test, true_values, predictions, error_average, error_max, _ = prep_data_cooling_surrogate(data_dir_single, scaled=False, n_outputs=1, n_samples=1e6, start_time=None, end_time=None).compare_with_data(PINN_cooling_down_experiment.predict)

# Plot the predicted vs actual values
plt.figure(figsize=(10, 6))

# Primary axis for Chamber Pressure

plt.plot(time_test[:, 0], true_values, '-', label="Experimental Data Temperature", color='darkorange')
plt.plot(sim_time, sim_temperature, '--', color='red', label='Numerical sim')
plt.plot(X_train[:, 0], y_train, 'o', label="Training Data Tempereature", color='green')
plt.plot(time_test[:, 0], predictions, '-', label="Predicted Temperature")
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




