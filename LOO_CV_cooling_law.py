# Imported modules
from NN_tools_thesis import NN, gradient
import numpy as np
import torch
from Validation_class import prep_data_cooling_surrogate, move_datasets_for_loo_cv
import time
from scipy.integrate import solve_ivp
import pandas as pd

# Constants
h = 5 # Convective heat coefficient of air 



# Import training data
# Multiple file usage
data_dir = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Training_data/'  # Path to CSV file(s)
data_dir_val = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Validation_data/'
data_dir_error_file = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Error_file/'

# Lists to save performance
average_error_lst = []
max_error_lst = []
train_time_lst = []
execution_time_lst = []
test_file_lst = []

# - - - - - - - - - - - - - - - - - - - - PINN - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# PINN Functions

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
    
        # Obtain dT/dt_scaled
        dT_dt = gradient(T_predicted, input_data)[:,0].unsqueeze(-1) #* (1 / (1 + Collocation_temps))
    
        # Newton's Law of Cooling
        NLC = (((h * A) / (m * c_p * 1e3)) * (T_env - T_predicted)) - dT_dt
        
        # Determine loss for specific case and add to total loss
        total_loss += torch.mean((NLC)**2) 

    return total_loss / len(used_inputs)

# Define physics loss learnable parameter
def Newtons_cooling_law_learnable_parameter(NN_output: torch.nn.Module, used_inputs):
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
    
        # Obtain dT/dt_scaled
        dT_dt = gradient(T_predicted, input_data)[:,0].unsqueeze(-1) #* (1 / (1 + Collocation_temps))
    
    
        # Obtain h as learnable parameter
        h = NN_output.h
    
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

# Initialize loss functions
loss_functions = {
    Newtons_cooling_law: 1,
    BC_start: 1
    }

# Initialize loss functions
loss_functions_learn = {
    Newtons_cooling_law_learnable_parameter: 1,
    BC_start: 1
    }

# Initialize learnable paramaters
learn_parameters = {
    "h" : 5.
    }



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - Data-driven - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - Numerical Model - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Simulation module
def cool_down(t, y, h, L, R, m, c_p, T_env):    
    # Inital conditions
    T0 = y[0]
    
    # Calculate area for formula
    A = 2 * np.pi * R * L
    
    if T0 <= T_env:
        dT0dt = 0
    else:
        # Newton's Law of Cooling
        dT0dt = ((h * A) / (m * c_p*1e3)) * (T_env - T0)
        
    dydt = [dT0dt]
    return dydt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# - - - - - - - - - - - - - - - - - - - - - - - - MAIN PROGRAM - - - - - - - - - - - - - - - - - - - - - - - - -

# Perform Leave-One-Out Cross-Validation
for test_file in move_datasets_for_loo_cv(data_dir, data_dir_val):
    print(f"Current test dataset: {test_file}")
    
    # Obtain training and validation data
    X_train, y_train, X_val_train, y_val_train = prep_data_cooling_surrogate(data_dir, scaled=False, n_outputs=1, n_samples=63, start_time=None, end_time=None).train_val_split()
    
    # PINN
    # Create network (re-init weights and biases)
    PINN_cooling_down_experiment = NN(7, 1, 64, 4, epochs=50000, batch_size=64, use_batch=False, lr=1e-3, loss_terms=loss_functions, learn_params=None, unique_input=True, show_plots=False)
    
    # Start time
    start_time = time.time()

    # Train network
    PINN_cooling_down_experiment.fit(X_train, y_train)

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time_pinn = end_time - start_time
    
    # Obtain errors
    _ , _, _, error_average_pinn, error_max_pinn, execution_time_pinn = prep_data_cooling_surrogate(data_dir_val, scaled=False, n_outputs=1, n_samples=1e6, start_time=None, end_time=None).compare_with_data(PINN_cooling_down_experiment.predict)
    
    # I-PINN
    # Create network (re-init weights and biases)
    IPINN_cooling_down_experiment = NN(7, 1, 64, 4, epochs=50000, batch_size=64, use_batch=False, lr=1e-3, loss_terms=loss_functions_learn, learn_params=learn_parameters, unique_input=True, show_plots=False)
    
    # Start time
    start_time = time.time()

    # Train network
    IPINN_cooling_down_experiment.fit(X_train, y_train)

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time_ipinn = end_time - start_time
    
    # Obtain errors
    _ , _, _, error_average_ipinn, error_max_ipinn, execution_time_ipinn = prep_data_cooling_surrogate(data_dir_val, scaled=False, n_outputs=1, n_samples=1e6, start_time=None, end_time=None).compare_with_data(IPINN_cooling_down_experiment.predict)
    
    # Obtain h for I-Numerical
    I_h = IPINN_cooling_down_experiment.h.data.numpy()[0]
    
    # Data driven
    # Create network (re-init weights and biases)
    NN_cooling_down_experiment = NN(7, 1, 64, 4, epochs=50000, batch_size=64, use_batch=False, lr=1e-3, show_plots=False)
    
    # Start time
    start_time = time.time()

    # Train network
    NN_cooling_down_experiment.fit(X_train, y_train)

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time_data_driven = end_time - start_time
    
    # Obtain errors
    _ , _, _, error_average_data_driven, error_max_data_driven, execution_time_data_driven = prep_data_cooling_surrogate(data_dir_val, scaled=False, n_outputs=1, n_samples=1e6, start_time=None, end_time=None).compare_with_data(NN_cooling_down_experiment.predict)
    
    # Numerical method
    X_test , T_true, _, _, _, _ = prep_data_cooling_surrogate(data_dir_val, scaled=False, n_outputs=1, n_samples=1e6, start_time=None, end_time=None).compare_with_data()
    
    inputs = X_test[0][1:]

    L = inputs[0]
    R = inputs[1]
    m = inputs[2]
    c_p = inputs[3]
    T_env = inputs[4]
    T_0 = inputs[5]
    

    # Set time of numerical simulation
    t_span =  (0, X_test[:,0][-1])

    # Create a dense set of time points for output
    t_eval = np.linspace(t_span[0], t_span[1], len(X_test[:,0]))  # Timesteps in simulation
    
    # Start time
    start_time = time.time()

    # Obtain_results
    sol = solve_ivp(cool_down, t_span, [T_0], method='RK45', dense_output=True, t_eval=t_eval, args=(h, L, R, m, c_p, T_env))

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    execution_time_num = end_time - start_time
    
    # Obtain results
    T_predict = sol.y[0]
    
    # Obtain errors
    error_avg_num = np.sqrt(np.mean((T_predict - T_true)**2))
    error_max_num = np.sqrt(np.max((T_predict - T_true)**2))
    
    # I-Numerical method
    # Start time
    start_time = time.time()

    # Obtain_results
    sol = solve_ivp(cool_down, t_span, [T_0], method='RK45', dense_output=True, t_eval=t_eval, args=(I_h, L, R, m, c_p, T_env))

    # End time
    end_time = time.time()

    # Calculate the elapsed time
    execution_time_inum = end_time - start_time
    
    # Obtain results
    T_predict = sol.y[0]
    
    # Obtain errors
    error_avg_inum = np.sqrt(np.mean((T_predict - T_true)**2))
    error_max_inum = np.sqrt(np.max((T_predict - T_true)**2))
    
    # Store
    test_file_lst.append(test_file)
    average_error_lst.append([error_average_pinn, error_average_ipinn, error_average_data_driven, error_avg_num, error_avg_inum])
    max_error_lst.append([error_max_pinn, error_max_ipinn, error_max_data_driven, error_max_num, error_max_inum])
    train_time_lst.append([elapsed_time_pinn, elapsed_time_ipinn, elapsed_time_data_driven])
    execution_time_lst.append([execution_time_pinn, execution_time_ipinn, execution_time_data_driven, execution_time_num, execution_time_inum])


# Total average RMSE error
Total_average_error = np.mean(np.array(average_error_lst), axis=0)


# Total average max error
Total_max_error = np.mean(np.array(max_error_lst), axis=0)


# Total average train time
Total_train_time = np.mean(np.array(train_time_lst), axis=0)

# Total average execution time
Total_execution_time = np.mean(np.array(execution_time_lst), axis=0)


# Print errors
formatted_errors_avg = " ".join(f"{error:.6f}" for error in Total_average_error)
print(f"Total average error: {formatted_errors_avg}")

formatted_errors_max = " ".join(f"{error:.6f}" for error in Total_max_error)
print(f"Total max error: {formatted_errors_max}")

formatted_errors_train = " ".join(f"{error:.6f}" for error in Total_train_time)
print(f"Total average train time: {formatted_errors_train}")

formatted_errors_execution = " ".join(f"{error:.6f}" for error in Total_execution_time)
print(f"Total average execution time: {formatted_errors_execution}")



save_data = int(input("Store data? [1 = yes, 0 = no]"))

if save_data == 1:
    # Save to CSV
    df = pd.DataFrame({
        "Test file": test_file_lst,
        "Average error": average_error_lst,
        "Maximum error": max_error_lst,
        "Train time": train_time_lst,
        "Execution time": execution_time_lst
    })
    
    output_csv_path = "/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Error_file/50_points_cooling.csv"  # Specify the path where you want to save the CSV
    df.to_csv(output_csv_path, index=False)

