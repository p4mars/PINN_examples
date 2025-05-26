# Imported modules
from NN_tools_thesis import NN
import numpy as np
import matplotlib.pyplot as plt
import torch
from Validation_class import prep_data_cooling_surrogate
import time
import os


# Import training data
# Multiple file usage
data_dir = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Training_data/'  # Path to CSV file(s)
data_dir_single = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Training_data_single/'
#data_dir_trained_PINN = '/Users/tristanhirs/Downloads/Thesis/Python code/PINN_trained_models/'
data_dir_val = '/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Validation_data/'

X_train, y_train, X_val, y_val = prep_data_cooling_surrogate(data_dir_val, scaled=False, n_outputs=1, n_samples=50, start_time=None, end_time=None).train_val_split()


'''
plt.plot(np.expm1(X_train[:,0]), y_train, 'o')
plt.plot(np.expm1(X_val[:,0]), y_val,'o')
plt.show()
'''

# Create network
NN_cooling_down_experiment = NN(7, 1, 64, 4, epochs=50000, batch_size=64, use_batch=False, lr=1e-5)

# Start time
start_time = time.time()

# Train network
NN_cooling_down_experiment.fit(X_train, y_train)

# End time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:f} seconds")


# Plot predictions and print final values for net paramater discovery values
#cp_data = PINN_cooling_down_experiment.c_p.data.numpy()[0]
#print(f'Estimation of parameter Cp: {cp_data}')

time_test, true_values, predictions, error_average, error_max, _ = prep_data_cooling_surrogate(data_dir_val, scaled=False, n_outputs=1, n_samples=1e6, start_time=None, end_time=None).compare_with_data(NN_cooling_down_experiment.predict)

# Plot the predicted vs actual values
plt.figure(figsize=(10, 6))

# Primary axis for Chamber Pressure
plt.plot(time_test[:, 0], predictions, '-', label="Predicted Temperature")
plt.plot(time_test[:, 0], true_values, '-', label="Experimental Data Temperature")
#plt.plot(X_train[:, 0], y_train, 'o', label="Training Data Tempereature")
#plt.plot(np.expm1(time_test[:,0]), predictions, '-', label="Predicted Temperature")
#lt.plot(np.expm1(time_test[:,0]), true_values, '-', label="Input Data Temperature")
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

#filt_data_filename = os.path.join(data_dir_trained_PINN, "Cooling_PINN_trained.pth")
#torch.save(PINN_solid_rocket_motor_network.state_dict(), filt_data_filename)






