# Imported modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import fsolve, curve_fit
import pandas as pd
import time
from Validation_class import prep_data_cooling_surrogate


# Constants
g_0 = 9.80665 # Gravitational acceleration
R_a = 8.3144 # Gas constant
P_a = 101325/1e5  # Standard pressure at sea level in bar


# Import training data
# Multiple file usage
data_dir = 'data_dir'
data_dir_single = 'data_dir_sinlge'
data_dir_val = 'validation_data'

time_true, T_true, _ , _, _, _ = prep_data_cooling_surrogate(data_dir_single, scaled=False, n_outputs=1, n_samples=1e6, start_time=None, end_time=None).compare_with_data()


# - - - - - - - - - - - - - - - - - - - - - Inputs - - - - - - - - - - - - - - - - - - - - - - - -
h = 5

'''
r = 0.0326
L = 0.0708
A = 2 * np.pi * r * L
m = 0.192
c_p = 0.84
T_0 = T_true[0]
T_env = 6
'''


inputs = time_true[0][1:]

L = inputs[0]
R = inputs[1]
m = inputs[2]
c_p = inputs[3]
T_env = inputs[4]
T_0 = inputs[5]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


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

# Initial conditions
y0 = [T_0]



# Solve IMPORTANT TO SET TIME LONG ENOUGH
#t_span =  (0, time_true[-1])
t_span =  (0, time_true[:,0][-1])


# Create a dense set of time points for output
t_eval = np.linspace(t_span[0], t_span[1], len(time_true[:,0]))  # Timesteps in simulation


# Start time
start_time = time.time()

# Obtain_results
sol = solve_ivp(cool_down, t_span, [T_0], method='RK45', dense_output=True, t_eval=t_eval, args=(h, L, R, m, c_p, T_env))

# End time
end_time = time.time()

# Obtain results
T_predict = sol.y[0]

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(1, 1, figsize=(10, 6))

# Start and stop times plot
t_start_plot = t_span[0] - 100
t_end_plot = t_span[-1] + 100

# Plot data in each subplot
#axs[0, 0].plot(sol.t, P_0_sol, label="Simulation data")
axs.plot(sol.t, T_predict, label="Simulation data")
axs.plot(sol.t, T_analytical, label="Analytcial model")
#axs.plot(time_true[:, 0], T_true, label="Experimental data")
#axs.plot(np.expm1(sol.t), T_predict, label="Simulation data")
#axs.plot(time_true, T_true, label="Experimental data")
axs.set_title('Temperature vs. Time')
axs.set_xlabel('Time [s]')
#axs[0, 0].set_ylabel('Chamber pressure [Bar]')
axs.set_ylabel(r'Temperature [$^\circ$C]')
#axs[0, 0].set_xscale('log')
axs.set_xlim(t_start_plot, t_end_plot)
axs.grid(which='both', linestyle='--')
axs.legend()

# Show the plot
plt.show()


# Average error
error_avg = np.sqrt(np.mean((T_predict - T_true)**2))
error_max = np.sqrt(np.max((T_predict - T_true)**2))
print(f"Average error: {error_avg:f} [deg C]")
print(f"Maximum error: {error_max:f} [deg C]")

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:f} seconds")



'''
save_data = int(input("Store data? [1 = yes, 0 = no]"))

if save_data == 1:
    # Save to CSV
    df = pd.DataFrame({
        "Time": sol.t,
        "Temperature": T_predict,
    })
    
    output_csv_path = "/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Num_sim_data/num_sim_test.csv"  # Specify the path where you want to save the CSV
    df.to_csv(output_csv_path, index=False)
else:
    pass
'''
