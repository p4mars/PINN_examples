# Imported modules
import numpy as np
import matplotlib.pyplot as plt
from Loss_function_tools import SRM_helper_tools
from Validation_class import prep_data_surrogate, prep_data
import torch
import pandas as pd
from scipy.integrate import solve_ivp, cumtrapz
from scipy.interpolate import interp1d


# Constants
g_0 = 9.80665 # Gravitational acceleration
R_a = 8.3144 # Gas constant
P_a = 101325/1e5  # Standard pressure at sea level in bar



# Data directory
data_dir_single_data = '/Users/tristanhirs/Downloads/Thesis/Motor test data/Training_data_single/All/'
data_dir_sim_data = '/Users/tristanhirs/Downloads/Thesis/Motor test data/Simulated_data/reference_test_2.csv'

true_time , Experimental_data, _, _, _ = prep_data_surrogate(data_dir_single_data, scaled=False, n_outputs=2, n_samples=5, start_time=None, end_time=None).compare_with_data()


sim_data = pd.read_csv(data_dir_sim_data)

# Define thresholds
pressure_lower = 0.8  # Set your desired lower threshold
pressure_upper = 55  # Set your desired upper threshold

# Apply threshold filtering (keep rows where all values are within range)
sim_data = sim_data[(sim_data["pressure"] >= pressure_lower) & (sim_data["pressure"] <= pressure_upper)]


sim_time = sim_data["time"].to_numpy()
sim_pressure = sim_data["pressure"].to_numpy()
sim_drdt = sim_data["drdt"].to_numpy()
sim_dPdt = sim_data["dPdt"].to_numpy()
sim_A_b = sim_data["A_b"].to_numpy()
sim_V_c = sim_data["V_c"].to_numpy()

# Interpolate dPdt to make it a continuous function
dPdt_interp = interp1d(sim_time, sim_dPdt, kind="linear", fill_value="extrapolate")

# Define the ODE function: dP/dt = f(t)
def dPdt_function(t, P):
    return dPdt_interp(t)  # Use interpolated dPdt values

# Initial pressure (modify as needed)
P0 = P_a

# Solve the ODE using RK45
solution = solve_ivp(dPdt_function, (sim_time[0], sim_time[-1]), [P0], method="RK45")

# Extract the integrated pressure values
pressure_values = solution.y[0]

# Compute pressure by integrating dP/dt using the trapezoidal rule
pressure_values_2 = P0 + cumtrapz(sim_dPdt, sim_time)



#plt.plot(sim_time, sim_pressure)
#plt.plot(solution.t, pressure_values, 'o')
#plt.plot(sim_time[1:], pressure_values_2, 'o')




# Unzip constant values from inputs
inputs = torch.tensor([true_time[0][1:]])[0]

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
t_b = 10 # Burn time [s]

P_C_real = torch.tensor(Experimental_data[:, 0]).unsqueeze(-1)
time_inputs_real = torch.tensor(true_time[:, 0])

sorted_pressure = torch.tensor(sim_pressure).unsqueeze(-1)
sim_time_tensor = torch.tensor(sim_time)

# Obtain r_integrator
radius = SRM_helper_tools.r_integrator(sorted_pressure, a, n, sim_time_tensor, d_grain_in/2)

# Check when motor is burning or not
burning_logic = torch.logical_and(radius <= (d_grain_out/2), (radius - (d_grain_in/2)) <= L_grain/2)

# Stop radius increase if burning_logic is False
radius[radius >= (d_grain_out/2)] = d_grain_out/2
radius[(radius - (d_grain_in/2)) >= L_grain/2] = L_grain/2

# Obtain dPdt from formula
dPdt_theory = SRM_helper_tools.dPdt_check(radius, sorted_pressure, P_a, burning_logic, a, n, rho_grain, R_g, T_0, A_t, gamma, uninhibited_core, uninhibited_sides, L_grain, d_grain_in/2, d_grain_out, N_grains, L_chamber, d_chamber, t_liner)

# ntegrate to get PC
P_C_theory = P0 + torch.cumulative_trapezoid(dPdt_theory, sim_time_tensor, dim=0).reshape(-1, 1)
P_C_theory = torch.cat((torch.tensor([P0]).unsqueeze(0), P0 + P_C_theory))


# Compute pressure by integrating dP/dt using the trapezoidal rule
radius_sim = r_grain_init + cumtrapz(sim_drdt, sim_time)

dPdt_theory = dPdt_theory.squeeze(-1)

plt.plot(sim_time, sim_dPdt, 'o')
plt.plot(sim_time, dPdt_theory, 'o')
#plt.plot(sim_time, torch.abs(dPdt_theory-sim_dPdt), 'o')


