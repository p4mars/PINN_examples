# Imported modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import fsolve, curve_fit
import pandas as pd
import time
from Validation_class import prep_data_surrogate


# Constants
g_0 = 9.80665 # Gravitational acceleration
R_a = 8.3144 # Gas constant
P_a = 101325/1e5  # Standard pressure at sea level in bar

# - - - - - - - - - - - - - - - - - - - - - Inputs - - - - - - - - - - - - - - - - - - - - - - - -
# Data directory
data_dir = 'data_dir'
data_dir_val = 'val_data'
data_dir_single = 'data_dir_single'

true_time , Experimental_data, _, _, _, _, _ = prep_data_surrogate(data_dir_single, scaled=False, n_outputs=2, n_samples=1e12, start_time=None, end_time=None).compare_with_data()


# Unzip constant values from inputs
inputs = true_time[0][1:]


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

t_shift = 0.32150415
a = 4.7037086
n = 0.14675261

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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Performance coefficients
eta_n = 1 # Works
eta_c = 1
eta_f = 1

# Simulation settings


# Modules (Note: not all of the functions are used)
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


# Initial conditions
y0 = [P_a, r_grain_init]


# Solve IMPORTANT TO SET TIME LONG ENOUGH
#t_span =  (0, time_true[-1])
t_span =  (0, true_time[:,0][-1])


# Create a dense set of time points for output
t_eval = np.linspace(t_span[0], t_span[1], len(true_time[:,0]))  # Timesteps in simulation

# Start time
start_time = time.time()

# Solve IMPORTANT TO SET TIME LONG ENOUGH
sol = solve_ivp(solid, t_span, y0, method='RK45', dense_output=True, t_eval=t_eval, args=(L_grain, d_grain_in, t_liner, d_grain_out, r_grain_init, mass_grain_init, rho_grain, N_grains, a, n, uninhibited_core, uninhibited_sides, gamma, W_g, T_0, R_g, d_chamber, L_chamber, d_t, d_e, r_t, r_e, A_e, A_t))

# End time
end_time = time.time()

# Obtain pressure from simulation
P_0_sol = sol.y[0]


# Obtain thrust
pepc = fsolve(AeAt_func, 0.001, args=(Gamma(gamma), gamma, A_e, A_t))
P_e_sol = P_0_sol*1e5 * pepc
C_f = Cf(Gamma(gamma), gamma, P_e_sol, P_0_sol*1e5, P_a*1e5, (A_e / A_t))
Thrust1 = C_f * P_0_sol*1e5 * A_t * eta_n


# Plot the predicted vs actual values
fig2, axs2 = plt.subplots(1,2,figsize=(12, 6))

# Start and stop times plot
t_start_plot = t_span[0] - 0.5
t_end_plot = t_span[-1] + 0.5

# Primary axis for Chamber Pressure
axs2[0].plot(true_time[:, 0], Experimental_data[:, 0], '-', label="Experimental Data Pressure", color='darkorange')
axs2[0].plot(sol.t, P_0_sol, '-', label="Predicted Pressure")
axs2[0].set_xlabel('Time [s]')
axs2[0].set_ylabel('Chamber Pressure [Bar]')
axs2[0].set_xlim(t_start_plot, t_end_plot)
axs2[0].grid(which='both', linestyle='--')
axs2[0].legend()

axs2[1].plot(true_time[:, 0], Experimental_data[:, 1], '-', label="Experimental Data Thrust", color='darkorange')
axs2[1].plot(sol.t, Thrust1, '-', label="Predicted Thrust")
axs2[1].set_xlabel('Time [s]')
axs2[1].set_ylabel('Thrust [N]')
axs2[1].set_xlim(t_start_plot, t_end_plot)
axs2[1].grid(which='both', linestyle='--')
axs2[1].legend()


# Title and legend
fig2.suptitle('Predicted vs Actual Chamber Pressure and Thrust')
plt.show()


error_average_pressure = np.sqrt(np.mean((Experimental_data[:, 0] - P_0_sol)**2))
error_max_pressure = np.sqrt(np.max((Experimental_data[:, 0] - P_0_sol)**2))
error_peak_pressure = np.sqrt((np.max(Experimental_data[:, 0]) - np.max(P_0_sol))**2) 

error_average_thrust = np.sqrt(np.mean((Experimental_data[:, 1] - Thrust1)**2))
error_max_thrust = np.sqrt(np.max((Experimental_data[:, 1] - Thrust1)**2))
error_peak_thrust = np.sqrt((np.max(Experimental_data[:, 1]) - np.max(Thrust1))**2)


# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:f} seconds")

# Print avergae error
print(f"Average error pressure: {error_average_pressure:f} [bar]")
print(f"Average error thrust: {error_average_thrust:f} [N]")

# Print max error
print(f"Max error pressure: {error_max_pressure:f} [bar]")
print(f"Max error thrust: {error_max_thrust:f} [N]")

# Print peak errors
print(f"Peak error pressure: {error_peak_pressure:f} [bar]")
print(f"Peak error thrust: {error_peak_thrust:f} [N]")




