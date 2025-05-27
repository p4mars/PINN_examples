# Imported modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simpson
from scipy.optimize import fsolve, curve_fit
import pandas as pd
import time



# Constants
g_0 = 9.80665 # Gravitational acceleration
R_a = 8.3144 # Gas constant
P_a = 101325/1e5  # Standard pressure at sea level in bar




# - - - - - - - - - - - - - - - - - - - - - Inputs - - - - - - - - - - - - - - - - - - - - - - - -

# Load thrust data
Thrust_data = 1



# Grain 
L_grain = 105e-3# Length of grain [m]
#L_grain = 50e-3
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
gamma =  1.137 # Specific heat of combustion mixture [-]
W_g = 39.9e-3  # Molecular mass of mixture [kg]
R_g = R_a / W_g 
T_0 = 1600 # Temperature of combustion mixture [K]


# Motor specifications
d_chamber = 80e-3 # Outside diamter of the chamber [m]
L_chamber = 118.5e-3 # Length of the chamber [m]
d_t = 8.37e-3
r_t = d_t/2 # Nozzle throat radius [m]
#r_t = 7e-3/2
d_e = 16.73e-3
r_e = d_e/2 # Nozzle exit radius [m]
#r_e = 14e-3/2
A_t = np.pi * r_t**2  # Nozzle throat area [m]
A_e = np.pi * r_e**2 # Nozzle exit area [m]
t_b = 10 # Burn time [s]
alpha = 1
alpha = np.deg2rad(alpha)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Performance coefficients
eta_n = 1 # Works
eta_c = 1
eta_f = 1

# Lists for intermediate values in simulation
V_c_lst = []
A_b_lst = []
t_lst = []
r_lst = []
dpdt_lst = []
test_lst = []

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




# Thrust, Cf, c* and realtime Isp calculation (Note: Real time Isp is negative for the negative thrust values, please ignore)
pepc = fsolve(AeAt_func, 0.001, args=(Gamma(gamma), gamma, A_e, A_t))
P_e_sol = P_0_sol*1e5 * pepc
C_f = Cf(Gamma(gamma), gamma, P_e_sol, P_0_sol*1e5, P_a*1e5, (A_e / A_t))
Thrust1 = C_f * P_0_sol*1e5 * A_t * eta_n
m_dot_tot = m_dot_id(Gamma(gamma), P_0_sol*1e5, A_t, R_g, T_0)
#U_e = Ue(gamma, R_g, T_0, P_e_sol*1e5, P_0_sol*1e5)
c_star_ideal = c_star(Gamma(gamma), R_g, T_0)


Thrust = C_f * P_0_sol*1e5 * A_t * eta_n




# Create a 2x2 grid of subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 12))

# Start and stop times plot
t_start_plot = -1
t_end_plot = t_b + 1





# Plot data in each subplot
#axs[0, 0].plot(sol.t, P_0_sol, label="Simulation data")
axs[0, 0].plot(sol.t, P_0_sol, label="Simulation data")
axs[0, 0].set_title('Chamber pressure vs. Time')
axs[0, 0].set_xlabel('Time [s]')
#axs[0, 0].set_ylabel('Chamber pressure [Bar]')
axs[0, 0].set_ylabel('Chamber pressure [bar]')
#axs[0, 0].set_xscale('log')
axs[0, 0].set_xlim(t_start_plot, t_end_plot)
axs[0, 0].grid(which='both', linestyle='--')


axs[0, 1].plot(sol.t, Thrust1, label="Simulation data")
axs[0, 1].set_title('Thrust vs. Time')
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('Thrust [N]')
#axs[0, 1].set_xscale('log')
axs[0, 1].set_xlim(t_start_plot, t_end_plot)
axs[0, 1].grid(which='both', linestyle='--')





# Add overall title and adjust layout
fig.suptitle('Solid rocket engine')
plt.tight_layout()

# Show the plot
plt.show()



'''
save_data = int(input("Store data? [1 = yes, 0 = no]"))

if save_data == 1:
    # Save to CSV
    df = pd.DataFrame({
        "time": sol.t,
        "pressure": P_0_sol/1e5,
        "thrust": Thrust1
    })
    
    output_csv_path = "output_data"  # Specify the path where you want to save the CSV
    df.to_csv(output_csv_path, index=False)
else:
    pass
'''
