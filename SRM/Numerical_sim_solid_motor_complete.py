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

'''
# Verification runs
# Grain 
L_grain = 35e-2/3 # Length of grain [m]
rho_grain = 1260 # Density of grain [kg/m3]
T_grain = 298 # Temperarture of grain [K]
d_grain_in = 3e-2 # Port diameter grain [m]
d_grain_out = 6.6e-2 # Outside diameter of grain [m]
t_liner = 0
r_grain_init = d_grain_in / 2 # Port radius [m]
mass_grain_init = L_grain * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2) * rho_grain # Inital mass of grain [kg]
n = 0.16
a = 0.132*10*(1000**n)

# Inhibiters
uninhibited_core = 1
uninhibited_sides = 1
N_grains = 3

# Combustion mixture (predicted using CEA)
gamma =  1.18 # Specific heat of combustion mixture[-]
W_g =  23e-3 # Molecular mass of mixture
R_g = R_a / W_g 
T_0 = 2900  # Temperature of combustion mixture

# Engine
d_chamber = d_grain_out
L_chamber = L_grain * N_grains
r_t = 1.55e-2/2 # Nozzle throat radius [m]
r_e = 3.1e-2/2 # Nozzle exit radius [m]
A_t = np.pi * r_t**2 # Nozzle throat area [m]
A_e = np.pi * r_e**2 # Nozzle exit area [m]
t_b = 5 # Burn time

# Initial conditions
P_a = 86e3/1e5
'''

# Example test case
'''
# Grain 
L = 35e-2 # Length of grain [m]
rho_grain = 1260 # Density of grain [kg/m3]
T_grain = 298 # Temperarture of grain [K]
d_grain_in = 3e-2 # Port diameter grain [m]
d_grain_out = 6.6e-2 # Outside diameter of grain [m]
r_grain_init = d_grain_in / 2 # Port radius [m]
mass_grain_init = L * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2) * rho_grain # Inital mass of grain [kg]
a = 0.132
n = 0.16

# Combustion mixture (predicted using CEA)
gamma =  1.18 # Specific heat of combustion mixture[-]
W_g =  23e-3 # Molecular mass of mixture
R_g = R_a / W_g 
T_0 = 2900  # Temperature of combustion mixture

# Engine
r_t = 1.55e-2/2 # Nozzle throat radius [m]
r_e = 3.1e-2/2 # Nozzle exit radius [m]
A_t = np.pi * r_t**2 # Nozzle throat area [m]
A_e = np.pi * r_e**2 # Nozzle exit area [m]
t_b = 5 # Burn time

# Initial conditions
P_a = 86e3
'''


# - - - - - - - - - - - - - - - - - - - - - Inputs - - - - - - - - - - - - - - - - - - - - - - - -

# Data directory
data_dir_single_data = 'data_dir'

true_time , Experimental_data, _, _, _, _, _ = prep_data_surrogate(data_dir_single_data, scaled=False, n_outputs=2, n_samples=1e12, start_time=None, end_time=None).compare_with_data()


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

a = 4.467266
n = 0.19012068



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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
'''

# - - - - - - - - - - - - - - - - - - - - - Inputs - - - - - - - - - - - - - - - - - - - - - - - -

# Grain 
L = 60e-3 # Length of grain [m]
d_grain_in = 20e-3 # Port diameter grain [m]
t_liner = 1.75e-3 # Wall thickness of the liner
d_grain_out = 43.5e-3 - 2 * t_liner # Outside diameter of grain [m]
r_grain_init = d_grain_in / 2 # Port radius [m]
#mass_grain_init_1 = L * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2) * rho_grain # Inital mass of grain [kg]
mass_grain_init = 97e-3 # [kg]
rho_grain =  mass_grain_init / ( L * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2))# Density of grain [kg/m3]
N_grains = 6
a = 5.13
n = 0.222

# Inhibiters
uninhibited_core = 1
uninhibited_sides = 1

# Combustion mixture (predicted using CEA)
gamma =  1.137 # Specific heat of combustion mixture[-]
W_g = 39.9e-3  # Molecular mass of mixture
R_g = R_a / W_g 
T_0 = 1600 # Temperature of combustion mixture


# Motor specifications
d_chamber = 44e-3 # Outside diamter of the chamber [m]
L_chamber = 409e-3 # Length of the chamber [m]
r_t = 11e-3/2 # Nozzle throat radius [m]
#r_t = 7e-3/2
r_e = 22e-3/2 # Nozzle exit radius [m]
#r_e = 14e-3/2
A_t = np.pi * r_t**2  # Nozzle throat area [m]
A_e = np.pi * r_e**2 # Nozzle exit area [m]
t_b = 6 # Burn time
alpha = 0.5
'''

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
drdt_lst = []
P_0_lst = []

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
def solid(t, y):
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
        
        
     
    
    # Append intermediate values
    t_lst.append(t)
    A_b_lst.append(A_b)
    V_c_lst.append(V_c)
    dpdt_lst.append(dP0dt)
    drdt_lst.append(drdt)
    P_0_lst.append(P0)
    
    dydt = [dP0dt, drdt]
    return dydt



# Initial conditions
y0 = [P_a, r_grain_init]

# Start time
start_time = time.time()

# Solve IMPORTANT TO SET TIME LONG ENOUGH
sol = solve_ivp(solid, (0, t_b), y0, method='RK45', dense_output=True)
P_0_sol = sol.y[0]
r_sol = sol.y[1]

# End time
end_time = time.time()

# Thrust, Cf, c* and realtime Isp calculation (Note: Real time Isp is negative for the negative thrust values, please ignore)
pepc = fsolve(AeAt_func, 0.001, args=(Gamma(gamma), gamma, A_e, A_t))
P_e_sol = P_0_sol*1e5 * pepc
C_f = Cf(Gamma(gamma), gamma, P_e_sol, P_0_sol*1e5, P_a*1e5, (A_e / A_t))
Thrust1 = C_f * P_0_sol*1e5 * A_t * eta_n
m_dot_tot = m_dot_id(Gamma(gamma), P_0_sol*1e5, A_t, R_g, T_0)
#U_e = Ue(gamma, R_g, T_0, P_e_sol*1e5, P_0_sol*1e5)
c_star_ideal = c_star(Gamma(gamma), R_g, T_0)



#area_force = simpson(np.abs(Thrust1), sol.t)
#area_mass = simpson(M_f_sol, sol.t) + simpson(M_ox_sol, sol.t)


Isp = (C_f * c_star_ideal) / g_0



# Create a 2x2 grid of subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

# Start and stop times plot
t_start_plot = -1
t_end_plot = t_b + 1





# Plot data in each subplot
#axs[0, 0].plot(sol.t, P_0_sol, label="Simulation data")
axs[0, 0].plot(sol.t, P_0_sol, label="Simulation data")
axs[0, 0].plot(true_time[:, 0], Experimental_data[:, 0], label='Experimental data')
axs[0, 0].set_title('Chamber pressure vs. Time')
axs[0, 0].set_xlabel('Time [s]')
#axs[0, 0].set_ylabel('Chamber pressure [Bar]')
axs[0, 0].set_ylabel('Chamber pressure [bar]')
#axs[0, 0].set_xscale('log')
axs[0, 0].set_xlim(t_start_plot, t_end_plot)
axs[0, 0].grid(which='both', linestyle='--')
axs[0, 0].legend()


axs[0, 1].plot(sol.t, Thrust1, label="Simulation data")
axs[0, 1].plot(true_time[:, 0], Experimental_data[:, 1], label='Experimental data')
axs[0, 1].set_title('Thrust vs. Time')
axs[0, 1].set_xlabel('Time [s]')
axs[0, 1].set_ylabel('Thrust [N]')
#axs[0, 1].set_xscale('log')
axs[0, 1].set_xlim(t_start_plot, t_end_plot)
axs[0, 1].grid(which='both', linestyle='--')
axs[0, 1].legend()

axs[1, 0].plot(sol.t, r_sol-r_grain_init)
axs[1, 0].set_title('Burned surface vs. Time')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].set_ylabel('Burned surface [m]')
#axs[1, 0].set_xscale('log')
axs[1, 0].set_xlim(t_start_plot, t_end_plot)
axs[1, 0].grid(which='both', linestyle='--')


axs[1, 1].plot(t_lst, A_b_lst, 'o')
axs[1, 1].set_title('Burn area vs. Time')
axs[1, 1].set_xlabel('Time [s]')
axs[1, 1].set_ylabel(f'Burn area [m$^{2}$]')
#axs[1, 1].set_xscale('log')
axs[1, 1].set_xlim(t_start_plot, t_end_plot)
axs[1, 1].grid(which='both', linestyle='--')

axs[2, 0].plot(sol.t, Isp)
axs[2, 0].set_title('Realtime Isp vs. Time')
axs[2, 0].set_xlabel('Time [s]')
axs[2, 0].set_ylabel('Isp [s]')
#axs[2, 0].set_xscale('log')
axs[2, 0].set_xlim(t_start_plot, t_end_plot)
axs[2, 0].set_ylim(0)
axs[2, 0].grid(which='both', linestyle='--')


axs[2, 1].plot(t_lst, V_c_lst, 'o')
axs[2, 1].set_title('Volume of chamber vs. Time')
axs[2, 1].set_xlabel('Time [s]')
axs[2, 1].set_ylabel(f'Volume of chamber [m$^{3}$]')
#axs[2, 1].set_xscale('log')
axs[2, 1].set_xlim(0)
axs[2, 1].set_ylim(0)
axs[2, 1].grid(which='both', linestyle='--')


# Add overall title and adjust layout
fig.suptitle('Solid rocket engine')
plt.tight_layout()

# Show the plot
plt.show()




# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:f} seconds")


'''

save_data = int(input("Store data? [1 = yes, 0 = no]"))

if save_data == 1:
    # Save to CSV
    df = pd.DataFrame({
        "time": sol.t,
        "pressure": P_0_sol,
        "thrust": Thrust1,
    })
    
    output_csv_path = "/Users/tristanhirs/Downloads/Thesis/Motor test data/Simulated_data/bates_verification_file_USU.csv"  # Specify the path where you want to save the CSV
    df.to_csv(output_csv_path, index=False)
else:
    pass

'''



'''

save_data = int(input("Store data? [1 = yes, 0 = no]"))

if save_data == 1:
    # Save to CSV
    df = pd.DataFrame({
        "time": t_lst,
        "pressure": P_0_lst,
        #"thrust": Thrust1,
        'drdt': drdt_lst,
        'dPdt': dpdt_lst,
        'A_b': A_b_lst,
        'V_c': V_c_lst
    })
    
    output_csv_path = "output_dir"  # Specify the path where you want to save the CSV
    df.to_csv(output_csv_path, index=False)
else:
    pass
'''
