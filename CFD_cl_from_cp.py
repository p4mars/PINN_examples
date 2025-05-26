# Imported modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps # Simpson's rule for integration

# Constants
g_0 = 9.80665 # Gravitational acceleration [m/s2]
R_a = 8.3144 # Gas constant
P_a = 101325/1e5 # Standard pressure at sea level [pa]
rho = 1.1325 # Stnadard density of air [kg/m3]
nu = 0.01 # Kinematic viscosity [m/s2]

# Inputs
alpha = 0.0169
M_inf = 0.3 # Mach number [-]
Re = 6e6 # [-]
c = 347


# Total pressure
u_inf = 1
p_tot = P_a + (0.5 * rho * u_inf**2)


# Import Cp data
data_dir_CFD = '/Users/tristanhirs/Downloads/Thesis/CFD_files/Ladson/Cp_data_CFD/cp_data_cfl3d_Re_6e6_alpha_15.txt'
data_dir_exp = '/Users/tristanhirs/Downloads/Thesis/CFD_files/Ladson/Cp_data/Cp_alpha_15_0299_Re6.txt'
data_dir_PINN = '/Users/tristanhirs/Downloads/Thesis/CFD_files/PINN_example_data/Cp_alpha_0.0169_Re6_test.csv'

cp_data = np.genfromtxt(data_dir_exp)
cp_data_CFD = np.genfromtxt(data_dir_CFD)
pinn_data = np.genfromtxt(data_dir_PINN, delimiter=',')

x = cp_data[:, 0]
cp = cp_data[:, 1]

x_cfd = cp_data_CFD[:, 0]
cp_cfd = cp_data_CFD[:, 1]

x_pinn = pinn_data[1:, 0] - 0.5
p_pinn = pinn_data[1:, 4]

cp_pinn = (p_pinn - 1.01325) / (0.5 * 1 * 1**2)

# Find the indices where the value 0 occurs
split_indices = np.where(x == 0)[0] + 1

split_cfd = np.where(cp_cfd == np.min(cp_cfd))[0] + 1

split_pinn = np.array([np.argmax(x_pinn) + 1])

# Now split the array at each occurrence of 0
x_upper, x_lower = np.split(x, split_indices)
cp_upper, cp_lower = np.split(cp, split_indices)

x_cfd_upper, x_cfd_lower = np.split(x_cfd, split_cfd)
cp_cfd_upper, cp_cfd_lower = np.split(cp_cfd, split_cfd)

x_pinn_upper, x_pinn_lower = np.split(x_pinn, split_pinn)
cp_pinn_upper, cp_pinn_lower = np.split(cp_pinn, split_pinn)



# Interpolate to match sizes (assuming x_upper has more points)
def Cl_calc(x_upper, x_lower, cp_upper, cp_lower):
    if len(x_upper) > len(x_lower):
        interp_cp_lower = interp1d(x_lower, cp_lower, kind='linear', fill_value="extrapolate")
        cp_lower_resampled = interp_cp_lower(x_upper)  # Interpolate to match x_upper
        x_common, cp_common_upper, cp_common_lower = x_upper, cp_upper, cp_lower_resampled
    else:
        interp_cp_upper = interp1d(x_upper, cp_upper, kind='linear', fill_value="extrapolate")
        cp_upper_resampled = interp_cp_upper(x_lower)  # Interpolate to match x_lower
        x_common, cp_common_upper, cp_common_lower = x_lower, cp_upper_resampled, cp_lower
    
    # Compute differential pressure distribution
    delta_cp = cp_common_lower - cp_common_upper
    
    # Integrate to find the lift coefficient
    C_L = simps(delta_cp, x_common)  # Numerical integration
    
    return C_L, print("Lift Coefficient (C_L):", C_L)

Cl_calc(x_upper, x_lower, cp_upper, cp_lower)
Cl_calc(x_cfd_upper, x_cfd_lower, cp_cfd_upper, cp_cfd_lower)
#Cl_calc(x_pinn_upper, x_pinn_lower, cp_pinn_upper, cp_pinn_lower)

plt.figure()
plt.plot(x, cp, 'o')
plt.plot(x_cfd, cp_cfd)
#plt.plot(x_pinn, cp_pinn)
plt.plot(x_pinn_upper, cp_pinn_upper, 'o')
plt.plot(x_pinn_lower, cp_pinn_lower)