# Imported modules
from NN_tools_thesis import NN, gradient
import numpy as np
import torch
import torch.nn as nn
import time
import os
from CFD_PINN_helper_functions import CFD_PINN_helpers
import matplotlib.pyplot as plt
import pandas as pd

# Constants
g_0 = 9.80665 # Gravitational acceleration [m/s2]
R_a = 8.3144 # Gas constant
P_a = 101325/1e5 # Standard pressure at sea level [pa]
P_a = 1
rho = 1.1325 # Stnadard density of air [kg/m3]
nu = 1.735e-5 # Kinematic viscosity [m/s2]


# Inputs
alpha = 0.0169
M_inf = 0.3 # Mach number [-]
Re = 6e6 # [-]
c = 347

# Set domain size
x_domain_size = 40
y_domain_size = 10


# Total pressure
u_inf = 104.1
p_tot = P_a + (0.5 * rho * u_inf**2)


# Data directories
data_dir_NACA = '/Users/tristanhirs/Downloads/Thesis/CFD_files/Objects/NACA_0012.txt'
data_dir_cp = '/Users/tristanhirs/Downloads/Thesis/CFD_files/Ladson/Cp_data/Cp_alpha_0169_Re6.txt'


# ------------------------------ Start of construction of PINN ---------------------------------

# ----------------------------- Setup of the domain ------------------------------------------


plot_check, points_domain, points_object, points_input, points_exit, points_up, points_down = CFD_PINN_helpers(data_dir_NACA, object_start=(10,5), object_angle=-alpha, num_object_points=300, num_boundary_points=200, num_domain_points=5000).domain_and_bc_setup(x_domain_size, y_domain_size)
inputs_test, outputs_test = plot_check.inputs_generator_cfd_pinn(data_dir_cp, rho, u_inf, P_a)

# Plot to check domain
#plot_check.plot_domain()



# ----------------------------- Setup of physics losses and boundary losses ----------------------------

# Define physics loss (Navier stokes equation fro 2D, steady state incompressible flow situation)
def Navier_stokes_loss(NN_output: torch.nn.Module, used_inputs):   
    # # Obtain predictions from the network within domain
    u, v, p = NN_output(points_domain).split(1, dim=1)
    
    # Obtain u_x, u_xx, u_y, u_yy, v_x, v_xx, v_y, v_yy, p_x, p_y
    du_dx = gradient(u, points_domain)[:, 0]
    du_dxx = gradient(du_dx, points_domain)[:, 0]
    du_dy = gradient(u, points_domain)[:, 1]
    du_dyy = gradient(du_dy, points_domain)[:, 1]
    dv_dx = gradient(v, points_domain)[:, 0]
    dv_dxx = gradient(dv_dx, points_domain)[:, 0]
    dv_dy = gradient(v, points_domain)[:, 1]
    dv_dyy = gradient(dv_dy, points_domain)[:, 1]
    dp_dx = gradient(p, points_domain)[:, 0]
    dp_dy = gradient(p, points_domain)[:, 1]

    # Continuity equation
    CE = du_dx + dv_dy
    
    # Momentum equations (in x and y directions)
    Mx = u * du_dx + v * du_dy + 1/rho * dp_dx - nu * (du_dxx + du_dyy)
    My = u * dv_dx + v * dv_dy + 1/rho * dp_dy - nu * (dv_dxx + dv_dyy)
    
    # Enforce pressure loss as ideal gas
    
    return ((CE**2).mean() + (Mx**2).mean() + (My**2).mean())


# Define inlet boundary loss
def BC_inlet_loss(NN_output: torch.nn.Module, used_inputs):
    
    # Obtain predictions from the network within domain
    u, v, p = NN_output(points_input).split(1, dim=1)
    
    return (((u-u_inf)**2).mean() + (v**2).mean() + ((p-p_tot)**2).mean())
    #return (((u-u_inf)**2).mean() + (v**2).mean())


# Define outlet boundary loss
def BC_outlet_loss(NN_output: torch.nn.Module, used_inputs):
    
    # Obtain predictions from the network within domain
    u, v, p = NN_output(points_exit).split(1, dim=1)

    # Obtain u_x and v_x
    du_dx = gradient(u, points_exit)[:, 0]
    dv_dx = gradient(v, points_exit)[:, 0]
    dp_dx = gradient(p, points_exit)[:, 0]
    
    return ((du_dx**2).mean() + (dv_dx**2).mean() + (dp_dx**2).mean())
    #return (p**2).mean()

# Define upper wall boundary loss
def BC_upper_wall_loss(NN_output: torch.nn.Module, used_inputs):

   
    # Obtain predictions from the network within domain
    u, v, p = NN_output(points_down).split(1, dim=1)
    
    return (((u-u_inf)**2).mean() + (v**2).mean())
    #return ((u**2).mean() + (v**2).mean())

# Define lower wall boundary loss
def BC_lower_wall_loss(NN_output: torch.nn.Module, used_inputs):

    
    # Obtain predictions from the network within domain
    u, v, p = NN_output(points_up).split(1, dim=1)
    
    return (((u-u_inf)**2).mean() + (v**2).mean())
    #return ((u**2).mean() + (v**2).mean())


# Define object boundary loss
def BC_object_loss(NN_output: torch.nn.Module, used_inputs):
    # Get x and y from inlet points 
    
    # Obtain predictions from the network within domain
    u, v, p = NN_output(points_object).split(1, dim=1)
    
    return ((u**2).mean() + (v**2).mean())

# Initialize loss functions
loss_functions = {
    Navier_stokes_loss: 1,
    BC_inlet_loss: 1,
    BC_outlet_loss: 1, 
    BC_upper_wall_loss: 1,
    BC_lower_wall_loss: 1,
    BC_object_loss: 1
    }


# Create network
PINN_CFD = NN(2, 3, 64, 2, epochs=20000, lr=1e-2, loss_terms=loss_functions, activation_fn=nn.Tanh, init_method=nn.init.xavier_normal_)

# Start time
start_time = time.time()

# Train network
# Initialize labels for training with dummy values for u, v, p

# Train network
# Initialize labels for training with dummy values for u, v, p
num_points = points_domain.shape[0]
labels = torch.zeros((num_points, 3))  # [u, v, p]

#PINN_CFD.fit(points_domain.detach().numpy(), labels.numpy())

PINN_CFD.fit(inputs_test, outputs_test)

# End time
end_time = time.time()


# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:f} seconds")

# Predict velocity and pressure
points_for_predictions = np.concatenate((points_domain.detach().numpy(), points_object.detach().numpy()))
predictions = PINN_CFD.predict(points_for_predictions)
u, v, p = predictions[:, 0], predictions[:, 1], predictions[:, 2]

# Total velocity
phi = np.sqrt(u**2 + v**2)


# U heat map
plot_check.heat_map_generator(torch.tensor(points_for_predictions), u, grid_size=len(predictions), labels=["U velocity [m/s]", "U velocity (->) heat map", "X", "Y"])

# V heat map
plot_check.heat_map_generator(torch.tensor(points_for_predictions), v, grid_size=len(predictions), labels=["V velocity [m/s]", "V velocity (^) heat map", "X", "Y"])

# Pressure heat map
plot_check.heat_map_generator(torch.tensor(points_for_predictions), p, grid_size=len(predictions), labels=["Pressure [Bar]", "Pressure heat map", "X", "Y"])

# Total velocity heat map
plot_check.heat_map_generator(torch.tensor(points_for_predictions), phi, grid_size=len(predictions), labels=["Total velocity [m/s]", "Total velocity heat map", "X", "Y"])

# Calculate lift and drag coefficient
predictions_lift = PINN_CFD.predict(points_object.detach().numpy())
p_lift = predictions_lift[:, 2]

plot_check.lift_calc(p_lift, P_a, rho, u_inf)


# Save entire trained model into a certain folder

#filt_data_filename = os.path.join(data_dir_trained_PINN, "CFD_PINN_trained.pth")
#torch.save(PINN_CFD.state_dict(), filt_data_filename)


'''
save_data = int(input("Store data? [1 = yes, 0 = no]"))

if save_data == 1:
    # Save to CSV
    df = pd.DataFrame({
        "x": points_object.detach().numpy()[:, 0],
        "y": points_object.detach().numpy()[:, 1],
        "u": u,
        "v": v,
        "p": p
    })
    
    output_csv_path = "/Users/tristanhirs/Downloads/Thesis/CFD_files/PINN_example_data/Cp_alpha_0.0169_Re6_test.csv"  # Specify the path where you want to save the CSV
    df.to_csv(output_csv_path, index=False)
else:
    pass
'''



