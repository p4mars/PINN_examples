# Imported modules
from NN_tools_thesis import NN, gradient
import numpy as np
import torch
import torch.nn as nn
import time
import os
from CFD_PINN_helper_functions import CFD_PINN_helpers

# Constants
g_0 = 9.80665 # Gravitational acceleration [m/s2]
R_a = 8.3144 # Gas constant
P_a = 101325/1e6 # Standard pressure at sea level [pa]
P_a = 0.2
rho = 1.1325 # Stnadard density of air [kg/m3]
rho = 1.5

# Inputs
nu = 0.02                             # Kinematic viscosity [m/s2]
u_inf = 1                              # flow speed [m/s]

# Set domain size
x_domain_size = 4
y_domain_size = 2


# Total pressure
p_inlet = P_a #+ ((0.5 * rho * u_inf**2)/1e5)

# Data directories
data_dir_circle = 'data_dir_object'
data_dir_circle_small = 'data_dir_object'
data_dir_airfoil = 'data_dir_object'
data_dir_cylinder_paper = 'data_dir_object'
data_dir_cylinder_scaled = 'data_dir_object'


# ------------------------------ Start of construction of PINN ---------------------------------

# ----------------------------- Setup of the domain ------------------------------------------


plot_check, points_domain, points_cylinder, points_input, points_exit, points_up, points_down = CFD_PINN_helpers(data_dir_cylinder_paper, object_start=(0.95, 1), object_angle=0, num_object_points=300, num_boundary_points=200, num_domain_points=5000).domain_and_bc_setup(x_domain_size, y_domain_size)


# Plot to check domain
plot_check.plot_domain()



# ----------------------------- Setup of physics losses and boundary losses ----------------------------

# Define physics loss (Navier stokes equation fro 2D, steady state incompressible flow situation)
def Navier_stokes_loss(NN_output: torch.nn.Module, used_inputs):   
    # Get x and y from domain points 
    x_points = points_domain[:,0].unsqueeze(-1)
    y_points = points_domain[:,1].unsqueeze(-1)
    inputs = torch.cat([x_points, y_points], dim=1)  # Combine x and y
    
    # # Obtain predictions from the network within domain
    u, v, p = NN_output(inputs).split(1, dim=1)
    
    # Obtain u_x, u_xx, u_y, u_yy, v_x, v_xx, v_y, v_yy, p_x, p_y
    du_dx = gradient(u, x_points)
    du_dxx = gradient(du_dx, x_points)
    du_dy = gradient(u, y_points)
    du_dyy = gradient(du_dy, y_points)
    dv_dx = gradient(v, x_points)
    dv_dxx = gradient(dv_dx, x_points)
    dv_dy = gradient(v, y_points)
    dv_dyy = gradient(dv_dy, y_points)
    dp_dx = gradient(p, x_points)
    dp_dy = gradient(p, y_points)

    # Continuity equation
    CE = du_dx + dv_dy
    
    # Momentum equations (in x and y directions)
    Mx = u * du_dx + v * du_dy + 1/rho * dp_dx - nu * (du_dxx + du_dyy)
    My = u * dv_dx + v * dv_dy + 1/rho * dp_dy - nu * (dv_dxx + dv_dyy)
    
    # Enforce pressure loss as ideal gas
    
    
    #print(torch.mean(CE**2 + Mx**2 + My**2))
    #print('\n')
    #print(torch.min(u), torch.min(v), torch.min(p), torch.max(p))
    
    return ((CE**2).mean() + (Mx**2).mean() + (My**2).mean())


# Define inlet boundary loss
def BC_inlet_loss(NN_output: torch.nn.Module, used_inputs):
    # Get x and y from inlet points 
    x_points = points_input[:,0].unsqueeze(-1)
    y_points = points_input[:,1].unsqueeze(-1)
    inputs = torch.cat([x_points, y_points], dim=1)  # Combine x and y
    
    # # Obtain predictions from the network within domain
    u, v, p = NN_output(inputs).split(1, dim=1)
    
    #print(torch.mean((u-u_inf)**2 + v**2))
    
    # Flow input condition
    #u_input = u_inf * (y_points + 1) * (y_points - 1)
    
    return (((u-u_inf)**2).mean() + (v**2).mean() + ((p-p_inlet)**2).mean())
    #return (((u-u_inf)**2).mean()) 


# Define outlet boundary loss
def BC_outlet_loss(NN_output: torch.nn.Module, used_inputs):
    # Get x and y from inlet points 
    x_points = points_exit[:,0].unsqueeze(-1)
    y_points = points_exit[:,1].unsqueeze(-1)
    inputs = torch.cat([x_points, y_points], dim=1)  # Combine x and y
    
    # # Obtain predictions from the network within domain
    u, v, p = NN_output(inputs).split(1, dim=1)

    
    # Obtain u_x and v_x
    du_dx = gradient(u, x_points)
    dv_dx = gradient(v, x_points)
    dp_dx = gradient(p, x_points)
    
   #print(torch.mean(du_dx**2 + dv_dx**2 + p**2))
    
    return ((du_dx**2).mean() + (dv_dx**2).mean() + (dp_dx**2).mean())
    #return ((p-P_a)**2).mean()
    #return(p**2).mean()

# Define upper wall boundary loss
def BC_upper_wall_loss(NN_output: torch.nn.Module, used_inputs):
    # Get x and y from inlet points 
    x_points = points_up[:,0].unsqueeze(-1)
    y_points = points_up[:,1].unsqueeze(-1)
    inputs = torch.cat([x_points, y_points], dim=1)  # Combine x and y
   
    # # Obtain predictions from the network within domain
    u, v, p = NN_output(inputs).split(1, dim=1)
    
    return (((u-u_inf)**2).mean() + (v**2).mean())
    #return ((u**2).mean() + (v**2).mean())
    #return (((u-u_inf)**2).mean())

# Define lower wall boundary loss
def BC_lower_wall_loss(NN_output: torch.nn.Module, used_inputs):
    # Get x and y from inlet points 
    x_points = points_down[:,0].unsqueeze(-1)
    y_points = points_down[:,1].unsqueeze(-1)
    inputs = torch.cat([x_points, y_points], dim=1)  # Combine x and y
    
    # # Obtain predictions from the network within domain
    u, v, p = NN_output(inputs).split(1, dim=1)
    
    return (((u-u_inf)**2).mean() + (v**2).mean())
    #return ((u**2).mean() + (v**2).mean())
    #return (((u-u_inf)**2).mean())


# Define cylinder boundary loss
def BC_object_loss(NN_output: torch.nn.Module, used_inputs):
    # Get x and y from inlet points 
    x_points = points_cylinder[:,0].unsqueeze(-1)
    y_points = points_cylinder[:,1].unsqueeze(-1)
    inputs = torch.cat([x_points, y_points], dim=1)  # Combine x and y
    
    # # Obtain predictions from the network within domain
    u, v, p = NN_output(inputs).split(1, dim=1)
    
    
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
PINN_CFD = NN(2, 3, 64, 2, epochs=20000, lr=1e-2, loss_terms=loss_functions, loss_init=None, activation_fn=nn.Tanh, init_method=nn.init.xavier_uniform_)

# Start time
start_time = time.time()

# Train network
# Initialize labels for training with dummy values for u, v, p
num_points = points_domain.shape[0]
labels = torch.zeros((num_points, 3))  # [u, v, p]

#PINN_CFD.fit(points_domain.detach().numpy(), labels.numpy())

# End time
end_time = time.time()


# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:f} seconds")


'''
# Generate random test points
num_test_points = 20000
x_test = torch.rand(num_test_points) * x_domain_size  # Scale to domain (0 ≤ x ≤ 20)
y_test = torch.rand(num_test_points) * y_domain_size  # Scale to domain (0 ≤ y ≤ 10)
test_points = torch.stack((x_test, y_test), dim=1)  # Shape: (num_test_points, 2)
'''

# Predict velocity and pressure
points_for_predictions = np.concatenate((points_domain.detach().numpy(), points_cylinder.detach().numpy()))
predictions = PINN_CFD.predict(points_for_predictions)
u, v, p = predictions[:, 0], predictions[:, 1], predictions[:, 2]

# Total velocity
phi = np.sqrt(u**2 + v**2)


# U heat map
plot_check.heat_map_generator(torch.tensor(points_for_predictions), u, grid_size=len(predictions), labels=["U velocity [m/s]", "U velocity (->) heat map", "X", "Y"])

# V heat map
plot_check.heat_map_generator(torch.tensor(points_for_predictions), v, grid_size=len(predictions), labels=["V velocity [m/s]", "V velocity (^) heat map", "X", "Y"])

# Pressure heat map
plot_check.heat_map_generator(torch.tensor(points_for_predictions), p, grid_size=len(predictions), labels=["Pressure [Mpa]", "Pressure heat map", "X", "Y"])

# Total velocity heat map
plot_check.heat_map_generator(torch.tensor(points_for_predictions), phi, grid_size=len(predictions), labels=["Total velocity [m/s]", "Total velocity heat map", "X", "Y"])


# Calculate lift and drag coefficient
predictions_lift = PINN_CFD.predict(points_cylinder.detach().numpy())
p_lift = predictions_lift[:, 2]

plot_check.lift_calc(p_lift, P_a, rho, u_inf)



# Save entire trained model into a certain folder

#filt_data_filename = os.path.join(data_dir_trained_PINN, "CFD_PINN_trained.pth")
#torch.save(PINN_CFD.state_dict(), filt_data_filename)




