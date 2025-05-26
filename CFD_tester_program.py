# Imported modules
import numpy as np
import matplotlib.pyplot as plt
import torch
from CFD_PINN_helper_functions import CFD_PINN_helpers

# Constants
data_dir_circle = '/Users/tristanhirs/Downloads/Thesis/CFD_files/Objects/cylinder_coordinates.txt'
data_dir_airfoil = '/Users/tristanhirs/Downloads/Thesis/CFD_files/Objects/seligdatfile.txt'
data_dir_NACA = '/Users/tristanhirs/Downloads/Thesis/CFD_files/Objects/NACA_0012.txt'


plot_window, points_domain, points_object, points_input, points_exit, points_up, points_down = CFD_PINN_helpers(data_dir_NACA, object_start=(0.5,1), object_angle=-5, num_object_points=100, num_boundary_points=100, num_domain_points=5000).domain_and_bc_setup(2, 2)




# Plot figure as test
plt.figure(figsize=(12, 6))
plt.scatter(points_domain[:,0].detach().numpy(), points_domain[:,1].detach().numpy())
plt.scatter(points_object[:, 0].detach().numpy(), points_object[:, 1].detach().numpy())
plt.scatter(points_input[:, 0].detach().numpy(), points_input[:, 1].detach().numpy())
plt.scatter(points_exit[:, 0].detach().numpy(), points_exit[:, 1].detach().numpy())
plt.scatter(points_up[:, 0].detach().numpy(), points_up[:, 1].detach().numpy())
plt.scatter(points_down[:, 0].detach().numpy(), points_down[:, 1].detach().numpy())
plt.show()



#plot_window.plot_domain()


