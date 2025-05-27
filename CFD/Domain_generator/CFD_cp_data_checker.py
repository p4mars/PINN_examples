# Imported modules
import numpy as np
import matplotlib.pyplot as plt
from CFD_PINN_helper_functions import CFD_PINN_helpers
from scipy.interpolate import griddata
from matplotlib.path import Path


# Constants
data_dir_circle = 'data_dir_object_data'
data_dir_airfoil = 'data_dir_object_data'
data_dir_NACA = 'data_dir_object_data'
data_dir_cp = 'data_dir_cp_data'

alpha = np.deg2rad(10.0130)

plot_window, points_domain, points_object, points_input, points_exit, points_up, points_down = CFD_PINN_helpers(data_dir_NACA, object_start=(0.5,1), object_angle=-0.0169, num_object_points=100, num_boundary_points=100, num_domain_points=5000).domain_and_bc_setup(2, 2)
test = plot_window.transform_cp_data(data_dir_cp)


X, Y, Cp = test[:, 0], test[:, 1], test[:, 2]

# Split x and y in upper and lower parts
idx_split = np.argmax(X)

X_upper, X_lower = np.split(X, [idx_split])
Y_upper, Y_lower = np.split(Y, [idx_split])
Cp_upper, Cp_lower = np.split(Cp, [idx_split])
c = 1

# Step 1: Define a common set of X coordinates (based on the upper surface, for example)
X_common = X_upper  # You can also use X_lower, depending on how you want to define the common set

# Step 2: Interpolate the lower surface Cp and Y to match the common X coordinates
Y_lower = np.interp(X_common, X_lower, Y_lower)
Cp_lower = np.interp(X_common, X_lower, Cp_lower)


# Calulcate Cl and Cd
# Compute dx and dy (finite difference)
dx = np.diff(X)
dy = np.diff(Y)

# Midpoint Cp values for integration
Cp_mid = 0.5 * (Cp[:-1] + Cp[1:])

# Integrate Cl and Cd
Cy = np.sum(Cp_mid * dy)
Cx = np.sum(Cp_mid * dx)

Cl = Cy * np.cos(alpha) - Cx * np.sin(alpha)
Cd = Cy * np.sin(alpha) + Cx * np.cos(alpha)

print(f"Lift Coefficient (Cl): {Cl:.4f}")
print(f"Drag Coefficient (Cd): {Cd:.4f}")

#plt.plot(X, Cp)




#inputs_test, output_test = plot_window.inputs_generator_cfd_pinn(data_dir_cp, 1, 0.35496478698597694, 1)



# Plot figure as test
plt.figure(figsize=(12, 6))
plt.scatter(points_domain[:,0].detach().numpy(), points_domain[:,1].detach().numpy(), label="Domain collocation points")
plt.scatter(points_object[:, 0].detach().numpy(), points_object[:, 1].detach().numpy(), label="Object boundary collocation points")
sc = plt.scatter(X, Y, c=Cp, cmap='turbo', edgecolors='k', label="Cp Experimental data points")
plt.colorbar(sc, label='Cp')  # Add color bar to indicate Cp values
plt.scatter(points_input[:, 0].detach().numpy(), points_input[:, 1].detach().numpy(), label="Input boundary collocation points")
plt.scatter(points_exit[:, 0].detach().numpy(), points_exit[:, 1].detach().numpy(), label="Exit boundary collocation points")
plt.scatter(points_up[:, 0].detach().numpy(), points_up[:, 1].detach().numpy(), label="Top wall collocation points")
plt.scatter(points_down[:, 0].detach().numpy(), points_down[:, 1].detach().numpy(), label="Bottom wall collocation points")
plt.title('Scatter Plot of Cp on Airfoil Surface')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()


'''
# Create scatter plot with color based on Cp values
plt.figure(figsize=(8, 6))
sc = plt.scatter(X, Y, c=Cp, cmap='turbo', edgecolors='k')
plt.colorbar(sc, label='Cp')  # Add color bar to indicate Cp values
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Cp on Airfoil Surface')
plt.show()
'''
