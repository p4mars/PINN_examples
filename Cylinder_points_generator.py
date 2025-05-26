# Imported modules
import numpy as np
import matplotlib.pyplot as plt


num_boundary_points_cylinder = 50
x_c, y_c = (1, 1)
R = 0.05/2

# Setup cylinder boundary
theta = np.linspace(0, 2*np.pi, num_boundary_points_cylinder)
x_bc_cylinder = x_c + R * np.cos(theta)
y_bc_cylinder = y_c + R * np.sin(theta)
boundary_cylinder_points = np.stack([x_bc_cylinder, y_bc_cylinder], axis=-1)

plt.plot(x_bc_cylinder, y_bc_cylinder, 'o')
plt.show()


# Save to a text file
np.savetxt("cylinder_coordinates_D_0_05.txt", boundary_cylinder_points, delimiter="\t", header="#x_coordinates \t y_coordinates", comments="", fmt="%.5f")