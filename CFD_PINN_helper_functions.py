# Imported modules
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata
from scipy.integrate import simps # Simpson's rule for integration
from matplotlib.path import Path

np.random.seed(42)

class CFD_PINN_helpers():
    
    def __init__(self, object_data_file, num_domain_points=1, num_boundary_points=1, num_object_points=1, object_start=(0,0), object_angle=0):
        
        self.object_data_file = object_data_file
        self.num_domain_points = num_domain_points
        self.num_boundary_points = num_boundary_points
        self.num_object_points = num_object_points
        self.object_data = self.load_object()
        self.object_start = object_start
        
        # Boundaries
        self.boundary_exit_points = None
        self.boundary_input_points = None
        self.boundary_object_points = None
        self.boundary_wall_points_down = None
        self.boundary_wall_points_up = None
        self.domain_points = None
        self.x_domain_size = None
        self.y_domain_size = None
        self.object_angle = np.deg2rad(object_angle)
        self.reference_area = None
        
    
    def load_object(self):
        # Load object file
        if self.object_data_file.endswith('.txt') or self.object_data_file.endswith('.dat'):
            all_data = np.loadtxt(self.object_data_file)
        elif self.object_data_file.endswith('.csv'):
            all_data = np.load(self.object_data_file)
        else:
            print("Error, file type not supported")
            all_data = None
        
        return all_data
    
    def rotate_object(self, object_points):
        
        theta = self.object_angle  
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]]) 

        return np.dot(object_points, R.T)  
    
    
    def convert_to_start(self):
        
        # Check for placement in domain
        half_value_x = self.object_data[:, 0].min()
        half_value_y = np.mean([self.object_data[:, 1].max(), self.object_data[:, 1].min()])
        
        # Convert to x=0 and half point on y-axis
        x_data_start = self.object_data[:, 0] - half_value_x 
        y_data_start = self.object_data[:, 1] - half_value_y
        
        # Centered object
        centered_object = np.column_stack((x_data_start, y_data_start))
        
        return centered_object
        
        
            
    def split_object(self):
        
        # Sort all the data
        sorted_data = self.convert_to_start()[np.argsort(self.convert_to_start()[:, 0])]
        
        
        # Split in upper and lower parts
        upper_part = np.unique(sorted_data[sorted_data[:, 1] >= 0], axis=0)
        lower_part = np.unique(sorted_data[sorted_data[:, 1] <= 0], axis=0)
        
        return upper_part, lower_part
    
    def resampled_object(self):
        # Obtain upper and lower parts
        upper_part, lower_part = self.split_object()

        # Resample both surfaces
        upper_part_resampled = self.interpolate_surface(upper_part, self.num_object_points)
        lower_part_resampled = self.interpolate_surface(lower_part, self.num_object_points)

        # Merge back into a full airfoil
        full_object = np.vstack((upper_part_resampled, lower_part_resampled[::-1]))  # Ensure correct order
        
        # Calculate reference area
        self.reference_area = full_object[:,0].max() - full_object[:, 0].min()
        
        # Rotate the object
        rotated_object = self.rotate_object(full_object)
        
        # Convert to desired location
        full_object = rotated_object + [self.object_start]
        
        return full_object
        
    def interpolate_surface(self, data, num_points):
            x, y = data[:, 0], data[:, 1]
            interp_func = interp1d(x, y, kind="linear", fill_value="extrapolate", assume_sorted=True)
            x_new = np.linspace(x.min(), x.max(), num_points)
            y_new = interp_func(x_new)
            return np.column_stack((x_new, y_new))
        
    def domain_and_bc_setup(self, X_domain, Y_domain):
        # Create the domain points
        self.x_domain_size = X_domain
        self.y_domain_size = Y_domain
        x = np.random.uniform(0, self.x_domain_size, self.num_domain_points)
        y = np.random.uniform(0, self.y_domain_size, self.num_domain_points)
        domain_points = np.stack([x, y], axis=-1)
        
        # Setup object boundary
        self.boundary_object_points = self.resampled_object()
        
        
        # Remove domain points from inside of object
        polygon = Path(self.boundary_object_points)
        self.domain_points = domain_points[~polygon.contains_points(domain_points)]
        
        
        #self.domain_points = self.generate_domain_points_around_object()
    
        # create boundary points
        y_wall_up = np.repeat(0, self.num_boundary_points)
        y_wall_down = np.repeat(self.y_domain_size, self.num_boundary_points)
        x_wall = np.random.uniform(0, self.x_domain_size, self.num_boundary_points)
        x_input = np.repeat(0, self.num_boundary_points)
        x_output = np.repeat(self.x_domain_size, self.num_boundary_points)
        y_in_out = np.random.uniform(0, self.y_domain_size, self.num_boundary_points)
    
        self.boundary_wall_points_up = np.stack([x_wall, y_wall_up], axis=-1)
        self.boundary_wall_points_down = np.stack([x_wall, y_wall_down], axis=-1)
        self.boundary_input_points = np.stack([x_input, y_in_out], axis=-1)
        self.boundary_exit_points = np.stack([x_output, y_in_out], axis=-1)
        
        # Add BC points to domain
        #domain_points_total = np.concatenate((domain_points, boundary_wall_points_up, boundary_wall_points_down, boundary_input_points, boundary_object_points), axis=0)

        
        return self, torch.tensor(self.domain_points, dtype=torch.float32, requires_grad=True) , torch.tensor(self.boundary_object_points, dtype=torch.float32, requires_grad=True), torch.tensor(self.boundary_input_points, dtype=torch.float32, requires_grad=True), torch.tensor(self.boundary_exit_points, dtype=torch.float32, requires_grad=True), torch.tensor(self.boundary_wall_points_up, dtype=torch.float32, requires_grad=True), torch.tensor(self.boundary_wall_points_down, dtype=torch.float32, requires_grad=True)
    
    def plot_domain(self):
        plt.figure(figsize=(12, 6))
        plt.scatter(self.domain_points[:, 0], self.domain_points[:, 1])
        plt.scatter(self.boundary_object_points[:, 0], self.boundary_object_points[:, 1])
        plt.scatter(self.boundary_input_points[:, 0], self.boundary_input_points[:, 1])
        plt.scatter(self.boundary_wall_points_up[:, 0], self.boundary_wall_points_up[:, 1])
        plt.scatter(self.boundary_wall_points_down[:, 0], self.boundary_wall_points_down[:, 1])
        plt.scatter(self.boundary_exit_points[:, 0], self.boundary_exit_points[:, 1])
        plt.show()
        
        return self
    
    def heat_map_generator(self, inputs, prediction, grid_size, labels=[]):
        # Generate a grid of points in the domain
        
    
        # Mask for cylinder region
        object_mask = Path(self.boundary_object_points)
    
        # Define structured grid for visualization
        grid_size = 500
        x_grid = np.linspace(0, self.x_domain_size, grid_size)
        y_grid = np.linspace(0, self.y_domain_size, grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Interpolate scattered (x, y, p) onto the structured grid
        quantity_grid = griddata(inputs.detach().numpy(), prediction, (xx, yy), method='cubic')
        
        # Remove object from plot
        object_mask = Path(self.boundary_object_points)
        mask = object_mask.contains_points(np.vstack([xx.ravel(), yy.ravel()]).T).reshape(xx.shape)
        quantity_grid[mask] = np.nan  # Set object region to NaN

        # Contour plot
        plt.figure(figsize=(12, 6))
        plt.contourf(xx, yy, quantity_grid, levels=100, cmap="turbo")
        plt.colorbar(label=labels[0])
        plt.title(labels[1])
        plt.xlabel(labels[2])
        plt.ylabel(labels[3])
        plt.show()

    
        return self
    
    def calc_velocity_gradient(self, boundary_points, u, v):
        # Number of boundary points
        N = len(boundary_points) - 1
        
        # Initialize velocity_gradients
        velocity_gradients = np.zeros((N, 2, 2))
    
        # Compute finite differences using central differencing
        
        for i in range(1, N-1):
            dx = boundary_points[i+1, 0] - boundary_points[i-1, 0]  # x difference
            dy = boundary_points[i+1, 1] - boundary_points[i-1, 1]  # y difference
    
            du_dx = (u[i+1] - u[i-1]) / dx
            du_dy = (u[i+1] - u[i-1]) / dy
            dv_dx = (v[i+1] - v[i-1]) / dx
            dv_dy = (v[i+1] - v[i-1]) / dy
    
            # Store
            velocity_gradients[i] = np.array([[du_dx, du_dy], [dv_dx, dv_dy]])
    
        # Use forward and backward differences for first and last points
        velocity_gradients[0] = velocity_gradients[1]  # Approximate first point
        velocity_gradients[-1] = velocity_gradients[-2]  # Approximate last point

        return velocity_gradients
    
    
    def lift_drag_calc(self, model, nu, rho, u_inf):
        
        # Get predictions from network
        predictions = model.predict(self.boundary_object_points)
        u, v, p = predictions[:, 0], predictions[:, 1], predictions[:, 2] * 1e5
        
        # Compute velocity gradients
        velocity_gradients = self.calc_velocity_gradient(self.boundary_object_points, u, v)
        
        # Get small surface elements
        dx = np.diff(self.boundary_object_points[:, 0])
        dy = np.diff(self.boundary_object_points[:, 1])
        ds = np.sqrt(dx**2 + dy**2)
        
        
        # Get normal vectors
        nx = dy / ds
        ny = -dx / ds
        
        
        # Extract velocity gradient components
        du_dx = velocity_gradients[:, 0, 0]
        du_dy = velocity_gradients[:, 0, 1]
        dv_dx = velocity_gradients[:, 1, 0]
        dv_dy = velocity_gradients[:, 1, 1]
    
        # Compute shear stress
        tau_x = nu * (2 * du_dx * nx + (du_dy + dv_dx) * ny)
        tau_y = nu * (2 * dv_dy * ny + (du_dy + dv_dx) * nx)
    
        # Compute pressure forces
        Fp_x = -p[:-1] * nx * ds
        Fp_y = -p[:-1] * ny * ds
    
        # Compute shear forces
        Fv_x = tau_x * ds
        Fv_y = tau_y * ds
    
        # Total forces
        F_D = np.sum(Fp_x + Fv_x)  # Drag force
        F_L = np.sum(Fp_y + Fv_y)  # Lift force
    
        # Compute coefficients
        dynamic_pressure = 0.5 * rho * u_inf**2
        C_D = F_D / (dynamic_pressure * self.reference_area)
        C_L = F_L / (dynamic_pressure * self.reference_area)
        
        # Print all values
        print(f"Lift force: {F_L:f} [N]")
        print(f"Drag force: {F_D:f} [N]")
        print(f"Lift coefficient: {C_L:f} [-]")
        print(f"Drag coefficient: {C_D:f} [-]")
        
        return self
    
    def lift_calc(self, p, p_inf, rho, u_inf):
        # Compute dynmaic pressure
        dynamic_pressure = 0.5 * rho * u_inf**2
        
        # Compute pressure coefficients
        C_p = (p - p_inf) / dynamic_pressure
        
        # Split airfoil into upper and lower parts, as well as Cp
        split_idx = np.where(C_p == np.min(C_p))[0] + 1
        split_idx = np.array([np.argmax(self.boundary_object_points[:, 0]) + 1])
        
        x_upper, x_lower = np.split(self.boundary_object_points[:, 0], split_idx)
        cp_upper, cp_lower = np.split(C_p, split_idx)
        
        
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
        
        # Print all values
        print(f"Lift coefficient: {C_L:f} [-]")
        return self
    
    def rotate_90_degrees(self, vector):
        # Rotate vector 90 degrees counterclockwise
        return np.array([-vector[1], vector[0]])

    
    def compute_tangent_angles(self, object_points):
        
        dx = np.diff(object_points[:, 0])
        dy = np.diff(object_points[:, 1])
        
        # Compute the angle (in radians) of the tangent at each point
        angles = np.arctan2(dy, dx)
        
        # Last point angle
        angles = np.concatenate([angles, [angles[-1]]])
        
        return angles

    def generate_log_spaced_points(self, start_point, normal_vector):
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        
        # Compute the log-spaced distances
        distances = np.logspace(np.log10(1e-6), np.log10(0.5), self.num_domain_points)  # Adjust spacing factor as needed
        
        # Compute the x and y offsets based on normal vector
        dx = distances * normal_vector[0]
        dy = distances * normal_vector[1]
        
        
        log_spaced_points = np.column_stack([start_point[0] + dx, start_point[1] + dy])
        
        return log_spaced_points
    
    
    def generate_domain_points_around_object(self):
        # Compute the tangent (direction) at each boundary point
        angles = self.compute_tangent_angles(self.boundary_object_points)

        # Store the generated log-spaced points
        log_spaced_points_all = []

        # Generate log-spaced points from each boundary point along the normal direction
        for i, point in enumerate(self.boundary_object_points):
            # Get the tangent vector at this point
            angle = angles[i]
            tangent_vector = np.array([np.cos(angle), np.sin(angle)])
            
            # Calculate the normal vector by rotating the tangent 90 degrees counterclockwise
            normal_vector = self.rotate_90_degrees(tangent_vector)
            
            # Generate log-spaced points along the normal direction
            log_spaced_points = self.generate_log_spaced_points(point, normal_vector)
            log_spaced_points_all.append(log_spaced_points)

        # Convert list of log-spaced points to a numpy array
        log_spaced_points_all = np.vstack(log_spaced_points_all)
        
        return log_spaced_points_all
    
    def interpolate_for_cp(self, airfoil_coordinates, x_cp_data):
        x_airfoil, y_airfoil = airfoil_coordinates[:, 0], airfoil_coordinates[:, 1]
        interp_func = interp1d(x_airfoil, y_airfoil, kind="linear", fill_value="extrapolate", assume_sorted=True)
        y_new = interp_func(x_cp_data)
        return np.column_stack((x_cp_data, y_new))
        
    
    
    def transform_cp_data(self, cp_data_file):
        # Load Cp data
        cp_data = np.loadtxt(cp_data_file)
        x_c, cp = cp_data[:, 0], cp_data[:, 1]
        
        # Split data into upper and lower parts
        # Find the indices where the value 0 occurs
        split_indices = np.where(x_c == 0)[0] + 1

        # Now split the array at each occurrence of 0
        x_c_down, x_c_up = np.split(x_c, split_indices)
        cp_data_y_down, cp_data_y_up = np.split(cp, split_indices)
    
        # Get airfoil data
        # Obtain upper and lower parts
        upper_part_airfoil, lower_part_airfoil = self.split_object()

        # Obtain correct y for x point cp data
        upper_part_cp_coordinates = self.interpolate_for_cp(upper_part_airfoil, x_c_up)
        lower_part_cp_coordinates = self.interpolate_for_cp(lower_part_airfoil, x_c_down)
        

        # Merge back into a full airfoil
        full_object = np.vstack((upper_part_cp_coordinates, lower_part_cp_coordinates))  # Ensure correct order
        
        # Rotate the object
        rotated_object = self.rotate_object(full_object)
        
        # Convert to desired location
        full_object = rotated_object + [self.object_start]
        
    
        # Combine transformed coordinates with Cp values
        stacked_array = np.concatenate((cp_data_y_up, cp_data_y_down))
        
        transformed_cp_data = np.column_stack((full_object, stacked_array))
    
        return transformed_cp_data

    def inputs_generator_cfd_pinn(self, cp_data_file, rho_inf, V_inf, p_inf):
        # Obtain data into correct shape and position
        data = self.transform_cp_data(cp_data_file)
        
        # Convert data into x, y, cp
        x, y, cp = data[:, 0], data[:, 1], data[:, 2]
        
        # Convert cp
        p = (1/2) * rho_inf * V_inf**2 * cp + p_inf
        
        # generate u and v velocity components
        u = np.zeros(len(p))
        v = np.zeros(len(p))
        
        # Add all data together for inputs and outputs PINN
        inputs_pinn = np.column_stack((x, y))
        outputs_pinn = np.column_stack((u, v, p))
    
        return inputs_pinn, outputs_pinn
        
        


        
        
        