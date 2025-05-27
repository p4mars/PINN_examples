# Imported modules
import torch

class SRM_helper_tools():
    
    def __init__(self):
        pass

    # Helper functions Pressure and dPdt
    @staticmethod
    def r_integrator(P0, a , n, time, r_grain_init):
        drdt = (a/1e3) * (torch.abs(P0)*1e5/1e6)**n
        r_int = torch.cumulative_trapezoid(drdt, time, dim=0)
        r = torch.cat((torch.tensor([r_grain_init]).unsqueeze(0), r_grain_init + r_int))
        return r
    
    @staticmethod
    def A_b_calc(w_array, L_grain, r_grain_init, d_grain_out, alpha):
        
        A_b_array = torch.zeros_like(w_array)
        
        
        for i, w in enumerate(w_array):
        
            if 0 < w <= L_grain * torch.tan(alpha):
                A_b_array[i] = (torch.pi * (2 * r_grain_init + w) * w) / torch.sin(alpha)
        
            elif L_grain * torch.tan(alpha) < w <= d_grain_out / 2 - r_grain_init:
                A_b_array[i] = (r_grain_init + w) * ((2 * torch.pi * L_grain) / torch.cos(alpha)) - torch.pi * L_grain**2 * (torch.tan(alpha) / torch.cos(alpha))
        
            elif d_grain_out / 2 - r_grain_init < w <= d_grain_out / 2 - r_grain_init + L_grain * torch.tan(alpha):
                x = r_grain_init + w - L_grain * torch.tan(alpha)
                A_b_array[i] = (torch.pi * (d_grain_out / 2 + x) * (d_grain_out / 2 - x)) / torch.sin(alpha)
        
            else:
                A_b_array[i] = 0
        
        return A_b_array
    
    @staticmethod
    def V_c_integrator(P0, a, n, A_b, time, V_init):
        dV_cdt = (a/1e3) * (torch.abs(P0)*1e5/1e6)**n * A_b
        V_c_int = torch.cumulative_trapezoid(dV_cdt, time, dim=0)
        V_c = torch.cat((torch.tensor([V_init]).unsqueeze(0), V_init + V_c_int))
        return V_c
    
    @staticmethod
    # dPdt similar to numerical sim
    def dPdt_check(r, P0_array, P_a, is_burning_array, a, n,rho_grain, R_g, T_0, A_t, gamma, uninhibited_core, uninhibited_sides, L_grain, r_grain_init, d_grain_out, N_grains, L_chamber, d_chamber, t_liner, A_b=0):
        
        if A_b is not int(0):
            A_b = A_b
        else:
            A_b = ((uninhibited_core * (2 * torch.pi * r * (L_grain - uninhibited_sides * 2 * (r - r_grain_init))) + uninhibited_sides * (2 * torch.pi * ((d_grain_out**2 - (2 * r)**2) / 4)) )) * N_grains
        V_c = ( uninhibited_core * (torch.pi * r**2 * (L_grain - uninhibited_sides * 2 * (r - r_grain_init))) + uninhibited_sides * (torch.pi * (d_grain_out**2/4) * 2 * (r - r_grain_init))) * N_grains + (torch.pi * (d_chamber**2/4) * L_chamber) - N_grains * (torch.pi * ((d_grain_out + 2*t_liner)**2 / 4) * L_grain)

        dP0dt_array = torch.zeros_like(P0_array)
        
        for i, P0 in enumerate(P0_array):
            is_burning = is_burning_array[i]
            if not is_burning:
                #A_b = 0
                if P0 <= P_a:
                    dP0dt_array[i] = 0
                else:
                    dP0dt_array[i] = (- P0 * 1e5 * (A_t / V_c[i]) * torch.sqrt(torch.tensor([gamma * R_g * T_0 * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))]))) / 1e5
            else:
                dP0dt_array[i] = (((A_b[i]* (a / 1e3) * (torch.abs(P0) * 1e5 / 1e6) ** n) / V_c[i]) * (rho_grain * R_g * T_0 - P0 * 1e5) - P0 * 1e5 * (A_t / V_c[i]) * torch.sqrt(torch.tensor([gamma * R_g * T_0 * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))]))) / 1e5
                
        return dP0dt_array
    
    @staticmethod
    # dPdt for numerical sim angle
    def dPdt_new(r, P0_array, a, n, A_b, V_c, rho_grain, R_g, T_0, A_t, gamma):
        
        dP0dt_array = torch.zeros_like(P0_array)
        
        for i, P0 in enumerate(P0_array):
                dP0dt_array[i] = (((A_b[i]* (a / 1e3) * (torch.abs(P0) * 1e5 / 1e6) ** n) / V_c[i]) * (rho_grain * R_g * T_0 - P0 * 1e5) - P0 * 1e5 * (A_t / V_c[i]) * torch.sqrt(torch.tensor([gamma * R_g * T_0 * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))]))) / 1e5
        return dP0dt_array
    
    @staticmethod
    #dPdt for learnable parameters
    def dPdt_learn_param(P0_array, a, n, A_b, V_c, rho_grain, R_g, T_0, A_t, gamma):
    
        dP0dt_array = torch.zeros_like(P0_array)
        
        
        for i, P0 in enumerate(P0_array):
                dP0dt_array[i] = (((A_b[i]* (a / 1e3) * (torch.abs(P0) * 1e5 / 1e6) ** n) / V_c[i]) * (rho_grain * R_g * T_0 - P0 * 1e5) - P0 * 1e5 * (A_t / V_c[i]) * torch.sqrt(torch.tensor([gamma * R_g * T_0 * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1))]))) / 1e5
        return dP0dt_array
    
    @staticmethod
    def Gamma(gamma):
        return torch.sqrt(gamma) * (2 / (gamma + 1)) **((gamma + 1) / (2 * (gamma - 1)))
    @staticmethod
    def AeAt(Gamma, gamma, Pe, Pc):
        return Gamma / torch.sqrt(((2 * gamma) / (gamma - 1)) * (Pe / Pc)**(2 / gamma) *(1 - (Pe / Pc)**((gamma - 1) / gamma)))
    
    @staticmethod
    def AeAtM(gamma, M):
        return (1 / M**2) * ((2 / (gamma + 1)) * (1 + ((gamma - 1) / 2) * M**2))**((gamma + 1) / (gamma - 1))

    @staticmethod
    def c_star(Gamma, R, Tc):
        return (1 / Gamma) * torch.sqrt(R * Tc)
        
    @staticmethod
    def Cf(Gamma, gamma, Pe, Pc, Pa, AeAt):
        return Gamma * torch.sqrt(((2 * gamma) / (gamma - 1)) * (1 - (Pe / Pc)**((gamma - 1) / gamma))) + ((Pe / Pc) - (Pa / Pc)) * AeAt
    
    @staticmethod
    def F(m_dot, Ue, Pe, Pa, Ae):
        return m_dot * Ue + (Pe - Pa) * Ae

    @staticmethod
    def Ue(gamma, R, Tc, Pe, Pc):
        return torch.sqrt(((2 * gamma) / (gamma - 1)) * R * Tc * (1 - (Pe / Pc)**((gamma - 1) / gamma)))

    @staticmethod
    def AeAt_func(PePc, Gamma, gamma, Ae, At):
        return Gamma / torch.sqrt(((2 * gamma) / (gamma - 1)) * torch.pow(PePc, (2 / gamma)) * (1 - torch.pow(PePc,((gamma - 1) / gamma)))) - (Ae / At)

    @staticmethod
    def m_dot_id(Gamma, Pc, At, R, Tc):
        return (Gamma * Pc * At) / torch.sqrt(R * Tc)
    
    
    @staticmethod
    # Define the Newton-Raphson method with extra arguments
    def fsolve_PyTorch(func, initial_guess, args=(), tol=1e-6, max_iter=100):
        """
        Newton-Raphson solver to find root of `func(x, *args) = 0`.
        
        Parameters:
        - func: function to find root for, i.e., func(x, *args) = 0
        - initial_guess: starting value for the root search
        - args: tuple of extra arguments to pass to `func`
        - tol: tolerance for convergence
        - max_iter: maximum number of iterations
        
        Returns:
        - x: the root found
        """
        # Initialize x as a tensor with autograd enabled
        x = torch.tensor(initial_guess, dtype=torch.float32, requires_grad=True)
        
        for _ in range(max_iter):
            # Compute function value and gradient
            y = func(x, *args)
            y.backward()  # Compute the derivative dy/dx using autograd
            
            # Update step: x_new = x - f(x) / f'(x)
            with torch.no_grad():
                x_new = x - y / x.grad  # x.grad contains dy/dx
                
            # Clear the gradients for the next iteration
            x.grad.zero_()

            # Check for convergence
            if torch.abs(x_new - x) < tol:
                return x_new.detach()  # Return the root as a tensor after convergence
            
            x = x_new.requires_grad_(True)  # Re-enable autograd for the next iteration

        raise RuntimeError("Newton-Raphson did not converge within the maximum number of iterations")


#---------------- OTHER LOSS FUNCTION EXAMPLES ---------------------
'''
# PINN loss with ignition time delay
def PINN_loss_angle(NN_output: torch.nn.Module):
    # Collocation points time array
    steps = 100
    time_inputs = torch.linspace(1e-16, 6, steps).view(-1, 1).requires_grad_(True)    
    
    # Obtain predictions from the network (Pc, a, n)
    P_C = NN_output(time_inputs) 
    #a = NN_output.a
    #n = NN_output.n
    alpha = NN_output.alpha
    alpha = alpha * torch.pi / 180
    
    # Obtain derivative of Pc with respect to time over the network
    dP_C_dt = gradient(P_C, time_inputs) 
    
    # Obtain r_integrator
    radius = r_integrator(P_C, a, n, time_inputs, 0)
    
    # Obtain A_b regions
    A_b = A_b_calc(radius, L_grain, r_grain_init, d_grain_out, alpha)
    
    # Obtain V_c_integrator
    V_c = V_c_integrator(P_C, a, n, A_b, time_inputs, V_init)
    
    # Obtain dPdt from formula
    dPdt_theory = dPdt_new(radius, P_C, a, n, A_b, V_c)
    
    return torch.mean((dPdt_theory - dP_C_dt)**2)


# PINN los with learnable paramaters -> sort of cheating and hard to justify
def PINN_loss_learn_param(NN_output: torch.nn.Module):
    # Collocation points time array
    steps = 100
    time_inputs = torch.linspace(1e-16, 8, steps).view(-1, 1).requires_grad_(True)    
    
    # Obtain predictions from the network (Pc, a, n)
    P_C = NN_output(time_inputs) 
    a = NN_output.a
    n = NN_output.n
    A_b = NN_output.A_b
    V_c = NN_output.V_c
    
    # Obtain derivative of Pc with respect to time over the network
    dP_C_dt = gradient(P_C, time_inputs) 
    
    # Obtain dPdt from formula
    dPdt_theory = dPdt_learn_param(P_C, a, n, A_b, V_c)
    
    return torch.mean((dPdt_theory - dP_C_dt)**2)

# New PINN loss with correct implemtnation of formulas
def PINN_loss_combined(NN_output: torch.nn.Module, used_inputs):
    # Collocation points time array
    start, end = 1e-16, 8
    steps = 200
    time_inputs = torch.linspace(start, end, steps).view(-1, 1).requires_grad_(True)
    log_space1 = torch.logspace(start=torch.log10(torch.tensor(1e-16)),end=torch.log10(torch.tensor(0.5)),steps=200).view(-1,1).requires_grad_(True)
    #log_space2 = torch.logspace(start=torch.log10(torch.tensor(end)),end=torch.log10(torch.tensor(start)),steps=steps).view(-1,1).requires_grad_(True)
    
    # Normalize to your range [start, end] by inverting
    #inverse_log_space = end - (end - start) * (log_space2 - log_space2.min()) / (log_space2.max() - log_space2.min())

    # Concatenate with reversed version to get more density at both start and end
    time_inputs = torch.cat((log_space1, time_inputs), dim=0)

    
    
    # Setup total loss
    total_loss = 0

    for scaled_inputs in used_inputs:
        inputs = torch.expm1(scaled_inputs)
        #time_shift = NN_output.t_shift
        time_shift = 0.0
        
        
        # Unzip constant values from inputs
        L_grain = inputs[0]
        d_grain_in = inputs[1]
        d_grain_out = inputs[2]
        t_liner = inputs[3]
        mass_grain = inputs[4]
        N_grains = inputs[5]
        uninhibited_core = inputs[6]
        uninhibited_sides = inputs[7]
        gamma = inputs[9]
        W_g = inputs[10]
        T_0 = inputs[11]
        d_chamber = inputs[12]
        L_chamber = inputs[13]
        d_t = inputs[14]
        d_e = inputs[15]
        a = inputs[17]
        n = inputs[18]
        

        # Perform calulcations for standard values
        R_g = R_a / W_g 
        A_t = torch.pi * d_t**2/4  # Nozzle throat area [m2]
        A_e = torch.pi * d_e**2/4 # Nozzle exit area [m2]
        rho_grain =  mass_grain / ( L_grain * (np.pi / 4) * (d_grain_out**2 - d_grain_in**2))# Density of grain [kg/m3]
        
        
        # Find best time shift
        time_inputs = time_inputs + time_shift

        # Repeat the constants for each time step (number of time points)
        # Assuming time_inputs is a tensor with shape (number of time points, 1)
        constant_features_repeated = inputs.unsqueeze(0).repeat(time_inputs.size(0), 1)
    
        # Reshape the time feature to make it compatible for concatenation
        # time_feature is assumed to be of shape (number of time points, 1)
        time_feature = time_inputs.reshape(-1, 1)  # This should already be (number of time points, 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_feature, constant_features_repeated), dim=1)
        
        # Scaled input data
        input_scaled = torch.log1p(input_data)
    
    
        # Obtain predictions from the network (Pc, F, a, n)
        outputs_of_network = NN_output(input_scaled)
        P_C_scaled = outputs_of_network[:,0].unsqueeze(-1)
        P_C = torch.expm1(P_C_scaled)
        F_network = torch.expm1(outputs_of_network[:,1].unsqueeze(-1))
        #a = NN_output.a
        #n = NN_output.n

        
        # Obtain derivative of Pc with respect to time over the network
        dP_C_dt = gradient(P_C_scaled, input_scaled)[:,0].unsqueeze(-1)
        
        #print(dP_C_dt)
        
        # Obtain r_integrator
        radius = SRM_helper_tools.r_integrator(P_C, a, n, time_inputs, d_grain_in/2)
        
        # Check when motor is burning or not
        burning_logic = torch.logical_and(radius <= (d_grain_out/2), (radius - (d_grain_in/2)) <= L_grain/2)

        # Obtain dPdt from formula
        dPdt_theory = SRM_helper_tools.dPdt_check(radius, P_C, P_a, burning_logic, a, n, rho_grain, R_g, T_0, A_t, gamma, uninhibited_core, uninhibited_sides, L_grain, d_grain_in/2, d_grain_out, N_grains, L_chamber, d_chamber, t_liner, A_b=0)
        #print(torch.abs(dPdt_theory - dP_C_dt))
        
        # Obtain correct pressure ratio
        pepc = SRM_helper_tools.fsolve_PyTorch(SRM_helper_tools.AeAt_func, 0.001, args=(SRM_helper_tools.Gamma(gamma), gamma, A_e, A_t))
        
        # Calculate nozzle exit pressure
        P_e_sol = P_C*1e5 * pepc  # Units = [Pa]
        
        # Obtain thrsut coefficient
        C_f = SRM_helper_tools.Cf(SRM_helper_tools.Gamma(gamma), gamma, P_e_sol, P_C*1e5, P_a*1e5, (A_e / A_t))
        
        # Calculate theoretical thrust
        F_theory = C_f * P_C*1e5 * A_t # Units = [N]
        
        
        # Determine loss for specific case and add to total loss
        total_loss += torch.mean((dPdt_theory - dP_C_dt)**2)  + torch.mean((F_theory - F_network)**2)
        #print(total_loss)

    return total_loss / len(used_inputs) 



def Positive_loss_combined(NN_output: torch.nn.Module, used_inputs):
    steps = 20
    time_inputs = torch.linspace(0, 10, steps).view(-1, 1).requires_grad_(True)
    
    # Setup total loss
    total_loss = 0
    
    for scaled_inputs in used_inputs:
        inputs = torch.expm1(scaled_inputs)
    
        # Repeat the constants for each time step (number of time points)
        # Assuming time_inputs is a tensor with shape (number of time points, 1)
        constant_features_repeated = inputs.unsqueeze(0).repeat(time_inputs.size(0), 1)
    
        # Reshape the time feature to make it compatible for concatenation
        # time_feature is assumed to be of shape (number of time points, 1)
        time_feature = time_inputs.reshape(-1, 1)  # This should already be (number of time points, 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_feature, constant_features_repeated), dim=1)
        
        # Scaled input data
        input_scaled = torch.log1p(input_data)

        # Obtain predictions from the network (Pc, F, a, n)
        outputs_network = NN_output(input_scaled)
        P_C = outputs_network[:,0].unsqueeze(-1)
        F_network = outputs_network[:,1].unsqueeze(-1)
        
        total_loss += torch.mean(torch.clamp(-P_C, min=0)**2) + torch.mean(torch.clamp(-F_network, min=0)**2)

    return total_loss / len(used_inputs)


# Define force boundary loss function (0 at start)
def Boundary_loss_combined(NN_output: torch.nn.Module, used_inputs):
    t_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)
    
    # Setup total loss
    total_loss = 0
    
    
    for scaled_inputs in used_inputs:
        inputs = torch.expm1(scaled_inputs)
        
        # Repeat the constants for each time step (number of time points)
        # Assuming time_inputs is a tensor with shape (number of time points, 1)
        constant_features_repeated = inputs.unsqueeze(0).repeat(t_boundary.size(0), 1)
    
        # Reshape the time feature to make it compatible for concatenation
        # time_feature is assumed to be of shape (number of time points, 1)
        time_feature = t_boundary.reshape(-1, 1)  # This should already be (number of time points, 1)
    
        # Concatenate the constant features with the time feature along the second axis (columns)
        input_data = torch.cat((time_feature, constant_features_repeated), dim=1)
        
        # Scaled input data
        input_scaled = torch.log1p(input_data)
    
        # Obtain predictions from the network (Pc, F, a, n)
        outputs_network = NN_output(input_scaled)
        P_C = outputs_network[:,0].unsqueeze(-1)
        F_network = outputs_network[:,1].unsqueeze(-1)
        
        total_loss += (torch.squeeze(P_C) - P_a)**2 + (torch.squeeze(F_network) - 0)**2
        
    return total_loss / len(used_inputs)











'''

