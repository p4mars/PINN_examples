# Imported modules
#from NN_tools import ParNet, gradient, Net
import numpy as np
import matplotlib.pyplot as plt
import torch
from NN_tools_thesis import NN, gradient


#initialize random
np.random.seed(10)

# Setup constants
T_env = 19.1
T_0 = 64
R = 2.5e-4
#R = 0.005

# Define scaling
def min_max_scale(data, min_val=None, max_val=None):
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    return (data - min_val) / (max_val - min_val), min_val, max_val

def inverse_min_max_scale(data_scaled, min_val, max_val):
    return data_scaled * (max_val - min_val) + min_val

# Define equation
def cooling_law(time, Tenv, T0, R):
    T = Tenv + (T0 - Tenv) * np.exp(-R * time)
    return T

true_time = np.arange(0, 15e3, 0.1)
true_temp = cooling_law(true_time, T_env, T_0, R)

# Sampled data
Data_time = np.linspace(0, 100, 10)
Data_time_scaled = np.log1p(Data_time)

Data_temp = cooling_law(Data_time, T_env, T_0, R) + 1 * np.random.randn(10)
Data_temp_scaled = np.log1p(Data_temp)
#np.random.shuffle(Data_temp)
#Data_time = np.append(Data_time, 450)
#Data_temp = np.append(Data_temp, cooling_law(450, T_env, T_0, R))


# Use same weight and bias intialization every time
torch.manual_seed(42)
'''

# Call network Vanilla
network = Net(1, 1, epochs=20000, lr=1e-5, loss2=None)

# Train network Vanilla
losses = network.fit(Data_time, Data_temp)

# Plot losses
plt.figure()
plt.plot(losses)
plt.yscale('log')

# Predict outcomes
pred_data = network.predict(true_time)

'''



# PINN

# Compute partial derivate of input with respect to output
#def gradient(outputs, inputs):
#    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)
    

# Define loss function PINN
def PINN_loss_original(NN_output: torch.nn.Module(), used_inputs):
    dt = torch.linspace(0, 15e3, steps=1000).view(-1, 1).requires_grad_(True)
    temp = NN_output(dt)# Obtains values from forward function of Net class
    dT = gradient(temp, dt)
    
    cooling_term = R * (T_env - temp)
    eq = cooling_term - dT

    #print(f"Cooling term: {cooling_term.mean().item()}, Derivative term: {dT.mean().item()}")
    return torch.mean(eq**2)



def PINN_loss(NN_output: torch.nn.Module()):
    dt = torch.linspace(0, 15e3, steps=1000).view(-1, 1).requires_grad_(True)
    dt_scaled = torch.log1p(dt)
    temp = NN_output(dt_scaled) # Obtains values from forward function of Net class
    dT = gradient(temp, dt_scaled) * (1 / (1 + dt))
    eq = R * (T_env - temp) - dT

    return torch.mean(eq**2)




loss_functions = {
    PINN_loss_original: 100
    }


# Create network
#PINN_network = Net(1, 1, epochs=30000, loss2=PINN_loss_original, loss2_weight=250, lr=1e-5)
PINN_network = NN(1, 1, 100, 3, epochs=30000, batch_size=64, use_batch=False, lr=1e-4, loss_terms=loss_functions, learn_params=None)

PINN_network.fit(Data_time, Data_temp)

# Predict outcomes
predict_time = np.log1p(true_time)
pred_data_PINN = PINN_network.predict(true_time)



'''
# PINN parameter finder
test_lst = []
test_lst_2 = []

# Define loss function PINN
def PINN_loss(NN_output: torch.nn.Module()):
    dt = torch.linspace(0, 1000, steps=1000).view(-1, 1).requires_grad_(True)
    temp = NN_output(dt) # Obtains values from forward function of Net class
    dT = gradient(temp, dt)[0]
    eq = NN_output.r * (T_env - temp) - dT
    
    test_lst.append(dT)
    test_lst_2.append(NN_output.r * (T_env - temp))

    return torch.mean(eq**2)


# Create network
PINN_network = ParNet(1, 1, epochs=4000, loss2=PINN_loss, loss2_weight=10, lr=1e-4)

losses2 = PINN_network.fit(Data_time, Data_temp)

# Plot losses
plt.figure()
plt.plot(losses2)
plt.yscale('log')

# Predict outcomes
pred_data_PINN = PINN_network.predict(true_time)
r_data = PINN_network.r.data.numpy()[0]
print(f'Estimation of parameter R for heat equation: {r_data:f}')
'''

# Plot data
plt.figure()
plt.plot(true_time, true_temp, label='Equation')
plt.plot(Data_time, Data_temp, 'o', label='Training data')
#plt.plot(true_time, pred_data, label='Vanilla NN')
plt.plot(true_time, pred_data_PINN, label='PINN')
plt.legend()
plt.ylabel('Temperature (C)')
plt.xlabel('Time (s)')
plt.show()