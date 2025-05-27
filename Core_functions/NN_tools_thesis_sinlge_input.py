import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR

# Compute partial derivative of input with respect to output for a NN
def gradient(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

class NN(nn.Module):
    "Create Connected Neural Network in PyTorch"
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, epochs=1000, loss=nn.MSELoss(), lr=1e-3, loss2=None, loss2_weight=0.1, lossBC=None, lossBC_weight=0.1, lossPositive=None, lossPositive_weight=0.1):
        super().__init__() # To activate nn.Module
        activation = nn.ReLU # Define activation function
        self.NN_Input = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()]) # Input layer
        self.NN_Hidden = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)]) # Hidden layers
        self.NN_Output = nn.Linear(N_HIDDEN, N_OUTPUT) # Output layer
        
        self.epochs = epochs
        self.loss = loss
        self.lr = lr
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lossBC = lossBC
        self.lossBC_weight = lossBC_weight
        self.lossPositive = lossPositive
        self.lossPositive_weight = lossPositive_weight
        #self.a = nn.Parameter(data=torch.tensor([5.13]))
        #self.n = nn.Parameter(data=torch.tensor([0.222]))
        #self.A_b = nn.Parameter(data=torch.rand(100, 1)*1e-2)
        #self.V_c = nn.Parameter(data=torch.rand(100, 1)*1e-2)
        #self.alpha = nn.Parameter(data=torch.tensor([0.]))
        #self.t_shift = nn.Parameter(data=torch.tensor([0.]))
        # Initialize weights
        self.apply(self.init_weights)
        



    # Function to go from numpy arrays to pytorch tensors
    def np_to_pt(self, x):
        return torch.from_numpy(x).to(torch.float).reshape(len(x), -1)

    # Forward pass function
    def forward(self, x):
        x = self.NN_Input(x)
        x = self.NN_Hidden(x)
        x = self.NN_Output(x)
        return x
    
    # Initialize weights
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            # He initialization for ReLU
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)  # Initialize biases to zero
    
    # Training algorithm
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Convert data to tensors
        X_train = self.np_to_pt(X_train)
        y_train = self.np_to_pt(y_train)
        
        # If validation data is provided, convert to tensors
        if X_val is not None and y_val is not None:
            X_val = self.np_to_pt(X_val)
            y_val = self.np_to_pt(y_val)
        
        # Put network in training mode and select optimizer + provide parameters to network (weights, biases and PINN)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)  # change learning rate
        self.train()
        
        # Initialize loss lists
        train_cost = []
        val_cost = []  # To store validation loss

        # For each epoch determine the losses and perform optimization
        for epoch in range(self.epochs):
            # Training step
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = self.loss(y_train, outputs)
            
            # Include additional physics-informed loss (if any)
            if self.loss2:
                loss += self.loss2_weight * self.loss2(self)
            
            # Include additional BC loss (if any)
            if self.lossBC:
                loss += self.lossBC_weight * self.lossBC(self)
            
            # Include additional positive value loss (if any)
            if self.lossPositive:
                loss += self.lossPositive_weight * self.lossPositive(self)
            
            # Perform backwards optimization (determine gradients)
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Append training loss
            train_cost.append(loss.item())
            
            # Validation step (if validation data is provided)
            if X_val is not None and y_val is not None:
                self.eval()  # Set the model to evaluation mode for validation
                with torch.no_grad():  # No need to calculate gradients during validation
                    val_outputs = self.forward(X_val)
                    val_loss = self.loss(y_val, val_outputs)
                    val_cost.append(val_loss.item())
                self.train()  # Set back to training mode

            # Print status for every 10 epochs
            if epoch % int(self.epochs / 10) == 0:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch}/{self.epochs}, Train cost: {train_cost[-1]:.4f}, Val cost: {val_cost[-1]:.4f}")
                else:
                    print(f"Epoch {epoch}/{self.epochs}, Train cost: {train_cost[-1]:.4f}")
                
        if X_val is not None and y_val is not None:
            return train_cost, val_cost
        return train_cost
    
    # Function to predict a certain output value
    def predict(self, X):
        self.eval()
        out = self.forward(self.np_to_pt(X))
        return out.detach().numpy()
