import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Use same weight and bias intialization every time for Neural Network
torch.manual_seed(42)

# Compute partial derivative of input with respect to output for a NN
def gradient(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

class NN(nn.Module):
    """"
    Create Connected Neural Network in PyTorch
    
    This class is designed and created to make the development of Feed-Forward Nerual Networks more straightfore, as well as to easily implment Physics-Informed Neural Networks usinf a Feed-Forward architecture.
    """
    
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS, activation_fn=nn.ReLU, init_method=nn.init.xavier_uniform_, optimizer_type=optim.Adam, batch_size=32, use_batch=False, epochs=1000, lr=1e-3, loss_init=nn.MSELoss(), loss_terms=None, learn_params=None, unique_input=False, show_plots=True, dropout_prob=0):
        """

        Parameters
        ----------
        N_INPUT : int
            Number of inputs of the Neural Network.
        N_OUTPUT : int
            Number of outputs of the Neural Network.
        N_HIDDEN : int
            Amount of Neurons per hidden layer.
        N_LAYERS : int
            Amount of hidden layers.
        activation_fn : nn.<activation_function>, optional
            Type of activation function. The default is nn.ReLU.
        init_method : nn.<init_method>, optional
            Type of weight initialization method. The default is nn.init.xavier_uniform_.
        batch_size : int, optional
            Batch size. The default is 32.
        use_batch : Boolean, optional
            Define if you would like to use batch training. The default is False.
        epochs : int, optional
            Amount of epochs. The default is 1000.
        lr : float, optional
            Learning rate. The default is 1e-3.
        loss_init : TYPE, optional
            Define if you would like to train the Neural Network with data points or not. Select None if you don't want this. The default is nn.MSELoss().
        loss_terms : Dictionary, optional
            Insert Physics and boundary losses. The default is None.
        learn_params : Dictionary, optional
            Insert learnable parameters to the newtowrk for paramater discover. The default is None.
        unique_input : Boolean, optional
            Define if you would like to have unique inputs for the physics loss, when using various inputs to the training loss. The default is False.
        show_plots : Boolean, optional
            Define if you would like to have a plot of the training and validation losses per epoch. The default is True.

        Returns
        -------
        None.

        """
        
        super().__init__() # To activate nn.Module
        
        
        self.activation = activation_fn # Define activation function
        self.init_method = init_method # Define weight intialization method
        
        '''
        # Setup the network and amount of hidden layers
        self.NN_Input = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        self.activation()]) # Input layer
        self.NN_Hidden = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            self.activation()]) for _ in range(N_LAYERS-1)]) # Hidden layers
        self.NN_Output = nn.Linear(N_HIDDEN, N_OUTPUT) # Output layer
        '''
        
        # Define input layer
        self.NN_Input = nn.Sequential(
            nn.Linear(N_INPUT, N_HIDDEN),
            activation_fn()  # Apply activation after the input layer
        )

        # Define hidden layers (a list of layers)
        hidden_layers = []
        for _ in range(N_LAYERS - 1):  # N_LAYERS - 1 because the first hidden layer is already added
            hidden_layers.append(nn.Linear(N_HIDDEN, N_HIDDEN))
            hidden_layers.append(activation_fn())  # Apply activation
            #hidden_layers.append(nn.Dropout(p=dropout_prob))  # Apply dropout after activation

        # Combine all hidden layers into a single Sequential block
        self.NN_Hidden = nn.Sequential(*hidden_layers)

        # Define the output layer
        self.NN_Output = nn.Linear(N_HIDDEN, N_OUTPUT)
        
        
        
        # Setup other parameters of the class
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.use_batch = use_batch
        self.epochs = epochs
        self.loss_init = loss_init
        self.lr = lr
        self.unique_input = unique_input
        self.show_plots = show_plots
        self.dropout_prob = dropout_prob
        
        # Setup various loss terms
        if loss_terms is None:
           loss_terms = {}
        self.loss_terms = loss_terms
        
        
        # Setup learnable parameters using names in learn_param_names
        if learn_params is None:
            learn_params = {}
        for param_name, initial_value in learn_params.items():
            # Use initial value or default to zero if None
            initial_value = torch.tensor([initial_value if initial_value is not None else 0.0], dtype=torch.float32, requires_grad=True)
            param = nn.Parameter(data=initial_value)  # Create learnable parameter
            setattr(self, param_name, param)  # Store each as an attribute with the given name
    
        # Initialize weights
        self.apply(self.init_weights)
        
        # Add dropout
        self.dropout = nn.Dropout(p=0.5)


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
            self.init_method(layer.weight)
            nn.init.constant_(layer.bias, 0.)  # Initialize biases to zero
        
        
        '''
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.bias, 0.)  # Initialize biases to zero
            if self.activation == nn.ReLU:
                # Initialization for ReLU
                nn.init.xavier_uniform_(layer.weight)
            elif self.activation == nn.Tanh:
                # Initialization for Tanh
                nn.init.xavier_uniform_(layer.weight)
            else:
                print('Error, this activation function is not yet supported')
                '''
    
    # Print the intial weights (debugging)            
    def print_initial_weights(self):
        print("\nInitial Weights and Biases:")
        for name, param in self.named_parameters():
            if 'weight' in name:
                print(f"{name} -> Mean: {param.data.mean().item()}, Std: {param.data.std().item()}")
            elif 'bias' in name:
                print(f"{name} -> Bias: {param.data}")
            
    
    # Training algorithm
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Convert data to tensors
        X_train = self.np_to_pt(X_train)
        y_train = self.np_to_pt(y_train)
        
        
        # Determine if Batch training is being used or not
        if self.use_batch:
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            train_loader = [(X_train, y_train)]  # In case you don't want to use batch training
        
        
        
        # If validation data is provided, convert to tensors
        if X_val is not None and y_val is not None:
            X_val = self.np_to_pt(X_val)
            y_val = self.np_to_pt(y_val)
            # Add this for batch training
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        # Select optimizer + provide parameters to network (weights, biases and PINN)
        optimizer = self.optimizer_type(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=100000, gamma=0.9)  # change learning rate by half each 1000 steps
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        
        
        # Initialize loss lists
        train_cost = []
        val_cost = []  # To store validation loss

        # For each epoch determine the losses and perform optimization
        for epoch in range(self.epochs):
            # Put network in training mode
            self.train()
            
            batch_losses = []
            
            for X_batch, y_batch in train_loader:
                
                # Optionally make X_batch unique for additional losses and remove first input 
                if self.unique_input is not False:
                    used_input = torch.unique(X_batch[:,1:], dim=0)
                else:
                    used_input = X_batch
                
                # Training step
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                
                # Optionally add experimental loss
                if self.loss_init is not None:
                   loss = self.loss_init(y_batch, outputs)
                else:
                    loss = 0
                    
            
                # Calulcate losses/costs
                for loss_function, weight in self.loss_terms.items():
                    if weight > 0:
                        loss += weight * loss_function(self, used_input) # Update this line to enable correct inputs into batch training, time has been removed already

                # Perform backwards optimization (determine gradients)
                loss.backward()
                
                # Clip gradients if needed
                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                
                # Add batch losses to list
                batch_losses.append(loss.item())
            
                # Append training loss (average of each batch)
                train_cost.append(sum(batch_losses) / len(batch_losses))
            
                # Validation step (if validation data is provided)
                if X_val is not None and y_val is not None:
                    self.eval()  # Set the model to evaluation mode for validation
                    with torch.no_grad():
                        val_losses = [nn.MSELoss()(self.forward(X_batch), y_batch).item() for X_batch, y_batch in val_loader]
                        val_cost.append(sum(val_losses) / len(val_losses))

                
                # Update learning rate
                scheduler.step()

            # Print status for every 10 epochs
            if epoch % int(self.epochs / 10) == 0:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch}/{self.epochs}, Train cost: {train_cost[-1]:.4f}, Val cost: {val_cost[-1]:.4f}")
                else:
                    print(f"Epoch {epoch}/{self.epochs}, Train cost: {train_cost[-1]:.4f}")
        
        if self.show_plots is True:
            if X_val is not None and y_val is not None:
                #return train_cost, val_cost
                return self.plot_losses(train_cost, val_cost)
            #return train_cost
            return self.plot_losses(train_cost)
        
    
    # Function to predict a certain output value
    def predict(self, X):
        self.eval()
        out = self.forward(self.np_to_pt(X))
        return out.detach().numpy()
    
    def plot_losses(self, train_loss=None, val_loss=None):
        plt.figure()
        plt.xlabel("Epochs")
        plt.ylabel("Mean squared error cost")
        if train_loss is not None:
            plt.plot(train_loss, label="Train cost")
        if val_loss is not None:
            plt.plot(val_loss, label="Validation cost")
        plt.legend()
        plt.yscale('log')
        plt.show()
        return train_loss, val_loss
    
    
    