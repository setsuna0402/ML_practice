import torch
import torch.nn as nn
import math
from torchinfo import summary

class FCN(nn.Module):
    def __init__(self, N_hidden=64, dropout_rate=0.0):
        super().__init__()
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(1, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, 1)
        )
        

    def forward(self, x):
        return self.network(x)
    
class FCN_two(nn.Module):
    def __init__(self, N_hidden=64, dropout_rate=0.0):
        super().__init__()
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(2, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, 1)
        )
        

    def forward(self, x):
        return self.network(x)
    

class FCN_four(nn.Module):
    def __init__(self, N_hidden=64, dropout_rate=0.0):
        super().__init__()
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(4, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, 1)
        )
        

    def forward(self, x):
        return self.network(x)


class FCN_four_short(nn.Module):
    def __init__(self, N_hidden=64, dropout_rate=0.0):
        super().__init__()
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(4, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, 1)
        )
        

    def forward(self, x):
        return self.network(x)



class FCN_tiny(nn.Module):
    # for multiple stage training
    def __init__(self, N_hidden=32, f_d=None, kappa=None):
        """
        Args:
            N_hidden: neurons in the hidden layers
            f_d: main frequency of the input data (used to compute scaling factor)
            kappa: manually specified scaling factor (if provided, f_d will be ignored)
        """
        super().__init__()
        self.activation = nn.Tanh()  
        
        self.network = nn.Sequential(
            nn.Linear(4, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, 1)
        )
        
        # If f_d or kappa is provided, scale the first layer weights accordingly
        if kappa is not None:
            self._scale_first_layer(kappa)
        elif f_d is not None:
            self._scale_first_layer_by_freq(f_d, N_hidden)
        # default initialization: no scaling if neither f_d nor kappa is provided

    def _scale_first_layer_by_freq(self, f_d, N_hidden):
        """
        kapp >= pi * f_d * sqrt(var), where var = 2 / (dim_input + N_hidden)
        """
        # dim_input is 4 for this network
        var = 2.0 / (4 + N_hidden)
        kappa = math.pi * f_d * math.sqrt(var)
        # Ensure kappa is larger than the condition
        kappa *= 1.2
        self._scale_first_layer(kappa)

    def _scale_first_layer(self, kappa):
        """
        scaling the first layer weights by kappa
        """
        first_linear = self.network[0]
        # In principle, we don't need to initialize the first layer weights and biases,
        # initialisation will be done when the model is created (self.network = xxx), 
        # but we can do it here to ensure that the scaling is applied correctly.
        # Initialize the first layer weights and biases by Xavier normal and zeros
        nn.init.xavier_normal_(first_linear.weight)
        nn.init.zeros_(first_linear.bias)
        # Scaling the first layer weights by kappa
        with torch.no_grad():
            first_linear.weight.data *= kappa
        # Initialize the rest of the layers with Xavier normal and zeros
        for layer in self.network[1:]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        # print(f"First layer weight scaled by kappa = {kappa:.2f}")

    def forward(self, x):
        return self.network(x)


class FCN_tiny_hard_constraint(nn.Module):
    # for multiple stage training
    def __init__(self, N_hidden=32, f_d=None, kappa=None):
        """
        Args:
            N_hidden: neurons in the hidden layers
            f_d: main frequency of the input data (used to compute scaling factor)
            kappa: manually specified scaling factor (if provided, f_d will be ignored)
        """
        super().__init__()
        self.activation = nn.Tanh()  
        
        self.network = nn.Sequential(
            nn.Linear(4, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, 1)
        )
        
        # If f_d or kappa is provided, scale the first layer weights accordingly
        if kappa is not None:
            self._scale_first_layer(kappa)
        elif f_d is not None:
            self._scale_first_layer_by_freq(f_d, N_hidden)
        # default initialization: no scaling if neither f_d nor kappa is provided

    def _scale_first_layer_by_freq(self, f_d, N_hidden):
        """
        kapp >= pi * f_d * sqrt(var), where var = 2 / (dim_input + N_hidden)
        """
        # dim_input is 4 for this network
        var = 2.0 / (4 + N_hidden)
        kappa = math.pi * f_d * math.sqrt(var)
        # Ensure kappa is larger than the condition
        kappa *= 1.2
        self._scale_first_layer(kappa)

    def _scale_first_layer(self, kappa):
        """
        scaling the first layer weights by kappa
        """
        first_linear = self.network[0]
        # In principle, we don't need to initialize the first layer weights and biases,
        # initialisation will be done when the model is created (self.network = xxx), 
        # but we can do it here to ensure that the scaling is applied correctly.
        # Initialize the first layer weights and biases by Xavier normal and zeros
        nn.init.xavier_normal_(first_linear.weight)
        nn.init.zeros_(first_linear.bias)
        # Scaling the first layer weights by kappa
        with torch.no_grad():
            first_linear.weight.data *= kappa
        # Initialize the rest of the layers with Xavier normal and zeros
        for layer in self.network[1:]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        # print(f"First layer weight scaled by kappa = {kappa:.2f}")

    def forward(self, x):   
        location = x[:, 0:1]  # Extract location from the input tensor, keep shape [N, 1]
        raw = self.network(x)
        # Apply the hard constraint for the initial condition
        out = location * raw  # hard constraint: I(x=0) = I_o, so residual is zero at x=0
        # out = I_o + location * raw  # hard constraint: I(x=0) = I_o
        return out


class FCN_hard_bc_four(nn.Module):
    def __init__(self, N_hidden=64, dropout_rate=0.0):
        super().__init__()
        self.activation = nn.ReLU()
        self.network = nn.Sequential(
            nn.Linear(4, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, N_hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(N_hidden, 1)
        )
        
    def forward(self, x):   
        location = x[:, 0:1]  # Extract location from the input tensor, keep shape [N, 1]
        I_o = x[:, 1:2]  # Extract I_o from the input tensor, keep shape [N, 1]
        raw = self.network(x)
        # Apply the hard constraint for the initial condition
        out = I_o + (1.0 - torch.exp(-location)) * raw  # hard constraint: I(x=0) = I_o
        # out = I_o + location * raw  # hard constraint: I(x=0) = I_o
        return out

class FF_FCN_four(nn.Module):
    def __init__(self, input_dim=4, N_hidden=64, ff_dim=64, sigma=8.0):
        super().__init__()
        # This might look like parammeter B will be different for each input, 
        # but B are fixed after initialization, and will not be updated during training.
        self.B = nn.Parameter(torch.randn(input_dim, ff_dim) * sigma, requires_grad=False)
        self.activation = nn.Tanh()
        self.network = nn.Sequential(
            nn.Linear(ff_dim*2, N_hidden), # *2 because we concatenate sin and cos features
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(N_hidden, 1)
        )

    def forward(self, x):
        # x: (batch_size, input_dim)
        # B: (input_dim, ff_dim)
        # x_proj: (batch_size, ff_dim)
        x_proj = 2 * torch.pi * x @ self.B
        ff_feat = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        out = self.network(ff_feat)
        return out

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=1.0, is_first=False):
        super().__init__()

        self.in_features = in_features
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(
                    -1.0 / in_features,
                     1.0 / in_features
                )
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class FCN_tiny_sine_hard_constraint(nn.Module):

    def __init__(self, N_hidden=32, omega_first=30.0 ,omega_hidden=1.0):
        super().__init__()

        self.network = nn.Sequential(
            SineLayer(4, N_hidden, omega_0=omega_first, is_first=True),
            SineLayer(N_hidden, N_hidden, omega_0=omega_hidden),
            SineLayer(N_hidden, N_hidden, omega_0=omega_hidden),
            SineLayer(N_hidden, N_hidden, omega_0=omega_hidden),
            nn.Linear(N_hidden, 1)
        )

    def forward(self, x):

        location = x[:, 0:1]
        raw = self.network(x)

        return location * raw

class FCN_tiny_sine_tanh_hard_constraint(nn.Module):

    def __init__(self, N_hidden=64, omega_first=30.0):
        super().__init__()

        self.network = nn.Sequential(
            SineLayer(4, N_hidden, omega_0=omega_first, is_first=True),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, 1)
        )
        with torch.no_grad():
        # Initialize the rest of the layers with Xavier normal and zeros
            for layer in self.network[1:]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):

        location = x[:, 0:1]
        raw = self.network(x)

        return location * raw

class FCN_tiny_sine_tanh_hard_constraint_b(nn.Module):
    '''
    transform the location input with a sine layer, then concatenate with the condition variables (I_o, kappa, j_emit), and feed into a fully connected network with Tanh activations.
    '''

    def __init__(self, N_hidden=64, N_sine=64, omega_first=30.0):
        super().__init__()

        self.omega_first = omega_first
        self.x_layer = nn.Linear(1, N_sine)

        self.network = nn.Sequential(
            nn.Linear(N_sine + 3, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, N_hidden),
            nn.Tanh(),
            nn.Linear(N_hidden, 1)
        )
        with torch.no_grad():
            self.x_layer.weight.uniform_(-1.0 / 1, 1.0 / 1)  # Initialize the first layer weights for x_layer
            nn.init.zeros_(self.x_layer.bias)  # Initialize the bias for x_layer
        # Initialize the rest of the layers with Xavier normal and zeros
            for layer in self.network[1:]:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):

        location = x[:, 0:1]
        condition = x[:, 1:4]  # Extract the condition variables (I_o, kappa, j_emit)
        x_transformed = torch.sin(self.omega_first * self.x_layer(location))
        h = torch.cat([x_transformed, condition], dim=1)  # Concatenate the transformed x with the condition variables
        raw = self.network(h)

        return location * raw


class FCN_tiny_LeakRELU(nn.Module):
    # for multiple stage training
    def __init__(self, N_hidden=64, f_d=None, kappa=None):
        """
        Args:
            N_hidden: neurons in the hidden layers
            f_d: main frequency of the input data (used to compute scaling factor)
            kappa: manually specified scaling factor (if provided, f_d will be ignored)
        """
        super().__init__()
        self.activation = nn.LeakyReLU(0.2) 
        
        self.network = nn.Sequential(
            nn.Linear(4, N_hidden),
            self.activation,
            nn.Linear(N_hidden, N_hidden),
            self.activation,
            nn.Linear(N_hidden, N_hidden),
            self.activation,
            nn.Linear(N_hidden, N_hidden),
            self.activation,
            nn.Linear(N_hidden, 1)
        )
        
        # If f_d or kappa is provided, scale the first layer weights accordingly
        if kappa is not None:
            self._scale_first_layer(kappa)
        elif f_d is not None:
            self._scale_first_layer_by_freq(f_d, N_hidden)
        # default initialization: no scaling if neither f_d nor kappa is provided

    def _scale_first_layer_by_freq(self, f_d, N_hidden):
        """
        kapp >= pi * f_d * sqrt(var), where var = 2 / (dim_input + N_hidden)
        """
        # dim_input is 4 for this network
        var = 2.0 / (4 + N_hidden)
        kappa = math.pi * f_d * math.sqrt(var)
        # Ensure kappa is larger than the condition
        kappa *= 1.2
        self._scale_first_layer(kappa)

    def _scale_first_layer(self, kappa):
        """
        scaling the first layer weights by kappa
        """
        first_linear = self.network[0]
        # In principle, we don't need to initialize the first layer weights and biases,
        # initialisation will be done when the model is created (self.network = xxx), 
        # but we can do it here to ensure that the scaling is applied correctly.
        # Initialize the first layer weights and biases by Xavier normal and zeros
        nn.init.xavier_normal_(first_linear.weight)
        nn.init.zeros_(first_linear.bias)
        # Scaling the first layer weights by kappa
        with torch.no_grad():
            first_linear.weight.data *= kappa
        # Initialize the rest of the layers with Xavier normal and zeros
        for layer in self.network[1:]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        # print(f"First layer weight scaled by kappa = {kappa:.2f}")

    def forward(self, x):
        location = x[:, 0:1]  # Extract location from the input tensor, keep shape [N, 1]
        raw = self.network(x)
        # Apply the hard constraint for the initial condition
        out = location * raw  # hard constraint: I(x=0) = I_o
        return out
