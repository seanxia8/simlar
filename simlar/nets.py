from collections import OrderedDict
import torch
from torch import nn
import numpy as np

class SineLayer(nn.Module):
    '''
    A linear layer with the sinusoidal activation function, a typical layer in siren
    '''

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        '''
        Parameters
        ----------
        in_features : int
            The number of input features
        out_features : int
            The number of outputs to be predicted
        bias : bool
            The bias argument for torch.nn.Linear
        is_first : bool
            If True, omega_0 is a frequency factor which simply multiplies the activations before the
            nonlinearity. If is_first=False, then the weights will be divided by omega_0 so as to keep
            the magnitude of activations constant, but boost gradients to the weight matrix (see
            supplement Sec. 1.5 in the original siren paper).
        omega_0 : float
            A multiplicative factor to the linear layer's output before applying sin function. For the
            intermediate layers, the weights are divided by omega_0. See the siren paper sec. 3.2 and
            the supplement Sec 1.5.
        '''
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        '''
        Custom weights initialization for the sine layer
        '''
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        '''
        For the model forward function
        '''
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        '''
        For visualization of activation distributions
        '''
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    '''
    A siren (whole) network implementation
    '''

    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.):
        '''
        Constructor

        Parameters
        ----------
        in_features : int
            The number of input features
        hidden_features : int
            The number of neurons in the intermediate layers
        hidden_layers : int
            The number of intermediate layers
        out_features : int
            The number of outputs to be predicted
        outermost_linear: bool
            If True, the final layer does not apply a sin activation function
        first_omega_0 : float
            The omega_0 parameter for the first SineLayer
        hidden_omega_0 : float
            The omega_0 parameter for the intermediate layers
        '''
        super().__init__()

        print(
            f'[Siren] {in_features} in => {out_features} out, hidden {hidden_features} features {hidden_layers} layers')
        print(f'        omega {first_omega_0} first {hidden_omega_0} hidden, the final layer linear {outermost_linear}')

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))


        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
            #add an activation layer to the final output to make positive values
            #self.net.append(nn.LeakyReLU(0.0001))

        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))


        self.net = nn.Sequential(*self.net)


    def forward(self, coords, clone=True):
        '''
        For the model forward computation.

        Parameters
        ----------
        coords : torch.Tensor
            The model input.
        clone : bool
            If True, allow to take derivatives w.r.t. input
        '''
        if clone:
            coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        '''
        Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!
        '''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__.__name__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__.__name__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
