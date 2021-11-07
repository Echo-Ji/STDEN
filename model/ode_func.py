import numpy as np
import torch
import torch.nn as nn

from lib import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerParams:
    def __init__(self, rnn_network: nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = nn.Parameter(torch.empty(*shape, device=device))
            nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter('{}_weight_{}'.format(self._type, str(shape)),
                                                 nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = nn.Parameter(torch.empty(length, device=device))
            nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter('{}_biases_{}'.format(self._type, str(length)),
                                                 biases)

        return self._biases_dict[length]

class ODEFunc(nn.Module):
    def __init__(self, num_units, latent_dim, adj_mx, gcn_step, num_nodes, 
                gen_layers=1, nonlinearity='tanh', filter_type="default"):
        """
        :param num_units: dimensionality of the hidden layers 
        :param latent_dim: dimensionality used for ODE (input and output). Analog of a continous latent state
        :param adj_mx:
        :param gcn_step:
        :param num_nodes:
        :param gen_layers: hidden layers in each ode func. 
        :param nonlinearity:
        :param filter_type: default
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """
        super(ODEFunc, self).__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        
        self._num_nodes = num_nodes
        self._num_units = num_units # hidden dimension
        self._latent_dim = latent_dim
        self._gen_layers = gen_layers
        self.nfe = 0
        
        self._filter_type = filter_type
        if(self._filter_type == "unkP"):
            ode_func_net = utils.create_net(latent_dim, latent_dim, n_units=num_units)
            utils.init_network_weights(ode_func_net)
            self.gradient_net = ode_func_net
        else:
            self._gcn_step = gcn_step
            self._gconv_params = LayerParams(self, 'gconv')
            self._supports = []
            supports = []
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
            
            for support in supports:
                self._supports.append(self._build_sparse_matrix(support))
    
    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        L = torch.sparse_coo_tensor(indices.T, L.data, L.shape, device=device)
        return L
    
    def forward(self, t_local, y, backwards = False):
        """
		Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

		t_local: current time point
		y: value at the current time point, shape (B, num_nodes * latent_dim)

        :return
        - Output: A `2-D` tensor with shape `(B, num_nodes * latent_dim)`.
        """
        self.nfe += 1
        grad = self.get_ode_gradient_nn(t_local, y)
        if backwards:
            grad = -grad
        return grad
    
    def get_ode_gradient_nn(self, t_local, inputs):
        if(self._filter_type == "unkP"):
            grad = self._fc(inputs)
        elif (self._filter_type == "IncP"):
            grad = - self.ode_func_net(inputs)
        else: # default is diffusion process
            # theta shape: (B, num_nodes * latent_dim)
            theta = torch.sigmoid(self._gconv(inputs, self._latent_dim, bias_start=1.0)) 
            grad = - theta * self.ode_func_net(inputs)
        return grad

    def ode_func_net(self, inputs):
        c = inputs
        for i in range(self._gen_layers):
            c = self._gconv(c, self._num_units)
            c = self._activation(c)
        c = self._gconv(c, self._latent_dim)
        c = self._activation(c)
        return c
    
    def _fc(self, inputs):
        batch_size = inputs.size()[0]
        grad = self.gradient_net(inputs.view(batch_size * self._num_nodes, self._latent_dim))
        return grad.reshape(batch_size, self._num_nodes * self._latent_dim) # (batch_size, num_nodes, latent_dim)
    
    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _gconv(self, inputs, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        # state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        # inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs.size(2)

        x = inputs
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._gcn_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)

                for k in range(2, self._gcn_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._gcn_step + 1  # Adds for x itself.
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])
