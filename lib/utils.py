import logging
import numpy as np
import os
import time
import pickle
import scipy.sparse as sp
import sys
# import tensorflow as tf
import torch 
import torch.nn as nn

from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def get_log_dir(kwargs):
    log_dir = kwargs['train'].get('log_dir')
    if log_dir is None:
        batch_size = kwargs['data'].get('batch_size')

        filter_type = kwargs['model'].get('filter_type')
        gcn_step = kwargs['model'].get('gcn_step')
        horizon = kwargs['model'].get('horizon')
        latent_dim = kwargs['model'].get('latent_dim')
        n_traj_samples = kwargs['model'].get('n_traj_samples')
        ode_method = kwargs['model'].get('ode_method')

        seq_len = kwargs['model'].get('seq_len')
        rnn_units = kwargs['model'].get('rnn_units')
        recg_type = kwargs['model'].get('recg_type')
        
        if filter_type == 'unkP':
            filter_type_abbr = 'UP'
        elif filter_type == 'IncP':
            filter_type_abbr = 'NV'
        else:
            filter_type_abbr = 'DF'
        
        
        run_id = 'STDEN_%s-%d_%s-%d_L-%d_N-%d_M-%s_bs-%d_%d-%d_%s/' % (
            recg_type, rnn_units, filter_type_abbr, gcn_step, latent_dim, n_traj_samples, ode_method, batch_size, seq_len, horizon, time.strftime('%m%d%H%M%S'))
        base_dir = kwargs.get('log_base_dir')
        log_dir = os.path.join(base_dir, run_id)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def load_dataset(dataset_dir, batch_size, val_batch_size=None, **kwargs):
    if('BJ' in dataset_dir):
        data = dict(np.load(os.path.join(dataset_dir, 'flow.npz'))) # convert readonly NpzFile to writable dict Object 
        for category in ['train', 'val', 'test']:
            data['x_' + category] = data['x_' + category] #[..., :4] # ignore the time index
    else:
        data = {}
        for category in ['train', 'val', 'test']:
            cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
            data['x_' + category] = cat_data['x']
            data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'].mean(), std=data['x_train'].std()) # 第0维是要预测的量，但是第1维是什么呢？
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], val_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], val_batch_size, shuffle=False)
    data['scaler'] = scaler

    return data


def load_graph_data(pkl_filename):
    adj_mx = np.load(pkl_filename)
    return adj_mx

def graph_grad(adj_mx):
    """Fetch the graph gradient operator."""
    num_nodes = adj_mx.shape[0]

    num_edges = (adj_mx > 0.).sum()
    grad = torch.zeros(num_nodes, num_edges)
    e = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mx[i, j] == 0:
                continue

            grad[i, e] = 1.
            grad[j, e] = -1.
            e += 1
    return grad

def init_network_weights(net, std = 0.1):
    """
    Just for nn.Linear net.
    """
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)

def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim//2
    
    res = data[..., :last_dim], data[..., last_dim:]
    return res

def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()

def create_net(n_inputs, n_outputs, n_layers = 0, 
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)