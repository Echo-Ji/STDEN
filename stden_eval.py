from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from lib.utils import load_graph_data
from model.stden_supervisor import STDENSupervisor

import numpy as np 
import torch 

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = STDENSupervisor(adj_mx=adj_mx, **supervisor_config)

        horizon = supervisor_config['model'].get('horizon')
        extract_latent = supervisor_config['model'].get('save_latent')
        supervisor.eval_more(dataset='test', 
                            save=args.save_pred, 
                            seq_len=np.arange(1, horizon+1, 1), 
                            extract_latent=extract_latent)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('-r', '--random_seed', type=int, default=2021, help="Random seed for reproduction.")
    parser.add_argument('--save_pred', action='store_true', help='Save the prediction.')
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    main(args)
