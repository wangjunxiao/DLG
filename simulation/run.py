import argparse
# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, 
                    default='configs/MNIST/2mnist.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')
args = parser.parse_args()
print(args.config)

import time
import importlib
import logging
importlib.reload(logging)
# Set logging
path = ('./LOG/'+time.strftime("%m-%d %Hh%Mm%Ss", time.localtime())+'.log').replace('\\','/')
logging.basicConfig(
    filename=path,
    filemode='a',
    format='%(message)s',
    level=logging.CRITICAL,
    datefmt='%H:%M:%S')

import config
import torch
import os
import server

# Set logging
#logging.basicConfig(
#    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation."""
    torch.cuda.empty_cache()

    # Read configuration file
    fl_config = config.Config(args.config)
    
    if os.path.exists(fl_config.paths.model + '/global'):
        os.remove(fl_config.paths.model + '/global')

    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config),
        "accavg": server.AccAvgServer(fl_config),
        "directed": server.DirectedServer(fl_config),
        "kcenter": server.KCenterServer(fl_config),
        "kmeans": server.KMeansServer(fl_config),
        "magavg": server.MagAvgServer(fl_config),
    }[fl_config.server]
    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Delete global model
    os.remove(fl_config.paths.model + '/global')


if __name__ == "__main__":
    main()
