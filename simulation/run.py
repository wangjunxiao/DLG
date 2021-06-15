import argparse
# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, 
                    default='configs/CIFAR-10/1cifar-10.json',
                    help='Federated learning configuration file.')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')
args = parser.parse_args()

import importlib
import logging
importlib.reload(logging)
# Set logging
logging.basicConfig(
    filename=__name__+'3.log',
    filemode='a',
    format='%(message)s',
    #format='[%(levelname)s][%(asctime)s]: %(message)s',
    level=logging.CRITICAL, 
    datefmt='%H:%M:%S')

import config
import os
import server

def main():
    """Run a federated learning simulation."""

    # Read configuration file
    fl_config = config.Config(args.config)

    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config),
        "accavg": server.AccAvgServer(fl_config),
        "magavg": server.MagAvgServer(fl_config),
    }[fl_config.server]
    fl_server.boot()

    # Run federated learning
    fl_server.run()

    # Delete global model
    os.remove(fl_config.paths.model + '/global')


if __name__ == "__main__":
    main()
