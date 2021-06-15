import importlib
import logging
importlib.reload(logging)
logging.basicConfig( 
    level=logging.INFO, 
    datefmt='%H')

logging.info("test logging here")
