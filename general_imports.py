
# Will be needed for defining the config file
import hydra
from omegaconf import DictConfig

# Useful
from os.path import join

# Load the environmental variables
from dotenv import load_dotenv
from os import getenv, environ

load_dotenv()

# Bring them into the python workspace
config_path_env = getenv('CONFIG_PATH')
config_name_env = getenv('CONFIG_NAME')
