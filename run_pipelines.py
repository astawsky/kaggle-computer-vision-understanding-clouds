from src.models.train_model import train_model
import hydra
from dotenv import load_dotenv

load_dotenv()


@hydra.main(config_path='/src/conf.yaml', config_name=config_name_env, version_base=None)
def train_pipeline():
    # Based on config available

    



if __name__ == '__main__':
    run()
