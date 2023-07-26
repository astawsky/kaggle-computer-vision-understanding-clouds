from src.data.preprocessing import rle2mask, image_label_parsing
from cv2 import imread
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from os import getenv
from os.path import join
import matplotlib.pyplot as plt

load_dotenv()

train_csv_path = getenv('KCVUC_TRAIN_CSV')
processed_data_path = getenv('KCVUC_PROCESSED_DATA')
config_path_env = getenv('CONFIG_PATH')
config_name_env = getenv('CONFIG_NAME')
train_images_path = getenv('KCVUC_TRAIN_IMAGES_PATH')


@hydra.main(config_path=config_path_env, config_name=config_name_env, version_base=None)
def plot_train_images_with_mask(config: DictConfig, filename_list: list = []):

    # The relevant config file
    pp_config = config.preprocessing

    # shape raw images come in
    raw_image_shape = (pp_config.raw_image_height, pp_config.raw_image_width)

    # Label parsing
    df_train = image_label_parsing(config=pp_config)

    # Show either specified training files or all
    files_to_show = filename_list if len(filename_list) > 0 else range(len(df_train))

    for i in files_to_show:
        encoded_mask = df_train.loc[i, 'EncodedPixels']
        image_filename = df_train.loc[i, 'image']
        label = df_train.loc[i, 'label']

        img = imread(join(train_images_path, image_filename))

        decoded_mask = rle2mask(mask_rle=encoded_mask, shape=raw_image_shape, label=1) * 255 # todo: put 1400 and 2100 as env variables of pixel height and width of raw images

        # img = mpimg.imread(f'data/raw/train_images/{image_filename}')
        # img = mpimg.imread(f'data/raw/train_images/{image_filename}')
        fig, ax = plt.subplots(1, 1, dpi=150)
        ax.imshow(img)
        ax.imshow(decoded_mask, cmap='gray', alpha=0.5)
        plt.title(f'{label} label in \'{image_filename}\'')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
        plt.close()