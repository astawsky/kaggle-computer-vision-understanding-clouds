import glob
import cv2
import os
import numpy as np
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from dotenv import load_dotenv
from os import getenv
from os.path import join
import hydra
from omegaconf import DictConfig
from pandas import read_csv, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt


load_dotenv()

train_csv_path = getenv('KCVUC_TRAIN_CSV')
processed_data_path = getenv('KCVUC_PROCESSED_DATA')
config_path_env = getenv('CONFIG_PATH')
config_name_env = getenv('CONFIG_NAME')
train_images_path = getenv('KCVUC_TRAIN_IMAGES_PATH')

def separate_images_and_labels(config: DictConfig):
    # Read the training table
    df_train = read_csv(train_csv_path, dtype=dict(config.train_csv_dtype_map))

    # Since the column Image_label encodes the filename and label separated by a single underscore
    df_train['image'] = df_train.Image_Label.map(lambda v: v[:v.index('_')]).astype(config.train_csv_dtype_map.Image)
    df_train['label'] = df_train.Image_Label.map(lambda v: v[v.index('_')+1:]).astype(config.train_csv_dtype_map.Label)

    return df_train  #.convert_dtypes()  # Infers all dtypes and converts them


def rle2mask(mask_rle: str, shape: tuple, label: int = 1):

    # Blank array-mask to fill in
    return_mask = np.zeros(shape[0]*shape[1])

    # If it is Nan then we return the blank mask
    if isinstance(mask_rle, float):
        if np.isnan(mask_rle):
            print(f'Should be Nan: {mask_rle}')
            return return_mask.reshape((shape[1], shape[0])).T

    # Parse the string RLE
    mask_rle = np.fromstring(mask_rle, dtype=int, sep=' ')

    # Must be even number of numbers in total
    assert len(mask_rle) % 2 == 0

    # Define the target pixels
    start_positions = np.array(mask_rle)[0::2] - 1
    run_lengths = np.array(mask_rle)[1::2]

    # Input the target pixels to the mask
    for sp, rl in zip(start_positions, run_lengths):
        return_mask[sp:(sp+rl)] = 1

    # reshape the length-wise mask
    return return_mask.reshape((shape[1], shape[0])).T


@hydra.main(config_path=config_path_env, config_name=config_name_env, version_base=None)
def run(config: DictConfig):
    return None


@hydra.main(config_path=config_path_env, config_name=config_name_env, version_base=None)
def plot_train_images_with_mask(config: DictConfig, filename_list: list = []):

    # The relevant config file
    pp_config = config.preprocessing

    # shape raw images come in
    raw_image_shape = (pp_config.raw_image_height, pp_config.raw_image_width)

    # Label parsing
    df_train = separate_images_and_labels(config=pp_config)
    
    # Show either specified training files or all
    files_to_show = filename_list if len(filename_list) > 0 else range(len(df_train))

    for i in files_to_show:
        encoded_mask = df_train.loc[i, 'EncodedPixels']
        image_filename = df_train.loc[i, 'image']
        label = df_train.loc[i, 'label']

        # img = cv2.imread(f'data/raw/train_images/{image_filename}')
        img = cv2.imread(join(train_images_path, image_filename))

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


if __name__ == '__main__':
    plot_train_images_with_mask()
    # run()



