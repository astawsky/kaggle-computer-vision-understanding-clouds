import glob
import cv2
import os
import numpy as np
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from dotenv import load_dotenv
from os import getenv
import hydra
from omegaconf import DictConfig
from pandas import read_csv, DataFrame


load_dotenv()

train_csv_path = getenv('KCVUC_TRAIN_CSV')
processed_data_path = getenv('KCVUC_PROCESSED_DATA')
config_path_env = getenv('CONFIG_PATH')
config_name_env = getenv('CONFIG_NAME')
train_images_path = getenv('KCVUC_TRAIN_IMAGES_PATH')

def separate_images_and_labels(config: DictConfig):
    # Define all the data processing parameters

    # Read the training table
    df_train = read_csv(train_csv_path, dtype=dict(config.train_csv_dtype_map))

    # Since the column Image_label encodes the filename and label separated by a single underscore
    df_train['image'] = df_train.Image_Label.map(lambda v: v[:v.index('_')]).astype(config.train_csv_dtype_map.Image)
    df_train['label'] = df_train.Image_Label.map(lambda v: v[v.index('_')+1:]).astype(config.train_csv_dtype_map.Label)
    # df_train['label_index'] = df_train.Label.map(config.label_mapping)#.astype(config.train_csv_dtype_map.Label_index)

    return df_train  #.convert_dtypes()  # Infers all dtypes and converts them


def rle2mask(mask_rle: str, shape: tuple, label: int = 1):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction


def check_image_and_mask(image, mask):
    print(image.shape)
    print(mask.shape)

    # import matplotlib.pyplot as plt
    # plt.


@hydra.main(config_path=config_path_env, config_name=config_name_env, version_base=None)
def run(config: DictConfig):
    dp_config = config.data_processing
    raw_image_shape = (dp_config.raw_image_height, dp_config.raw_image_width)
    df_train = separate_images_and_labels(config=dp_config)
    decoded_mask = rle2mask(mask_rle=df_train.loc[0, 'EncodedPixels'], shape=raw_image_shape, label=1) # todo: put 1400 and 2100 as env variables of pixel height and width of raw images
    # df_train.EncodedPixels.map()

    vis2 = cv2.cvtColor(decoded_mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow('mask', vis2)
    cv2.waitKey()
    print('end of run')


if __name__ == '__main__':
    run()



