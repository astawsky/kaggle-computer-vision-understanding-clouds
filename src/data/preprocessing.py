from os import environ, getenv, mkdir
environ["SM_FRAMEWORK"] = "tf.keras"
import numpy as np
from dotenv import load_dotenv
from os.path import join, exists
from pandas import read_csv
import hydra
from omegaconf import DictConfig
from cv2 import imwrite
import logging
from pandas import DataFrame
from glob import glob
from shutil import copyfile

from os.path import basename
from glob import glob

from typing import Tuple


processed_data_path = getenv('KCVUC_PROCESSED_DATA')
binary_mask_path = getenv('BINARY_MASK_PATH')


def mask_filenaming_convention(image_filename, label, data_path: str = processed_data_path, extension: str = '.jpg'):  # Todo: Have this .jpg string be in the config file
    # Make sure we are given the name and extension of file
    assert '.' in image_filename

    mask_filename = f"{image_filename.split('.')[0]}_{label}_0{extension}"  # All mask filenames will be of this format
    full_mask_path = join(data_path, mask_filename)

    # To test we need to create a folder with files 8902.jpg and 8902_flower.jpg and 8902_flower_1.jpg, etc...
    masks_of_same_label = glob(full_mask_path)

    if len(masks_of_same_label) > 0:
        # parse the index of latest addition
        current_label_index = max([int(basename(mask.split('.')[0]).split('_')[-1]) for mask in glob(full_mask_path)]) + 1

        # Update return variables
        mask_filename = f"{image_filename.split('.')[0]}_{label}_{current_label_index}{extension}"
        full_mask_path = join(data_path, mask_filename)

    return [mask_filename, full_mask_path]

logger = logging.getLogger(str(__name__))  # Todo: specify logging configurations

load_dotenv()

train_csv_path = getenv('KCVUC_TRAIN_CSV')
processed_data_path = getenv('KCVUC_PROCESSED_DATA')
config_path_env = getenv('CONFIG_PATH')
config_name_env = getenv('CONFIG_NAME')
train_images_path = getenv('KCVUC_TRAIN_IMAGES_PATH')
processed_images_path = getenv('PROCESSED_IMAGES_PATH')

def image_label_parsing(config: DictConfig):
    # Read the training table
    df_train = read_csv(train_csv_path, dtype=dict(config.train_csv_dtype_map))

    logger.info(f'Read training CSV from {train_csv_path}.')

    # Since the column Image_label encodes the filename and label separated by a single underscore
    df_train['image'] = df_train.Image_Label.map(lambda v: v[:v.index('_')]).astype(config.train_csv_dtype_map.Image)
    df_train['label'] = df_train.Image_Label.map(lambda v: v[v.index('_')+1:]).astype(config.train_csv_dtype_map.Label)

    logger.info(f'Parsed image and labels into columns.\n{df_train.info}')

    return df_train  #.convert_dtypes()  # Infers all dtypes and converts them


def rle2mask(mask_rle: str, shape: tuple, seg_ind: int = 255) -> Tuple[np.array, bool]:
    """Converts (De-encodes) a 1D run-length array of the segmentation mask to a 2D binary mask.

    Args:
        mask_rle (str): 1D run-length encoded str with space delimiter.
        shape (tuple): Shape of image encoding corresponds to.
        seg_ind (int, optional): The non-zero value that indicates a segment. Defaults to 255 (b/c of grayscale).

    Returns:
        np.array: 2D binary segmentation mask.
    """

    # Blank array-mask to fill in
    return_mask = np.zeros(shape[0]*shape[1])

    # If it is Nan then we return the blank mask
    if isinstance(mask_rle, float):
        if np.isnan(mask_rle):
            logger.info(f'Blank mask (should be a Nan):\n{mask_rle}')
            return [return_mask.reshape((shape[1], shape[0])).T, True]
        else:
            logger.info(f'Mask is not Nan but is a float:\n{mask_rle}')
            raise ValueError

    # Parse the string RLE
    mask_rle = np.fromstring(mask_rle, dtype=int, sep=' ')

    # Must be even number of numbers in total
    assert len(mask_rle) % 2 == 0

    # Define the target pixels
    start_positions = np.array(mask_rle)[0::2] - 1
    run_lengths = np.array(mask_rle)[1::2]

    # Input the target pixels to the mask
    for sp, rl in zip(start_positions, run_lengths):
        return_mask[sp:(sp+rl)] = seg_ind

    # reshape the length-wise mask
    return [return_mask.reshape((shape[1], shape[0])).T, False]


def save_decoded_masks(df_train: DataFrame, raw_image_shape: tuple) -> None:
    # Create the folder for processed data
    if not exists(processed_data_path):
        mkdir(processed_data_path)
        logger.info(f'Created processed data path: {processed_data_path}')

    # write all masks and labels even if they are blank
    for i in range(len(df_train)):
        encoded_mask = df_train.loc[i, 'EncodedPixels']
        image_filename = df_train.loc[i, 'image']
        label = df_train.loc[i, 'label']

        # Decode the mask
        decoded_mask, is_empty_mask = rle2mask(mask_rle=encoded_mask, shape=raw_image_shape)

        # # Don't save empty masks
        # if is_empty_mask:
        #     continue

        # Get the filename for the mask
        decoded_mask_filename, full_mask_path = mask_filenaming_convention(image_filename=image_filename, label=label, data_path=processed_data_path, extension='.jpg')  # Todo: put extension as argument and config

        # Save to extension
        imwrite(full_mask_path, decoded_mask)
        logger.info(f'Saved the decoded mask: {decoded_mask_filename}.')


def copy_raw_images2processed(config: DictConfig) -> None:
    # Create the folder for processed image data
    if not exists(processed_images_path):
        mkdir(processed_images_path)
        logger.info(f'Created processed data path: {processed_images_path}')

    # Get the filenames of all the raw images
    list_of_jpg_images = glob(join(train_images_path, '*.jpg'))

    for im in list_of_jpg_images:
        im = im.split('/')[-1]

        # Specify the full path
        source_file = join(train_images_path, im)
        dest_file = join(processed_images_path, im)

        # copy the file to processed
        copyfile(source_file, dest_file)
        logger.info(f'Copied the file {source_file} to {dest_file} succesfully.')


def data_augmentation():
    # Move the given training data to processed
    return


def split_data(config: DictConfig):
    return


@hydra.main(config_path=config_path_env, config_name=config_name_env, version_base=None)
def run(config: DictConfig):
    pre_config = config.preprocessing

    # shape raw images come in
    raw_image_shape = (pre_config.raw_image_height, pre_config.raw_image_width)

    # Read the data and parse the images and labels
    df_train = image_label_parsing(pre_config)

    # Decode the RLE masks and save them
    save_decoded_masks(df_train, raw_image_shape)

    # Augment the training data
    data_augmentation()

    # Have all the data in the processed folder ready to be loaded
    copy_raw_images2processed(pre_config)


if __name__ == '__main__':
    run()
