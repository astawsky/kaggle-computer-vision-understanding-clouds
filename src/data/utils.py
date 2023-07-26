from  general_imports import *
from os.path import basename
from glob import glob


processed_data_path = getenv('KCVUC_PROCESSED_DATA')
binary_mask_path = getenv('BINARY_MASK_PATH')


def mask_filenaming_convention(image_filename, label, data_path: str = processed_data_path, extension: str = '.jpg'):  # Todo: Have this .jpg string be in the config file
    # Make sure we are given the name and extension of file
    assert '.' in image_filename

    mask_filename = f"{image_filename.split('.')[0]}_{label}_0"  # All mask filenames will be of this format
    full_mask_path = join(data_path, mask_filename)

    # To test we need to create a folder with files 8902.jpg and 8902_flower.jpg and 8902_flower_1.jpg, etc...
    masks_of_same_label = glob(full_mask_path)

    if len(masks_of_same_label) > 0:
        # parse the index of latest addition
        current_label_index = max([int(basename(mask.split('.')[0]).split('_')[-1]) for mask in glob(full_mask_path)]) + 1

        # Update return variables
        mask_filename = f"{image_filename.split('.')[0]}_{label}_{current_label_index}"
        full_mask_path = join(data_path, mask_filename)

    return [mask_filename, full_mask_path]


def load_data():
    
    return


