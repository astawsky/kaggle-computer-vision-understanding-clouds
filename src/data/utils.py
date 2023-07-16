from dotenv import load_dotenv
from os import getenv
from subprocess import run as run_bash_cmd

import hydra
from omegaconf import DictConfig

from pandas import read_csv, DataFrame

from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
import matplotlib.pyplot as plt
# from sklearn.model_selection import StratifiedKFold
# import tqdm

load_dotenv()

train_csv_path = getenv('KCVUC_TRAIN_CSV')
processed_data_path = getenv('KCVUC_PROCESSED_DATA')
config_path_env = getenv('CONFIG_PATH')
config_name_env = getenv('CONFIG_NAME')
train_images_path = getenv('KCVUC_TRAIN_IMAGES_PATH')

def if_exists_cmd_wrap(name_of_obj: str, cmd: str, obj_desc: str = 'Folder'):
    return f"""if [ {'-e' if obj_desc == 'File' else '-d'} {name_of_obj} ]; then
    echo '{obj_desc} {name_of_obj} already exists'
else
    {cmd}
fi"""

# @hydra.main(config_path=config_path_env, config_name=config_name_env, version_base=None)
def split_data_basedonoldscripts(config: DictConfig):
    # Define all the data processing parameters
    dp_params = config.data_processing

    # Read the training table
    df_train = read_csv(train_csv_path, dtype=dict(dp_params.train_csv_dtype_map))

    # Since the column Image_label encodes the filename and label separated by a single underscore
    df_train['Image'] = df_train.Image_Label.map(lambda v: v[:v.index('_')]).astype(dp_params.train_csv_dtype_map.Image)
    df_train['Label'] = df_train.Image_Label.map(lambda v: v[v.index('_')+1:]).astype(dp_params.train_csv_dtype_map.Label)
    df_train['label_index'] = df_train.Label.map(dp_params.label_mapping).astype(dp_params.train_csv_dtype_map.Label_index)

    # # Stratified means that we aim to maintain the same multi-variate distribution across splits and folds
    # skf = StratifiedKFold(n_splits=dp_params.skf_n_splits, shuffle=dp_params.skf_shuffle, random_state=dp_params.random_state if dp_params.skf_shuffle else None)

    # df_group = df_train.groupby('Image')
    # X=[]
    # y=[]
    # for i, (key, df) in tqdm.tqdm(enumerate(df_group), total=len(df_group)):
    #     X.append([i])
    #     ml = np.array([0,0,0,0])
    #     df = df.dropna()
    #     ml[np.array(df.LabelIndex)-1] = 1
    #     y.append(ml)
    #     image_ids.append(key)

    # # Run the splitting
    # skf.get_n_splits(X, y)

    # for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    #     print(f"Fold {i}:")
    #     print(f"  Train: index={train_index}")
    #     print(f"  Test:  index={test_index}")

    return df_train

# @hydra.main(config_path=config_path_env, config_name=config_name_env, version_base=None)
def create_image_label_structure(config: DictConfig, df_w_labels: DataFrame, rm_raw: bool = False):
    def create_bash_command():
        for label, label_data in df_w_labels.groupby('Label'):
            print(label, len(label_data))
        for label, label_data in df_w_labels.groupby('Label'):
            dir_to_store_images = processed_data_path + label
            mkdir_cmd = if_exists_cmd_wrap(dir_to_store_images, f"mkdir {dir_to_store_images}", "Folder")
            # mkdir_cmd = f"""if [-e {dir_to_store_images}]; then
            # echo "Folder {dir_to_store_images} already exists"
            # else
            # mkdir {dir_to_store_images}"""
            run_bash_cmd(mkdir_cmd, shell=True)

            label_images = ' '.join([train_images_path + image for image in label_data.Image.unique()])
            print()
            mv_files_cmd = f"mv {label_images} {dir_to_store_images}"
            run_bash_cmd(mv_files_cmd, shell=True)

        folders_to_create = [processed_data_path + label for label in df_w_labels.Label.unique()].join(' ')
        # First we must create the correct folders corresponding to 
        bash_cmd = f"""mkdir {folders_to_create}"""
        run_bash_cmd(bash_cmd)
        df_w_labels

    # First make the directory where they will be kept
    run_bash_cmd(if_exists_cmd_wrap(processed_data_path, f"mkdir {processed_data_path}", "Folder"), shell=True)
    create_bash_command()

    return 0


@hydra.main(config_path=config_path_env, config_name=config_name_env, version_base=None)
def run(config: DictConfig):
    df_train = split_data_basedonoldscripts(config=config)
    run_bash_cmd('env')
    create_image_label_structure(config=config, df_w_labels=df_train, rm_raw=False)


if __name__ == '__main__':
    # split_data()
    run()
