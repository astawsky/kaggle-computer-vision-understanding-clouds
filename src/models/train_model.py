# from general_imports import *
from os import environ, getenv
from os.path import basename
environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
# from src.data.preprocessing import load_data
from sklearn.model_selection import train_test_split
from logging import getLogger


from glob import glob
import cv2
from numpy import array, expand_dims
from pathlib import Path
from dotenv import load_dotenv
import hydra
from yaml import load, dump, safe_load
from tqdm import tqdm
from sklearn.model_selection import train_test_split

logger = getLogger()
load_dotenv()


# @hydra.main(config_path=getenv('MODELS_CONFIG_PATH'), config_name=getenv('MODELS_CONFIG_NAME'), version_base=None)
def run_all_in_config(config):
    inputs = config.copy()
    inputs['backbone'] = config['backbone'][0]  # config.backbone[0]
    # for config_combination in combination of paramters
    run_training_logic(inputs)

def run_training_logic(inputs):
    preprocess_input = sm.get_preprocessing(inputs['backbone'])

    #Resizing images is optional, CNNs are ok with large images
    SIZE_X = int(inputs['raw_image_height'] / 3) #Resize images (height  = X, width = Y)
    SIZE_Y = int(inputs['raw_image_width'] / 3)

    #Capture mask/label info as a list
    train_masks = [] 
    masks_filenames = []

    processed_binarymask_path = getenv('BINARY_MASK_PATH')

    # for directory_path in glob(processed_binarymask_path):
    for mask_path in tqdm(glob(str(Path(processed_binarymask_path) / Path("*.jpg")))):
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
        #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
        train_masks.append(mask)
        #train_labels.append(label)
        masks_filenames.append(basename(mask_path))

    #Convert list to array for machine learning processing          
    train_masks = array(train_masks)

    #Capture training image info as a list
    train_images = []

    processed_images_path = getenv('PROCESSED_IMAGES_PATH')
    # processed_images = glob(str(Path(processed_images_path) / Path("*.jpg")))

    # for directory_path in glob(processed_images_path):
    for filename in tqdm(masks_filenames):
        # logger.info(f'{count} of {len(processed_images)}: filename "{img_path}"', stdout=True)

        img_filename = filename.split('_')[0] + '.jpg'
        
        img_path = str(Path(processed_images_path) / Path(img_filename))
        
        #print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
            
    #Convert list to array for machine learning processing        
    train_images = array(train_images)
    x_train, x_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)

    # preprocess input
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    # define model
    model = sm.Unet(inputs['backbone'], encoder_weights=inputs['encoder_weights'])
    model.compile(
        inputs['model_compile_name'],
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    # print(model.summary())

    # fit model
    # if you use data generator use model.fit_generator(...) instead of model.fit(...)
    # more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,  # Todo: This should be a parameter in the config
        epochs=10,  # Todo: This should be a parameter in the config
        validation_data=(x_val, y_val),
    )

    model.save('first_model.h5')  # Todo: Sklearn with pipeline or MLflow

    # return [train_images, train_masks]


# BACKBONE = 'resnet34'  # todo: needs to be in config file
# preprocess_input = sm.get_preprocessing(BACKBONE)

# # # load your data
# # x_train, y_train, x_val, y_val = load_data(...)

# #Resizing images is optional, CNNs are ok with large images
# SIZE_X = 128 #Resize images (height  = X, width = Y)
# SIZE_Y = 128

# #Capture training image info as a list
# train_images = []

# for directory_path in glob.glob("/content/drive/My Drive/Colab Notebooks/data/membrane/train/image"):
#     for img_path in glob.glob(os.path.join(directory_path, "*.png")):
#         #print(img_path)
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
#         img = cv2.resize(img, (SIZE_Y, SIZE_X))
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         train_images.append(img)
#         #train_labels.append(label)
# #Convert list to array for machine learning processing        
# train_images = array(train_images)

# #Capture mask/label info as a list
# train_masks = [] 
# for directory_path in glob.glob("/content/drive/My Drive/Colab Notebooks/data/membrane/train/label"):
#     for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
#         mask = cv2.imread(mask_path, 0)       
#         mask = cv2.resize(mask, (SIZE_Y, SIZE_X))
#         #mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
#         train_masks.append(mask)
#         #train_labels.append(label)
# #Convert list to array for machine learning processing          
# train_masks = array(train_masks)

     

# #Use customary x_train and y_train variables
# X = train_images
# Y = train_masks
# Y = expand_dims(Y, axis=3) #May not be necessary.. leftover from previous code 
#   # As numpy array of cv2.imread objects




  

# # preprocess input
# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)

# # define model
# model = sm.Unet(BACKBONE, encoder_weights='imagenet')
# model.compile(
#     'Adam',
#     loss=sm.losses.bce_jaccard_loss,
#     metrics=[sm.metrics.iou_score],
# )
# print(model.summary())

# # fit model
# # if you use data generator use model.fit_generator(...) instead of model.fit(...)
# # more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
# model.fit(
#    x=x_train,
#    y=y_train,
#    batch_size=16,
#    epochs=100,
#    validation_data=(x_val, y_val),
# )

# model.save('/content/drive/My Drive/Colab Notebooks/data/membrane3000.h5')  # Todo: Sklearn with pipeline or MLflow


if __name__ == '__main__':
    import hydra

    configpath = Path(getenv('MODELS_CONFIG_PATH')) / Path(getenv('MODELS_CONFIG_NAME'))

    with open(configpath, 'r') as c:
        train_config = safe_load(c)

    # @hydra.main(config_path=getenv('MODELS_CONFIG_PATH'), config_name=getenv('MODELS_CONFIG_PATH'), version_base=None)
    run_all_in_config(train_config)


