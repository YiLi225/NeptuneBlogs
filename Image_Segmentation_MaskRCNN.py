'''
ImageSegmentationProject
Mask R-CNN model image segmentation modeling and tuning 
'''
import os
import neptune
# Use case for image segmentation 
# Connect your script to Neptune
project = neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'),
                       project_qualified_name='katyl/ImageSegmentationProject') ## 'YourUserName/YourProjectName'

## How to track the weights and predictions in Neptune
npt_exp = project.create_experiment('implement-MaskRCNN-Neptune', 
                                    tags=['image segmentation', 'mask rcnn', 'keras', 'neptune'])

import sys
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
# Root directory of the project
ROOT_DIR = os.path.abspath(r"..\entty\Desktop\0.Article_Pitching\Image_Segmentation\aktwelve_mask_rcnn")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
from mrcnn.model import log

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
npt_exp.send_text('Model Config Pars', str(config.to_dict()))


# Create model object in inference mode.
def runMaskRCNN(modelConfig, imagePath, MODEL_DIR=MODEL_DIR, COCO_MODEL_PATH=COCO_MODEL_PATH):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=modelConfig)
    # Load weights trained on MS-COCO
    ## Exclude the last layers because they require a matching number of classes
    # exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
    #          "mrcnn_bbox", "mrcnn_mask"]    
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    
    ## img = load_img(r'C:\Users\entty\Desktop\0.Article_Pitching\Image_Segmentation\teddybear.jpg')
    image_path = imagePath    
    img = load_img(image_path)
    img = img_to_array(img)
    
    # make prediction
    results = model.detect([img], verbose=1)
    # get dictionary for first prediction
    modelOutput = results[0]
    
    return modelOutput, img


#### CUSTOMIZE MODEL CONFIG ####
class CustomConfig(coco.CocoConfig):
    """Configuration for inference on the teddybear image.
    Derives from the base Config class and overrides values specific
    to the teddybear image.
    """
    # configuration name
    NAME = "customized"
    # number of classes: +1 for background
    NUM_CLASSES = 1 + 80 
    
    # batch size = 1 for one image
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # how many steps in one epoch
    STEPS_PER_EPOCH = 500
    # min. probability for segmentation
    DETECTION_MIN_CONFIDENCE = 0.71
    
    # learning rate, momentum and weight decay for regularization
    LEARNING_RATE = 0.06      
    LEARNING_MOMENTUM = 0.7
    WEIGHT_DECAY = 0.0002
  
    VALIDATION_STEPS = 30
   
config = CustomConfig()
## Log current config to Neptune
npt_exp.send_text('Model Config Pars', str(config.to_dict()))



###### Model inference:
# cur_image_path = r'C:\Users\entty\Desktop\0.Article_Pitching\Image_Segmentation\monks_and_dogs.jpg'    
cur_image_path = r'C:\Users\entty\Desktop\0.Article_Pitching\Image_Segmentation\teddybear.jpg'    
image_results, img = runMaskRCNN(modelConfig=config, imagePath=cur_image_path)

# Create model object in inference mode.
# show image with bounding boxes, masks, class labels and scores
fig_images, cur_ax = plt.subplots(figsize=(15, 15))
display_instances(img, image_results['rois'], image_results['masks'], 
                  image_results['class_ids'], class_names, image_results['scores'], ax=cur_ax)

# Log Predicted images to Neptune
npt_exp.log_image('Predicted Image', fig_images)


##########===========Review model stats ================================#######
# Show stats of trainable model weights    
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

LAYER_TYPES = ['Conv2D']
# Get layers
layers = model.get_trainable_layers()
layers = list(filter(lambda l: l.__class__.__name__ in LAYER_TYPES, layers))
print(f'Total layers = {len(layers)}')

## select a subset of layers
layers = layers[:5]  

# Display Histograms
fig, ax = plt.subplots(len(layers), 2, figsize=(10, 3*len(layers)+10),  
                       gridspec_kw={"hspace":1})

for l, layer in enumerate(layers):
    weights = layer.get_weights()
    for w, weight in enumerate(weights):
        tensor = layer.weights[w]
        ax[l, w].set_title(tensor.name)
        _ = ax[l, w].hist(weight[w].flatten(), 50)

npt_exp.log_image('Model_Weights', fig)









