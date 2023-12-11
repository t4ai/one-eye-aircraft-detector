![oneye](https://github.com/t4ai/one-eye-aircraft-detector/assets/132633661/9fe79103-630d-4d87-9bf4-506c50cb9493)

# OneEye: Military Aircraft Detection with Computer Vision
# AAI-521-Final
Final project for AAI-521: Computer Vision, University of San Diego MS-AAI.


**Demo App:** https://huggingface.co/spaces/t4ai/oneeye 


## Business Understanding (Hypothetical)
OneDefense, LLC is a defense contractor that provides high resolution satellite imagery with global coverage to the United States military and intelligence services.  The company has embarked on a strategy to provide AI-enrichment across its entire portfolio of imagery products in order to enable turnkey, high value use cases to customers.

With tensions rising in many theaters of operations, it has become difficult for human analysts to stay on top of key tasks, such as tracking the movement and readiness of military aircraft.  To help address this challenge, a key agency has requested OneDefense to develop a capability to monitor military airfields of key adversaries automatically using computer vision.  

Leveraging imagery collected on OneDefense satellite platforms, the application must meet the following objectives: 
Ingest and prepare satellite imagery for analysis 
Perform object detection and classification of key military aircraft located within these airfields.  
Provide counts of each aircraft type within the image frame

## Dataset

MAR20 dataset (https://www.kaggle.com/datasets/khlaifiabilel/military-aircraft-recognition-dataset/data) was used for transfer learning.  The selected dataset contained a total of 3,842 images labeled with 20 classes of aircraft types.  In total, 22,341 instances of aircraft were labeled (Military Aircraft Recognition Dataset, 2022)

## Data Preparation
### Image Augmentation and Resizing
  - Image resizing to 640x640 px (YOLO optimized)
  - Execute random horizontal flip and 90 degree rotate
  - Execute random brightness and contrast adjust

### Translate dataset from Pascal VOC to YOLO
  - YOLO model architecture requires dataset file structure
  - Label files translated from Pascal XML to flat txt
  - Bounding boxes translated from Pascal to YOLO
### Perform Train/Val/Test Split
  - Dataset had a train/test split specified in a manifest file (no val) but was not split in filesystem
  - Execute the split into YOLO format


## Model Training
### Model training:
  - Leverage Ultralytics framework
  - Hyperparameters largely set to defaults
  - 100 epochs on T4 GPU

### Experiments and Observations:
  - Initial training run uncovered low representation of many classes in train split (and low performance)
  - Second experiment executed with modified train/test split yielded much better results

### Model performance on test data:
  - YOLOv8 with transfer learning:
  - mAP50 score: 88
  - mAP-95 score: 66
