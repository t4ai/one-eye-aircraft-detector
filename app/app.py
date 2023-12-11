import streamlit as st
import numpy as np
import PIL
from PIL import Image
from streamlit_image_select import image_select
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import pathlib
import PIL
import PIL.Image
import xml.etree.ElementTree as ET
import pybboxes as pbx
from pybboxes import BoundingBox
from pathlib import Path
import colorsys
import random

####################################################
# Support functions
####################################################

# helper function to generate random colors for class boxes
def generate_label_colors(count):
  colors = []
  for c in range(count):
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    colors.append(tuple([int(r), int(g), int(b)]))
  return colors

# helper function to run model inference
def run_inference(model, img_paths):
  return model.predict(img_paths)

# helper function to process result and return image with bbox overlays
def process_inference_result(result, class_colors):

  # setup label counts
  label_counts = {'class': [], 'count': []}
  # extract result objects
  img = result.orig_img
  dh, dw, _ = img.shape
  boxes = result.boxes.xywhn.tolist()
  labels = [int(label) for label in result.boxes.cls]
  conf = [float(label) for label in result.boxes.conf]

  # create image
  for i, bbox in enumerate(boxes):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    voc_box = pbx.convert_bbox([x, y, w, h], from_type="yolo", to_type="voc", image_size=(dw, dh))
    voc_x1 = voc_box[0]
    voc_y1 = voc_box[1]
    voc_x2 = voc_box[2]
    voc_y2 = voc_box[3]
    class_name = aircraft_lookup[classes[labels[i]]]
    cv2.rectangle(img, (voc_x1, voc_y1), (voc_x2, voc_y2), class_colors[labels[i]], 2)
    cv2.putText(img, class_name  + ' ' + str(round(conf[i], 2)), (voc_x1, voc_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, class_colors[labels[i]], 2)
    if class_name not in label_counts['class']:
        label_counts['class'].append(class_name)
        label_counts['count'].append(1)
    else:
        label_counts['count'][label_counts['class'].index(class_name)] += 1

  return img, label_counts

def get_detection_count_display(classes): 
  class_names = classes["class"]
  counts = classes["count"]
  if(len(classes)):
    disp_str = "[Aircraft detected] "
    for i, c in enumerate(class_names):
      disp_str += c + ": " + str(counts[i]) + "  " 
  else:
    disp_str = "[No aircraft detected]"
  return disp_str

def run_process_show(img_path):
  results = model(img_path)
  processed_image = process_inference_result(results[0], rand_class_colors)
  return processed_image

####################################################
# Setup model and class parameters
####################################################

# init model
model = YOLO("weights/best.pt")

# setup label classes
classes = ['A6', 'A17', 'A16', 'A15', 'A5', 'A20', 'A14', 'A12', 'A8', 'A2', 'A7', 'A18', 'A13', 'A4', 'A19', 'A1', 'A3', 'A10', 'A11', 'A9']

# setup mapping of class labels to real aircraft names
aircraft_names = ['SU-35', 'C-130', 'C-17', 'C-5', 'F-16', 'TU-160', 'E-3', 'B-52', 'P-3C', 'B-1B', 'E-8', 'TU-22', 'F-15', 'KC-135', 'F-22', 'FA-18', 'TU-95', 'KC-10', 'SU-34', 'SU-24']
aircraft_lookup = {}
for i in range(len(classes)):
    aircraft_lookup['A' + str(i+1)] = aircraft_names[i]

# generate bbox colors for each class
rand_class_colors = generate_label_colors(len(classes))

# logo
logo = cv2.imread("images/oneeye2.jpg")
logo_img = st.image(logo)

####################################################
# Main UX Loop
####################################################
img = image_select(
    label="Select an airbase",
    images=[
        cv2.imread("images/edwards.jpg"),
        cv2.imread("images/edwards2.jpg"),
        cv2.imread("images/buturlinovka2.jpg"),
        cv2.imread("images/engels.jpg"),
        cv2.imread("images/nellis1.jpg"),
        cv2.imread("images/littlerock.jpg"),
        cv2.imread("images/hmeimim.jpg"),
        cv2.imread("images/hill.jpg"),
    ],
    captions=["Edwards AFB 1", "Edwards AFB 2", "Buturlinovka District", "Engels", "Nellis AFB", "Little Rock AFB", "Hmiemim Syria", "Hill AFB"],
)

# process image through detector
status = st.empty()

# show un-classified image
big_img = st.image(img)

# show status message
status.write("Running OneEye detector...")

# process image through detector
img2, detection_labels = run_process_show(img)
big_img.image(img2)

status.write(get_detection_count_display(detection_labels))