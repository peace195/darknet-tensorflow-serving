from pymongo import MongoClient
from os.path import exists
from os import makedirs, environ
import requests
import xml.etree.cElementTree as ET
import cv2
from urllib.request import urlopen
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose
)

client = MongoClient('10.20.30.48', 27017)
db = client.athena
collection = db.athena_label
prefix = "https://guat.tikicdn.com/ts/"
ANNOTATION_DIR = "./data/annotations/"
IMAGE_DIR = "./data/images/"

f_label = open("labels.txt", "w")
labels = []

if not exists(ANNOTATION_DIR):
    makedirs(ANNOTATION_DIR)

if not exists(IMAGE_DIR):
    makedirs(IMAGE_DIR)

def export_xml(img_name, xml_name, class_name, img, coors):
    height, width, channels = img.shape
    root = ET.Element("annotation")

    ET.SubElement(root, "filename").text = img_name
    size = ET.SubElement(root, "size")

    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(channels)

    for i in range(len(coors)):
        object = ET.SubElement(root, "object")
        ET.SubElement(object, "name").text = class_name[i]
        bounding_box = ET.SubElement(object, "bndbox")
        ET.SubElement(bounding_box, "xmin").text = str(int(coors[i][0]))
        ET.SubElement(bounding_box, "ymin").text = str(int(coors[i][1]))
        ET.SubElement(bounding_box, "xmax").text = str(int(coors[i][2]))
        ET.SubElement(bounding_box, "ymax").text = str(int(coors[i][3]))

    tree = ET.ElementTree(root)
    tree.write(xml_name)

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['category_id']})

cnt = 0
category_id_to_name = {}
for i, obj in enumerate(collection.find()):
    if (obj["class_id"] in [18802, 17827, 48462, 40780, 25422] and obj["annotation"]):
        category_id_to_name[obj["class_id"]] = obj["class_name"]
        url = prefix + obj["image_url"]
        labels.append(obj["class_name"])
        img_dir = IMAGE_DIR + str(cnt) + '.jpg'
        with open(img_dir, 'wb') as handle:
            img = requests.get(url).content
            handle.write(img)

        img = cv2.imread(img_dir)
        height, width, channels = img.shape
        coors_tmp = obj["annotation"][0]["values"]
        coors = [[width * coor[0], height * coor[1], width * coor[2], height * coor[3]] for coor in coors_tmp]
        export_xml(str(cnt) + ".jpg",
                   ANNOTATION_DIR + str(cnt) + ".xml",
                   [obj["class_name"]] * len(coors),
                   img,
                   coors)

        # Data augmentation
        annotations = {'image': img,
                       'bboxes': coors,
                       'category_id': [obj["class_id"]] * len(coors)}
        aug = get_aug([VerticalFlip(p=1)])
        augmented = aug(**annotations)
        cv2.imwrite(IMAGE_DIR + str(cnt) + "_VerticalFlip.jpg", augmented['image'])
        export_xml(str(cnt) + "_VerticalFlip.jpg",
                   ANNOTATION_DIR + str(cnt) + "_VerticalFlip.xml",
                   [category_id_to_name[id] for id in augmented['category_id']],
                   augmented['image'],
                   augmented['bboxes'])

        aug = get_aug([HorizontalFlip(p=1)])
        augmented = aug(**annotations)
        cv2.imwrite(IMAGE_DIR + str(cnt) + "_HorizontalFlip.jpg", augmented['image'])
        export_xml(str(cnt) + "_HorizontalFlip.jpg",
                   ANNOTATION_DIR + str(cnt) + "_HorizontalFlip.xml",
                   [category_id_to_name[id] for id in augmented['category_id']],
                   augmented['image'],
                   augmented['bboxes'])

        aug = get_aug([VerticalFlip(p=1)])
        augmented = aug(**annotations)
        cv2.imwrite(IMAGE_DIR + str(cnt) + "_VerticalFlip.jpg", augmented['image'])
        export_xml(str(cnt) + "_VerticalFlip.jpg",
                   ANNOTATION_DIR + str(cnt) + "_VerticalFlip.xml",
                   [category_id_to_name[id] for id in augmented['category_id']],
                   augmented['image'],
                   augmented['bboxes'])

        cnt += 1

for label in set(labels):
    f_label.write(label)
    f_label.write("\n")

f_label.close()