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

MONGODB_USER = "athena"
MONGODB_PASSWORD = "Noo8naiMaupeob3IbieF4pah"
MONGODB_HOST = "10.20.40.199"
MONGODB_PORT = "27017"
MONGODB_NAME = "athena"
MONGODB_COLLECTION = "athena-label"

connection_string = "mongodb://{}:{}@{}:{}/{}?authSource=admin&authMechanism=SCRAM-SHA-1&replicaset=marketplace-prod".\
                    format(MONGODB_USER, MONGODB_PASSWORD, MONGODB_HOST, MONGODB_PORT, MONGODB_NAME)

client = MongoClient(connection_string)
db = client.athena
collection = db['athena-label']
prefix = "https://salt.tikicdn.com/"
ANNOTATION_DIR = "./train/annotations/"
IMAGE_DIR = "./train/images/"
TEST_DIR = "./test/"

f_label = open("labels.txt", "w")
labels = []

def create_base_path(url):
    base_path = ""
    if url.startswith("http"):
        base_path = ""
    elif url.startswith("product") or url.startswith("tmp"):
        base_path = "ts/"
    else:
        base_path = "media/catalog/product/"

    return base_path

if not exists(ANNOTATION_DIR):
    makedirs(ANNOTATION_DIR)

if not exists(IMAGE_DIR):
    makedirs(IMAGE_DIR)

if not exists(TEST_DIR):
    makedirs(TEST_DIR)

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
brand_images_count = {}
brand_list = ["Apple", "Kingston", "Adidas", "LEGO", "Calvin Klein",
              "Bosch", "Guess", "Puma", "Gucci", "Rayban", "Nike", "Levi's",
              "Disney", "iBasic", "3M", "Hugo Boss", "Prada"]
for i, obj in enumerate(collection.find()):
    if obj["annotation"] and (obj["class_name"].strip() in brand_list):
        try:
            if "jpg" not in obj["image_url"] or "png" not in obj["image_url"]:
                continue

            brand_images_count.setdefault(obj["class_name"], 0)
            brand_images_count[obj["class_name"]] += 1
            if brand_images_count[obj["class_name"]] > 20:
                image_save_dir = IMAGE_DIR
                annotation_save_dir = ANNOTATION_DIR
            else:
                image_save_dir = TEST_DIR
                annotation_save_dir = TEST_DIR

            category_id_to_name[obj["class_id"]] = obj["class_name"]
            url = prefix + create_base_path(obj["image_url"]) + obj["image_url"]
            labels.append(obj["class_name"])
            img_dir = image_save_dir + str(cnt) + '.jpg'

            with open(img_dir, 'wb') as handle:
                img = requests.get(url).content
                handle.write(img)

            img = cv2.imread(img_dir)
            height, width, channels = img.shape

            coors_tmp = obj["annotation"][0]["values"]
            coors = [[width * coor[0], height * coor[1], width * coor[2], height * coor[3]] for coor in coors_tmp]

            export_xml(str(cnt) + ".jpg",
                       annotation_save_dir + str(cnt) + ".xml",
                       [obj["class_name"]] * len(coors),
                       img,
                       coors)

            annotations = {'image': img,
                           'bboxes': coors,
                           'category_id': [obj["class_id"]] * len(coors)}

            aug = get_aug([VerticalFlip(p=1)])
            augmented = aug(**annotations)
            '''
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

            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from PIL import Image
            import numpy as np

            im = np.array(Image.open(IMAGE_DIR + str(cnt) + "_VerticalFlip.jpg"), dtype=np.uint8)
            # Create figure and axes
            fig, ax = plt.subplots(1)
            # Display the image
            ax.imshow(im)
            # Create a Rectangle patch
            rect = patches.Rectangle((augmented['bboxes'][0][0], augmented['bboxes'][0][1]), augmented['bboxes'][0][2] - augmented['bboxes'][0][0], augmented['bboxes'][0][3] - augmented['bboxes'][0][1],
                                     linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.show()

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
            '''
            cnt += 1
        except ValueError:
            print(url, img_dir)

for label in set(labels):
    f_label.write(label)
    f_label.write("\n")

f_label.close()