from pymongo import MongoClient
from os.path import exists
from os import makedirs, environ
import requests
import xml.etree.cElementTree as ET
import cv2

from pprint import pprint
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

cnt = 0
for i, obj in enumerate(collection.find()):
    if (obj["class_id"] in [18802, 17827, 48462, 40780, 25422] and obj["annotation"]):
        url = prefix + obj["image_url"]
        labels.append(obj["class_name"])
        img_dir = IMAGE_DIR + str(cnt) + '.jpg'
        with open(img_dir, 'wb') as handle:
            img = requests.get(url).content
            handle.write(img)

        coors = obj["annotation"][0]["values"][0]
        img = cv2.imread(img_dir)
        height, width, channels = img.shape
        root = ET.Element("annotation")

        ET.SubElement(root, "filename").text = str(cnt) + '.jpg'
        size = ET.SubElement(root, "size")
        object = ET.SubElement(root, "object")

        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(channels)

        ET.SubElement(object, "name").text = obj["class_name"]

        bounding_box = ET.SubElement(object, "bndbox")

        ET.SubElement(bounding_box, "xmin").text = str(int(width * coors[0]))
        ET.SubElement(bounding_box, "ymin").text = str(int(height * coors[1]))
        ET.SubElement(bounding_box, "xmax").text = str(int(width * coors[2]))
        ET.SubElement(bounding_box, "ymax").text = str(int(height * coors[3]))

        tree = ET.ElementTree(root)
        tree.write(ANNOTATION_DIR + str(cnt) + ".xml")
        cnt += 1

for label in set(labels):
    f_label.write(label)
    f_label.write("\n")

f_label.close()