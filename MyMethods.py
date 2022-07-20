import pyvips
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from sklearn.model_selection import StratifiedKFold
import numpy as np
import cv2
import pickle
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import openpyxl
import re
import os
import ast
import random
from torch.utils.data import Dataset
from torchvision import models
import torch
import torch.nn as nn
import OtherMethods


# ---------- Create background mask ----------

# Returns a mask highlighting only the purple colours
def HSV_thresh(path, name, level):
    # Read image with width imgWidth, without the alpha channel
    img = read_WSI(path, name, level)
    img = img[0:3]

    # Make numpy array and transform it to HSV
    img_array = np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8, shape=[img.height, img.width, img.bands])
    img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Get the purple colour of the WSI
    mask_HSV = cv2.inRange(img_hsv, (100, 0, 0), (180, 255, 255))

    return mask_HSV


# Use labeling to remove regions in an image smaller then size2remove
def remove_small_regions(img, size2remove=500):
    # Closing of the image and gather the labels of the image
    img = closing(img, square(3))
    label_image = label(img)

    # Run through the image labels and set the small regions to zero
    props = regionprops(label_image)
    for region in props:
        if region.area < size2remove:
            minY, minX, maxY, maxX = region.bbox
            img[minY:maxY, minX:maxX] = 0

    return img


def create_background_mask(path, name, level, size2remove=500):
    # Thresholding the image
    mask = HSV_thresh(path, name, level)

    # Remove small regions in the image
    mask = remove_small_regions(mask, size2remove)

    # Close the holes in the image
    maskInv = cv2.bitwise_not(mask)
    maskInv_closed = remove_small_regions(maskInv, size2remove)
    mask = cv2.bitwise_not(maskInv_closed)

    return mask


# ---------- Create mask from annotations in XML file ----------

def read_XMLfile(path, name, level):
    # Read the data
    tree = ET.parse('{}/{}.xml'.format(path, name))
    root = tree.getroot()

    # Dictionary containing coordinates to each region
    xml_dict = {}

    # Display the data as coordinates in tuples
    # Go through each region
    for region in root.iter('Region'):
        # Make a new list for each uniq key
        if not (region.get('tags') in xml_dict.keys()):
            xml_dict[region.get('tags')] = []

        # List of coordinates belonging to one region
        xy = []

        # Rescale and append coordinates to list
        for vertex in region.find('Vertices'):
            x = int(float(vertex.get('X'))) // (2 ** level)
            y = int(float(vertex.get('Y'))) // (2 ** level)
            xy.append((x, y))

        # Append list of coordinates to each uniq key
        xml_dict[region.get('tags')].append(xy)

    return xml_dict


# Create mask out of annotations
def maskFromAnnotations(dict_list, width, height, tag='Lesion malign'):
    # Containing multiple masks if given tag recurs multiple times in dict_list
    masks = []

    # make mask from given annotations
    for list in dict_list[tag]:
        # make new black image
        mask = Image.new('L', (width, height))
        # Draw the mask on the black image
        ImageDraw.Draw(mask).polygon(list, outline=1, fill=1)
        # Convert from 0/1 to 0/255
        mask = np.array(mask) * 255

        masks.append(mask)

    return masks


# ---------- Set up data, dataset and model for DNN ----------

# Read xlsx file containing tissue and prognosis values
def read_xlsx(path, min_row=3, max_row=55, positive_value='God prognose'):
    filename = '{}.xlsx'.format(path)

    # Read from file
    wb = openpyxl.load_workbook(filename)
    sheet = wb.active

    # Name of tissues
    tissue = []
    # Good or bad prognosis of tissues (good==1, bad==0)
    prognosis = []

    # Read values from xlsx file and append to lists
    for row in sheet.iter_rows(min_row=min_row, max_row=max_row):
        tissue.append(row[0].value)

        if row[1].value == positive_value:
            prognosis.append(1)
        else:
            prognosis.append(0)

    return tissue, prognosis


# Use stratified K-fold to divide WSIs in to train and validation
def makeTrainValDataset(coord_path, xlsx_file, n_splits=5, xlsx_ant_elements=53, xlsx_min_row=3,
                        positive_value='God prognose', crossVal=True, crossValIndex=[]):
    # If no index for doing cross-validation on is chosen, use all the splits of data
    if not crossValIndex:
        crossValIndex = range(n_splits)

    # Get names and category of WSIs in xlsx file
    xlsx_max_row = xlsx_ant_elements + xlsx_min_row - 1
    imgNames_xlsx, categories_xlsx = read_xlsx(xlsx_file, min_row=xlsx_min_row, max_row=xlsx_max_row,
                                               positive_value=positive_value)

    WSInames = os.listdir(coord_path)

    # Get the categories corresponding to the WSIs with gathered coordinates
    categories = []
    for nameWSI in WSInames:
        pos_of_current_category = imgNames_xlsx.index(nameWSI)
        current_category = categories_xlsx[pos_of_current_category]
        categories.append(current_category)

    # Devide WSInames and categories in to train- and validation-sets
    skf = StratifiedKFold(n_splits=n_splits)
    WSInames = np.array(WSInames)
    categories = np.array(categories)
    skfSplit = skf.split(WSInames, categories)

    dataset_x = []
    dataset_y = []
    i = -1

    for train_index, val_index in skfSplit:
        i += 1
        if i in crossValIndex:
            dataset_x.append({})
            dataset_y.append({})
            dataset_x[len(dataset_x) - 1]['train'] = WSInames[train_index]
            dataset_y[len(dataset_y) - 1]['train'] = categories[train_index]
            dataset_x[len(dataset_x) - 1]['val'] = WSInames[val_index]
            dataset_y[len(dataset_y) - 1]['val'] = categories[val_index]
            if not crossVal:
                break

    return dataset_x, dataset_y


# Map-style dataset
class tile_datasets(Dataset):
    def __init__(self, transform, coord_path, dataset_x, dataset_y=None, magnification_levels=None, tileSize=256,
                 tilePerWSI=0):

        self.coord_path = coord_path
        self.magLevels = magnification_levels
        self.tile_size = tileSize
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.transform = transform

        # If tiles to find is limited (tilePerWSI > 0), find random positions in the WSI to gather the tiles from
        # Find the amount of gathered tiles in each WSI
        self.coord_tiles_indexes = []
        self.lenWSIs = []
        for tileName in self.dataset_x:
            tile_coord = \
                read_pickle(self.coord_path, tileName + '/' + 'coordinate_tissue_predictions_pickle')[0]
            if tilePerWSI <= 0 or len(tile_coord) < tilePerWSI:
                self.lenWSIs.append(len(tile_coord))
                indexes = random.sample(range(0, len(tile_coord)), len(tile_coord))
                self.coord_tiles_indexes.append(indexes)
            else:
                self.lenWSIs.append(tilePerWSI)
                indexes = random.sample(range(0, len(tile_coord)), tilePerWSI)
                self.coord_tiles_indexes.append(indexes)

    def __len__(self):
        return sum(self.lenWSIs)

    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        all_magLevels = ['40x', '20x', '10x']

        # Find WSI name and tile position, in coordinate list, for the current tile
        WSInames = self.dataset_x
        lenPreWSI = 0
        for i, WSIlength in enumerate(self.lenWSIs):
            if lenPreWSI + WSIlength - 1 >= idx:
                index_tile = idx - lenPreWSI
                index = self.coord_tiles_indexes[i][index_tile]
                nameWSI = WSInames[i]
                break
            else:
                lenPreWSI += WSIlength

        WSI_coord = read_pickle(self.coord_path, nameWSI + '/' + 'coordinate_tissue_predictions_pickle')
        WSI_coord = WSI_coord[0]

        # Get the WSI file path and tile coordinates
        WSI_path = WSI_coord[index]['path']

        # Extract tiles
        tiles = []
        for magLevel in self.magLevels:
            # magnification level for extracting tile
            level = all_magLevels.index(magLevel)
            # Get the string, referring to the magnification level in the coordinate directory
            coordLevel = 'coordinates_' + magLevel
            # Extract tile from WSI image and transform to fit model
            WSI_img = pyvips.Image.new_from_file(WSI_path, level=level).flatten().rot(1)
            startX, startY = WSI_coord[index][coordLevel]
            tile_obj = WSI_img.extract_area(startX, startY, self.tile_size, self.tile_size)
            tile_img = np.ndarray(buffer=tile_obj.write_to_memory(), dtype=np.uint8,
                                  shape=[tile_obj.height, tile_obj.width, tile_obj.bands])
            tile_img = np.array(tile_img).reshape(tile_img.shape[0], tile_img.shape[1], 3)
            tile_img = Image.fromarray(tile_img)
            tile_img = self.transform(tile_img)
            tiles.append(tile_img)

        # Find the category of the tile
        if self.dataset_y is not None:
            pos_of_category = np.where(self.dataset_x == nameWSI)[0][0]
            category = self.dataset_y[pos_of_category]
        else:
            if len(tiles) == 1:
                return tiles[0], nameWSI
            return tiles, nameWSI

        if len(tiles) == 1:
            return tiles[0], category, nameWSI
        return tiles, category, nameWSI


def make_modelLayers(num_classes, feature_extract, num_img_inputs, use_pretrained):
    model_ft = models.vgg16(pretrained=use_pretrained)

    # Extend input in the first classifier layer to accept features from multiple images
    num_in = model_ft.classifier[0].in_features
    num_out = model_ft.classifier[0].out_features
    model_ft.classifier[0] = nn.Linear(num_in * num_img_inputs, num_out)

    # Freeze feature layers
    OtherMethods.set_parameter_requires_grad(model_ft, feature_extract)

    # Change last layer to have an output equal to number of classes
    num_in = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_in, num_classes)

    return nn.Sequential(*list(model_ft.children()))  # features, avgpool, classifier


# Make model based on multiple input images
class make_multiModel(nn.Module):
    def __init__(self, num_classes, feature_extract, num_img_inputs, use_pretrained=True):
        super(make_multiModel, self).__init__()

        if num_img_inputs == 3:
            self.features_10x, self.avgpool_10x, _ = make_modelLayers(num_classes, feature_extract,
                                                                  num_img_inputs, use_pretrained)

        self.features_20x, self.avgpool_20x, _ = make_modelLayers(num_classes, feature_extract,
                                                              num_img_inputs, use_pretrained)

        self.features_40x, self.avgpool_40x, _ = make_modelLayers(num_classes, feature_extract,
                                                              num_img_inputs, use_pretrained)

        _, _, self.classifier = make_modelLayers(num_classes, feature_extract,
                                                 num_img_inputs, use_pretrained)

    def forward(self, inputs):
        # Extract features
        if len(inputs) == 2:
            x1 = self.features_20x(inputs[0])
            x1 = self.avgpool_20x(x1)
            x1 = torch.flatten(x1, 1)

            x2 = self.features_40x(inputs[1])
            x2 = self.avgpool_40x(x2)
            x2 = torch.flatten(x2, 1)

            # Concatenate
            x = torch.cat([x1, x2], 1)

        elif len(inputs) == 3:
            x1 = self.features_10x(inputs[0])
            x1 = self.avgpool_10x(x1)
            x1 = torch.flatten(x1, 1)

            x2 = self.features_20x(inputs[1])
            x2 = self.avgpool_20x(x2)
            x2 = torch.flatten(x2, 1)

            x3 = self.features_40x(inputs[2])
            x3 = self.avgpool_40x(x3)
            x3 = torch.flatten(x3, 1)

            # Concatenate
            x = torch.cat([x1, x2, x3], 1)

        # Classify
        x = self.classifier(x)

        return x


# ------------------------------


# Read WSI
def read_WSI(path, name, level):
    filename = '{}/{}.ndpi'.format(path, name)
    img = pyvips.Image.new_from_file(filename, level=level).flatten()
    return img


# Get width and height of WSI
def get_imgWidthHeight(path, name, level):
    img = read_WSI(path, name, level)
    img_array = np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8, shape=[img.height, img.width, img.bands])
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    height, width = img_gray.shape
    return width, height


# Save image as jpg
def save_as_jpg(img, path, name, xPos='', yPos='', fromWSI=False):
    if xPos != '':
        filename = '{}/{}#{}#{}.jpg'.format(path, name, xPos, yPos)
    else:
        filename = '{}/{}.jpg'.format(path, name)
    if fromWSI:
        img = np.ndarray(buffer=img.write_to_memory(), dtype=np.uint8, shape=[img.height, img.width, img.bands])
    cv2.imwrite(filename, img)


# Save object as pickle
def save_as_pickle(obj, path, name):
    pickle_filename = '{}/{}.obj'.format(path, name)

    with open(pickle_filename, 'wb') as pickle_writer:
        pickle.dump(obj, pickle_writer)


# Read pickle file to list
def read_pickle(path, name):
    pickle_filename = '{}/{}.obj'.format(path, name)
    obj = []

    with open(pickle_filename, 'rb') as pickle_file:
        while True:
            try:
                obj.append(pickle.load(pickle_file))
            except EOFError:
                break
    return obj


# Save listed objects as text files
# Expects the input "list" to be a list containing objects
def save_listedObject_as_text(list, path):
    text_filename = '{}.text'.format(path)
    with open(text_filename, 'wt') as text_writer:
        for obj in list:
            text = str(obj) + '\n'
            text_writer.write(text)


# Read listed objects, saved as text files
# Expects the text file from the input "path" to contain a list of objects with valid Python datatypes
# Will raise SyntaxError if the list does not contain a valid Python datatype
def read_listedObject_from_text(path):
    text_filename = '{}.text'.format(path)
    list_obj = []
    with open(text_filename) as text_reader:
        list_text = text_reader.readlines()
        for text in list_text:
            obj = ast.literal_eval(text)
            list_obj.append(obj)
    return list_obj


# Find name of file
# returns a string containing: input "rootName" + the first number after the input "rootName"
def get_name(filename, rootName='SUShud'):
    loc = filename.find(rootName) + len(rootName)
    name = filename[loc:]
    name = re.search('[0-9]+', name).group()
    return rootName + name
