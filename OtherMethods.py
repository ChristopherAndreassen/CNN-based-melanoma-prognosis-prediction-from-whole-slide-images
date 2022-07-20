from skimage.measure import regionprops
from skimage import morphology
from scipy import ndimage
import numpy as np
import pyvips
import pickle
import time
import cv2
import os
from torchvision import models
import torch.nn as nn

# Extract tiles from WSI and save coordinates of these tiles
def extractTiles(wsi_dataset_file_path, mask_dataset_file_path, wsi_filename, wsi_name, output_folder='Output', magLevel = '10x'):

    # Set this option to True to save tiles as JPEG images in a separate folder.
    save_tiles_as_jpeg = False

    # Set this option to True to save the binary annotation mask as a JPEG image.
    save_binary_annotation_mask = True

    # In the WSI folder, there is a file containing a dict with the 7 binary masks.
    # To specify which of these masks to use, list the tissue types in the following variable.
    # Available options: ['tissue', 'Lesion malign']
    tissue_classes_to_fit_tiles_on = ['Lesion malign']

    # How large percentage of the tile must cover the region to be consider a valid tile.
    # Float between 0-1.
    PHI = 0.7

    # Which level in the WSI to use when checking for valid tiles (level 0=40x, level 1=20x, and level 2=10x).
    # Available options: '10x', '20x', '40x'.
    ALPHA = magLevel

    # All valid tiles are displayed on the 10x image and saved as JPEG to the folder.
    # This option determines which of the three levels to include in the final image.
    # Tiles from all three levels in the WSI are saved, this option is only for visualization.
    # Available options: ['10x', '20x', '40x'].
    TILES_TO_SHOW = [magLevel]

    # Size of width/height of the tiles to extract. Integer.
    TILE_SIZE = 256

    # The level the annotation mask is on, and also the level our tiles are on. For our mask it's '10x'.
    TAU = '10x'

    # The binary masks contain small regions which is not of interest.
    # These are removed using the remove_small_objects() function.
    # This variable sets the minimum size to remove.
    # Available options: Integer values, usually between 500 and 20000.
    SMALL_REGION_REMOVE_THRESHOLD = 3000

    # Paths
    extracted_tiles_folder = 'Extracted_tiles'
    os.makedirs(output_folder, exist_ok=True)
    if save_tiles_as_jpeg:
        os.makedirs(extracted_tiles_folder, exist_ok=True)

    # Variable initialization
    dict_of_all_predicted_coordinates_dict = dict()
    list_of_valid_tiles_from_current_wsi = []
    i = 0

    # Create a dict containing the ratio for each level
    region_masks = dict()
    ratio_dict = dict()
    ratio_dict['40x'] = 1
    ratio_dict['20x'] = 2
    ratio_dict['10x'] = 4

    # Create a dict containin the index of each class
    tissue_class_to_index = dict()
    tissue_class_to_index['tissue'] = 0
    tissue_class_to_index['Lesion malign'] = 1


    # Start timer
    current_wsi_start_time = time.time()

    # Create folder for each WSI to store output
    os.makedirs(output_folder + '/' + wsi_name, exist_ok=True)
    if save_tiles_as_jpeg:
        os.makedirs(extracted_tiles_folder + '/' + wsi_name, exist_ok=True)

    # Load annotation mask. For us, this is a pickle file containing the annotation mask for all tissue classes.
    annotation_mask_path = mask_dataset_file_path + '/' + wsi_name + '/' + 'mask.obj'
    with open(annotation_mask_path, 'rb') as handle:
        annotation_mask_all_classes = pickle.load(handle)

    # Read images
    full_image_40 = pyvips.Image.new_from_file(wsi_dataset_file_path + '/' + wsi_filename + '.ndpi', level=0, autocrop=True).flatten().rot(1)
    full_image_20 = pyvips.Image.new_from_file(wsi_dataset_file_path + '/' + wsi_filename + '.ndpi', level=1, autocrop=True).flatten().rot(1)
    full_image_10 = pyvips.Image.new_from_file(wsi_dataset_file_path + '/' + wsi_filename + '.ndpi', level=2, autocrop=True).flatten().rot(1)

    # Find width/heigh of 10x image
    scn_width_10x = full_image_10.width
    scn_height_10x = full_image_10.height
    print('Loaded WSI with size {} x {}'.format(str(scn_width_10x), str(scn_height_10x)))

    # Loop through each tissue class to fit tiles on
    for current_class_to_copy in tissue_classes_to_fit_tiles_on:
        print('Now processing {} regions'.format(current_class_to_copy))

        # Extract mask for current class
        current_class_mask = annotation_mask_all_classes[tissue_class_to_index[current_class_to_copy]].copy()

        # Resize colormap to the size of 10x overview image
        current_class_mask = cv2.resize(current_class_mask, dsize=(scn_width_10x, scn_height_10x), interpolation=cv2.INTER_CUBIC)
        print('Loaded segmentation mask with size {} x {}'.format(current_class_mask.shape[1], current_class_mask.shape[0]))

        # Save the annotation mask image (If option is set to True)
        if save_binary_annotation_mask:
            annotation_mask_for_saving = current_class_mask * 255
            cv2.imwrite(output_folder + '/' + wsi_name + '/Binary_annotation_mask_{}.jpg'.format(current_class_to_copy), annotation_mask_for_saving, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        # Use a boolean condition to find where pixel values are > 0.75
        blobs = current_class_mask > 0.75

        # label connected regions that satisfy this condition
        labels, regions_found_in_wsi_before_removing_small_obj = ndimage.label(blobs)
        print('\tFound {} regions'.format(regions_found_in_wsi_before_removing_small_obj))

        # Remove all the small regions
        labels = morphology.remove_small_objects(labels, min_size=SMALL_REGION_REMOVE_THRESHOLD)

        # Get region properties
        list_of_regions = regionprops(labels)

        n_regions_after_removing_small_obj = len(list_of_regions)
        print('\tFound {} regions (after removing small objects)'.format(n_regions_after_removing_small_obj))

        # Create a new binary map after removing small objects
        region_masks[current_class_to_copy] = np.zeros(shape=(current_class_mask.shape[0], current_class_mask.shape[1]))

        # Extract all coordinates (to draw region on overview image)
        for current_region in list_of_regions:
            for current_region_coordinate in current_region.coords:
                region_masks[current_class_to_copy][current_region_coordinate[0], current_region_coordinate[1]] = 1

        # Create a grid of all possible x- and y-coordinates (starting position (0,0))
        all_x_pos, all_y_pos = [], []
        for x_pos in range(0, int(current_class_mask.shape[1] - TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))):
            all_x_pos.append(x_pos)
        for y_pos in range(0, int(current_class_mask.shape[0] - TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))):
            all_y_pos.append(y_pos)

        # Create a new list with all xy-positions in current SCN image
        list_of_valid_tiles_from_current_class = []
        for y_pos in all_y_pos:
            for x_pos in all_x_pos:
                # Equation 1 in paper
                if int(sum(sum(region_masks[current_class_to_copy][y_pos:y_pos + int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])),
                               x_pos:x_pos + int(TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU]))]))) >= (pow((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])), 2) * PHI):
                    list_of_valid_tiles_from_current_class.append((x_pos, y_pos))

        # Add the tiles to the list of tiles of current wsi
        list_of_valid_tiles_from_current_wsi.extend(list_of_valid_tiles_from_current_class)
        tile_x = dict()
        tile_y = dict()

        # Save coordinates for each tiles to a dict to create a dataset
        for current_xy_pos in list_of_valid_tiles_from_current_wsi:

            # Equation 2 in paper.
            for BETA in ['10x', '20x', '40x']:
                tile_x[BETA] = (current_xy_pos[0] + (TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2) * (ratio_dict[TAU] / ratio_dict[BETA]) - TILE_SIZE / 2
                tile_y[BETA] = (current_xy_pos[1] + (TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2) * (ratio_dict[TAU] / ratio_dict[BETA]) - TILE_SIZE / 2

            # Save tile to coordinate dict (coordinate of top-left corner)
            id_number = len(dict_of_all_predicted_coordinates_dict.keys())
            dict_of_all_predicted_coordinates_dict[id_number] = dict()
            dict_of_all_predicted_coordinates_dict[id_number]['path'] = wsi_dataset_file_path + '/' + wsi_filename + '.ndpi'
            dict_of_all_predicted_coordinates_dict[id_number]['coordinates_40x'] = (int(tile_x['40x']), int(tile_y['40x']))
            dict_of_all_predicted_coordinates_dict[id_number]['coordinates_20x'] = (int(tile_x['20x']), int(tile_y['20x']))
            dict_of_all_predicted_coordinates_dict[id_number]['coordinates_10x'] = (int(tile_x['10x']), int(tile_y['10x']))
            dict_of_all_predicted_coordinates_dict[id_number]['tissue_type'] = current_class_to_copy

            # Extract and save tiles as jpeg-images (If option is set to True)
            if save_tiles_as_jpeg:
                tile_40x = full_image_40.extract_area(int(tile_x['40x']), int(tile_y['40x']), TILE_SIZE, TILE_SIZE)
                tile_20x = full_image_20.extract_area(int(tile_x['20x']), int(tile_y['20x']), TILE_SIZE, TILE_SIZE)
                tile_10x = full_image_10.extract_area(int(tile_x['10x']), int(tile_y['10x']), TILE_SIZE, TILE_SIZE)
                tile_40x.jpegsave(extracted_tiles_folder + '/' + wsi_name + '/tile_{}_40x.jpeg'.format(i), Q=100)
                tile_20x.jpegsave(extracted_tiles_folder + '/' + wsi_name +'/tile_{}_20x.jpeg'.format(i), Q=100)
                tile_10x.jpegsave(extracted_tiles_folder + '/' + wsi_name + '/tile_{}_10x.jpeg'.format(i), Q=100)
                i += 1

    # Save predicted coordinates dict as pickle
    with open(output_folder + '/' + wsi_name + '/coordinate_tissue_predictions_pickle.obj', 'wb') as handle:
        pickle.dump(dict_of_all_predicted_coordinates_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save overview image
    filename = output_folder + '/' + wsi_name + '/image_clean.jpeg'
    full_image_10.jpegsave(filename, Q=100)

    # Read overview image again using cv2, and add alpha channel to overview image.
    overview_jpeg_file = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    overview_jpeg_file = np.dstack([overview_jpeg_file, np.ones((overview_jpeg_file.shape[0], overview_jpeg_file.shape[1]), dtype="uint8") * 255])

    # Convert masks from 0-1 -> 0-255 (can also be used to set the color)
    for n in tissue_classes_to_fit_tiles_on:
        region_masks[n] *= 255

    # Resize masks to same size as the overview image
    for n in tissue_classes_to_fit_tiles_on:
        region_masks[n] = cv2.resize(region_masks[n], dsize=(overview_jpeg_file.shape[1], overview_jpeg_file.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Create a empty alpha channel
    alpha_channel = np.zeros(shape=(region_masks[tissue_classes_to_fit_tiles_on[0]].shape[0], region_masks[tissue_classes_to_fit_tiles_on[0]].shape[1]))

    # Each mask is 1-channel, merge them to create a 3-channel image (RGB), the order is used to set the color for each mask. Add a alpha-channel.
    if len(tissue_classes_to_fit_tiles_on) >= 1:
        region_masks[tissue_classes_to_fit_tiles_on[0]] = cv2.merge((region_masks[tissue_classes_to_fit_tiles_on[0]], alpha_channel, alpha_channel, alpha_channel))
    if len(tissue_classes_to_fit_tiles_on) >= 2:
        region_masks[tissue_classes_to_fit_tiles_on[1]] = cv2.merge((alpha_channel, region_masks[tissue_classes_to_fit_tiles_on[1]], alpha_channel, alpha_channel))
    if len(tissue_classes_to_fit_tiles_on) >= 3:
        region_masks[tissue_classes_to_fit_tiles_on[2]] = cv2.merge((alpha_channel, alpha_channel, region_masks[tissue_classes_to_fit_tiles_on[2]], alpha_channel))
    if len(tissue_classes_to_fit_tiles_on) >= 4:
        region_masks[tissue_classes_to_fit_tiles_on[3]] = cv2.merge((region_masks[tissue_classes_to_fit_tiles_on[3]], region_masks[tissue_classes_to_fit_tiles_on[3]], alpha_channel, alpha_channel))
    if len(tissue_classes_to_fit_tiles_on) >= 5:
        region_masks[tissue_classes_to_fit_tiles_on[4]] = cv2.merge((region_masks[tissue_classes_to_fit_tiles_on[4]], alpha_channel, region_masks[tissue_classes_to_fit_tiles_on[4]], alpha_channel))
    if len(tissue_classes_to_fit_tiles_on) >= 6:
        region_masks[tissue_classes_to_fit_tiles_on[5]] = cv2.merge((alpha_channel, region_masks[tissue_classes_to_fit_tiles_on[5]], region_masks[tissue_classes_to_fit_tiles_on[5]], alpha_channel))

    # Draw the selected regions on the overview image
    for _, current_tissue_mask in region_masks.items():
        overview_jpeg_file = cv2.addWeighted(current_tissue_mask, 1, overview_jpeg_file, 1.0, 0, dtype=cv2.CV_64F)

    # Draw tiles on the overview image
    for current_xy_pos in list_of_valid_tiles_from_current_wsi:
        start_x = dict()
        start_y = dict()
        end_x = dict()
        end_y = dict()

        # Equation 3 in paper.
        for BETA in ['10x', '20x', '40x']:
            start_x[BETA] = int(((current_xy_pos[0] + ((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2)) * (ratio_dict[TAU] / ratio_dict['10x'])) - ((TILE_SIZE * (ratio_dict[BETA] / ratio_dict['10x'])) / 2))
            start_y[BETA] = int(((current_xy_pos[1] + ((TILE_SIZE * (ratio_dict[ALPHA] / ratio_dict[TAU])) / 2)) * (ratio_dict[TAU] / ratio_dict['10x'])) - ((TILE_SIZE * (ratio_dict[BETA] / ratio_dict['10x'])) / 2))
            end_x[BETA] = int(start_x[BETA] + TILE_SIZE * (ratio_dict[BETA] / ratio_dict['10x']))
            end_y[BETA] = int(start_y[BETA] + TILE_SIZE * (ratio_dict[BETA] / ratio_dict['10x']))

        # Draw tiles (Red tiles indicate which level ALPHA is, and the corresponding levels are shown in green)
        for draw_level in TILES_TO_SHOW:
            color = (0, 0, 255) if draw_level == ALPHA else (0, 255, 0)
            cv2.rectangle(overview_jpeg_file, (start_x[draw_level], start_y[draw_level]), (end_x[draw_level], end_y[draw_level]), color, 3)

    # Save overview image
    cv2.imwrite(output_folder + '/' + wsi_name + '/image_with_mask_and_tiles_alpha_{}_phi_{}_{}.jpg'.format(ALPHA, PHI, len(list_of_valid_tiles_from_current_wsi)), overview_jpeg_file, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # Calculate elapse time for current run
    elapse_time = time.time() - current_wsi_start_time
    m, s = divmod(elapse_time, 60)
    h, m = divmod(m, 60)
    model_time = '%02d:%02d:%02d' % (h, m, s)

    # Print out results
    print('Found {} tiles in image'.format(len(list_of_valid_tiles_from_current_wsi)))
    print('Finished. Duration: {}'.format(model_time))



# Freeze parameters in given model
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for name, param in model.named_parameters():
            if 'features' in name:
                param.requires_grad = False


# Make model based on one input image
def make_Model(num_classes, feature_extract, use_pretrained=True):
    model_ft = models.vgg16(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    return model_ft
