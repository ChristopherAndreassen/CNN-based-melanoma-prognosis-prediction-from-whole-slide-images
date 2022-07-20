from OtherMethods import extractTiles
import MyMethods
import numpy as np
import cv2
import os


def make_AnnotatedLesionMasks(WSI_path, annotation_path, mask_path='Masks', output_path='Output_tiles',
                              rootName='SUShud', tag='Lesion malign'):
    # Magnification level to make the masks from
    #   (the levels available depends on the pyramid structure of the WSI-image)
    # Level 0 = 40x magnification, level 1 = 20x, level 2 = 10x, Level 3 = 5x,
    #   level 4 = 2.5x, level 5 = 1.25x, and level 6 = 0.625x
    # Available options: integer from 0-6
    level = 5

    # Get the WSIs
    WSI_filenames = os.listdir(WSI_path)
    # Delete name from list, if it is not an WSI
    for filename in WSI_filenames:
        if os.path.splitext(filename)[1] != '.ndpi':
            WSI_filenames.remove(filename)

    # Get the annotation data
    all_annotation_filenames = os.listdir(annotation_path)
    # Gather all annotation data, corresponding to the WSIs
    annotation_filenames = []
    for filename in all_annotation_filenames:
        if rootName in filename:
            annotation_filenames.append(filename)


    # Go through every WSI and save tissue masks for each WSI
    for filename_WSI in WSI_filenames:
        WSI_name = MyMethods.get_name(filename_WSI)

        filename_annotation = ''

        # Get annotation mask with same name as current WSI
        for filename in annotation_filenames:
            if WSI_name in filename:
                name = os.path.splitext(filename)[0]
                # Check that annotations contains the wanted mask
                if tag in MyMethods.read_XMLfile(annotation_path, name, level=level).keys():
                    filename_annotation = filename
                    break

        # Go to next if WSI is not annotated with given tag
        if filename_annotation == '':
            continue

        # To avoid the file type in the filename
        filename_WSI = os.path.splitext(filename_WSI)[0]
        filename_annotation = os.path.splitext(filename_annotation)[0]

        # Make folders for mask- and output-files, if these folders do not exist
        os.makedirs(mask_path + '/' + WSI_name, exist_ok=True)
        os.makedirs(output_path + '/' + WSI_name, exist_ok=True)

        # Get the annotations of the WSI
        annotations = MyMethods.read_XMLfile(annotation_path, filename_annotation, level=level)

        # Make Milign mask
        width, height = MyMethods.get_imgWidthHeight(WSI_path, filename_WSI, level=level)
        maskMalign = MyMethods.maskFromAnnotations(annotations, width, height, tag=tag)

        # Make Tissue mask
        maskTissue = MyMethods.create_background_mask(WSI_path, filename_WSI, level=level)

        # Find the malign lesion mask of each area in the WSI
        malignLesionMasks = []
        for maskID in range(len(maskMalign)):
            malignLesionMasks.append(cv2.bitwise_and(maskMalign[maskID], maskTissue))
        # Set the masks together in one mask containing all the malign lesion
        for maskID in range(len(malignLesionMasks)):
            if maskID == 0:
                maskMalignLesion = malignLesionMasks[0]
            else:
                maskMalignLesion = cv2.bitwise_or(maskMalignLesion, malignLesionMasks[maskID])

        maskMalignLesion = np.rot90(maskMalignLesion, -1)

        # Save Tissue mask and Malign Lesion mask as a pickle file
        masks = [maskTissue, maskMalignLesion]
        MyMethods.save_as_pickle(masks, mask_path + '/' + WSI_name, 'mask')

        print('Saved lesion mask for WSI image: ' + WSI_name)


def extract_tiles(WSI_path, mask_path='Masks', output_path='Output_tiles'):
    # Extract wanted tiles from WSI
    # Magnification level to check for valid tiles
    # Level 0 = 40x magnification, level 1 = 20x, and level 2 = 10x
    # Available options: '10x', '20x', '40x'
    magLevel = '20x'

    # Get the WSIs
    WSI_filenames = os.listdir(WSI_path)
    # Get Masks
    Mask_foldernames = os.listdir(mask_path)

    # Go through every mask and use it to extract wanted tiles from each WSI
    for WSI_name in Mask_foldernames:

        # Get WSI with same name as current mask
        for filename in WSI_filenames:
            if WSI_name in filename:
                filename_WSI = os.path.splitext(filename)[0]

        print('Extract tiles from WSI image: ' + WSI_name)
        extractTiles(WSI_path, mask_path, wsi_filename=filename_WSI, wsi_name=WSI_name, output_folder=output_path,
                     magLevel=magLevel)
