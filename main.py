import Preprocessing
import TorchDNN
import Inference
import Prediction

# Folder path to WSIs, of filetype ".ndpi"
WSI_path = 'Data/WSI'  # '/home/prosjekt/Histology/Melanoma_SUS/MSc_Good_Bad_prognosis'

# Folder path to save tile coordinates
output_path = 'Output_tiles'

# What mode to use
# Available options: 0, 1
# Mode "1" trains a model and evaluates based on the validation,
#   and mode "0" uses inference, where a trained model is used to predict prognosis of WSIs.
mode = 1

# --- Preprocessing ---

# Folder path to annotated areas in WSIs, located in ".xml"-file
annotation_path = 'Data/Annotations'  # '/home/prosjekt/Histology/.xmlStorage'

# Folder path to save mask images
mask_path = 'Masks'

# The common name of all WSI, followed by an ID number
rootName = 'SUShud'

# Tag from the XLM file to make mask from
tag = 'Lesion malign'

# --- DNN network ---

# Path to the ".xlsx" file, containing the global label of each WSI
xlsx_file = 'Master_god_dårlig_prognose'

# Position row of the first element in the ".xlsx" file
xlsx_min_row = 3

# Number of WSI in the ".xlsx" file
xlsx_ant_elements = 53

# The value displayed as class 2 (of 2) in the training, and in the confusion matrix
# This value should be the same as one of the two prognosis values used in the xlsx file
xlsx_value = 'God prognose'

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Magnification level(s) to use tiles from
# Level 0 = 40x magnification, level 1 = 20x magnification, and level 2 = 10x magnification
# Available options: ['10x', '20x', '40x']
magnification_levels = ['20x']

# Number of tiles to gather from each WSI
# If this is equal to (or less than) 0, all tiles in each WSI will be used
tilePerWSI = 250

# Size to scale the tiles to. Should be set to "224".
input_size = 224

# --- Folders and files where model and prediction values will be saved ---

# If true, then text files and stored model, made when training, can be overwritten
allow_textFile_overwrite = False

# Name of current folder to save text files and model
text_folder_name = 'mag10x20x40x_OutTiles20x_0.001_crossVal'

# Path to folders, to stored text files in
# Full path to folder, to store text files in: text_folder_path + '/' + text_folder_name
text_folder_path = 'Text_files'

# These names are used to create text/model files in: text_folder_path + '/' + text_folder_name

# Name of text file containing the prediction values of training the model
predictValues_name = 'predictValues'

# Name of text file containing the prediction values of inference
inference_predictValues_name = 'predictValues_inference'

# Name of text file containing accuracy and loss of training the model
hist_name = 'history'

# Name of model file
model_path_name = 'model'

# Name of text file containing information about the settings of this run
text_info_name = 'info'

# Name of text file to save prediction of WSIs
prediction_name = 'predictions'

# ---------


# ___ Preprocessing ___

# Preprocessing saves masks and coordinates of tiles from WSIs.
# This does not have to be done multiple times.

# Make lesion masks from annotated data and tissue masks
Preprocessing.make_AnnotatedLesionMasks(WSI_path, annotation_path, mask_path, output_path, rootName, tag)

# Extract tiles from lesion masks
Preprocessing.extract_tiles(WSI_path, mask_path, output_path)

# _________

if mode:
    # Training and validating of the model
    TorchDNN.DNN(feature_extract, num_classes, batch_size, magnification_levels, tilePerWSI, input_size,
                 xlsx_file, xlsx_min_row, xlsx_ant_elements, xlsx_value, text_folder_path, allow_textFile_overwrite,
                 text_folder_name, predictValues_name, hist_name, model_path_name, text_info_name, output_path)
    # Predict prognosis
    Prediction.Predict_validationData(text_folder_path, text_folder_name, predictValues_name, hist_name)
else:
    # Inference
    Inference.Inference(feature_extract, num_classes, batch_size, magnification_levels, tilePerWSI, input_size,
                        output_path, text_folder_path, text_folder_name, model_path_name, inference_predictValues_name)
    # Predict prognosis
    Prediction.Predict_inference(text_folder_path, text_folder_name, inference_predictValues_name, prediction_name)
