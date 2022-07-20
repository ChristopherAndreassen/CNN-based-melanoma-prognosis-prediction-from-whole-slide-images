import OtherMethods
import MyMethods
import torch
import os
import numpy as np
from torchvision import transforms


def Inference(feature_extract, num_classes, batch_size, magnification_levels, tilePerWSI, input_size,
              coord_path, text_folder_path, text_folder_name, model_path_name, inference_predicValues_name):
    # Find paths for model and parameters
    text_path = text_folder_path + '/' + text_folder_name
    model_path = text_path + '/' + model_path_name

    # Make model
    if len(magnification_levels) == 1:
        model = OtherMethods.make_Model(num_classes, feature_extract)
    else:
        model = MyMethods.make_multiModel(num_classes, feature_extract, num_img_inputs=len(magnification_levels))

    # Data normalization
    data_transforms = transforms.Compose([transforms.Resize(input_size), transforms.CenterCrop(input_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load model weights
    model.load_state_dict(torch.load(model_path))

    # Make dataset
    dataset_x = np.array(os.listdir(coord_path))

    image_datasets = MyMethods.tile_datasets(transform=data_transforms, coord_path=coord_path,
                                             dataset_x=dataset_x, tilePerWSI=tilePerWSI,
                                             magnification_levels=magnification_levels)

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # --- Inference ---

    model.eval()

    running_names = []
    running_preds = []

    for data_inputs, names in dataloaders:
        if len(magnification_levels) > 1:
            inputs = []
            for x in data_inputs:
                inputs.append(x.to(device))
        else:
            inputs = data_inputs.to(device)

        # forward
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # Make lists with WSI names and predicted class
        for i, name in enumerate(names):
            if name not in running_names:
                running_names.append(name)
                running_preds.append([])

            name_index = running_names.index(name)
            running_preds[name_index].append(preds[i].item())

    predValues = [{
        'names': running_names,
        'predict': running_preds,
    }]

    # Save prediction values to text file+
    MyMethods.save_listedObject_as_text(predValues, text_path + '/' + inference_predicValues_name)
