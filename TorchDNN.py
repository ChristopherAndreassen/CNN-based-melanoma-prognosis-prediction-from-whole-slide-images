from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import time
import copy
import MyMethods
import OtherMethods
import os


# Train model
def train_model(model, dataloaders, criterion, optimizer, device, magnification_levels, num_epochs, patience=-1):
    since = time.time()

    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    acc_atBestLoss = 0

    patience_count = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            running_names = []
            running_labels = []
            running_preds = []
            running_conf_mtx = [[0, 0], [0, 0]]

            # Iterate over data.
            for data_inputs, labels, names in dataloaders[phase]:
                if len(magnification_levels) > 1:
                    inputs = []
                    inputSize = 0
                    for x in data_inputs:
                        inputs.append(x.to(device))
                        inputSize += x.size(0)
                else:
                    inputs = data_inputs.to(device)
                    inputSize = inputs.size(0)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputSize
                running_corrects += torch.sum(preds == labels.data)

                # Make lists with WSI names, labels and predicts from this epoch
                for i, name in enumerate(names):
                    if name not in running_names:
                        running_names.append(name)
                        running_labels.append(labels[i].item())
                        running_preds.append([])

                    name_index = running_names.index(name)
                    running_preds[name_index].append(preds[i].item())

                # Make confusion matrix
                conf_mtx = confusion_matrix(labels.data, preds)
                for i in range(len(conf_mtx)):
                    for j in range(len(conf_mtx[0])):
                        running_conf_mtx[i][j] += conf_mtx[i][j]

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model and statistic parameters
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                names_atBestLoss = running_names
                labels_atBestLoss = running_labels
                preds_atBestLoss = running_preds
                acc_atBestLoss = epoch_acc
                conf_mtx_atBestLoss = running_conf_mtx
                best_model_wts = copy.deepcopy(model.state_dict())
                patience_count = 0
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
                val_loss_history.append(epoch_loss)
            else:
                train_acc_history.append(epoch_acc.item())
                train_loss_history.append(epoch_loss)
            # Early stopping
            if phase == 'val' and epoch_loss >= best_loss and patience >= 0:
                if patience_count == patience:
                    print()
                    print('Training stopped because of early stopping')
                    break
                patience_count += 1

        # Break loop if previous loop is terminated by a break statement
        else:
            print()
            continue
        break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Validation accuracy: {:4f}, with lowest validation loss: {:4f}'.format(acc_atBestLoss, best_loss))

    history = {
        'accuracy': train_acc_history,
        'val_accuracy': val_acc_history,
        'loss': train_loss_history,
        'val_loss': val_loss_history
    }

    predValues = {
        'names': names_atBestLoss,
        'labels': labels_atBestLoss,
        'predict': preds_atBestLoss,
    }

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history, conf_mtx_atBestLoss, predValues


def DNN(feature_extract, num_classes, batch_size, magnification_levels, tilePerWSI, input_size,
        xlsx_file, xlsx_min_row, xlsx_ant_elements, xlsx_value, text_folder_path, allow_textFile_overwrite,
        text_folder_name, predictValues_name, hist_name, model_path_name, text_info_name, coord_path):
    # --- Parameters for training the model ---

    # Number of epochs to train for
    num_epochs = 15

    # Number of epochs before early stopping (patience < 0 results in no early stopping)
    patience = 10

    # --- Parameters for making the optimizer ---

    # Learning rate
    lr = 0.0001

    # Moment in the training
    momentum = 0.9

    # --- Parameters for gathering tiles, and to split them up in train and validation sets ---

    # Flag for cross-validation. If true, multiple train- and validation-set will be used to evaluate the network,
    #   if false, only one train- and validation-set will be used
    crossVal = False

    # Number of times to split the data-set, corresponding to the train/validation relationship
    # Percentage of train-sets from the whole data-set is equal to 100-(100/n_splits),
    #   percentage of validation-sets is equal to 100/n_splits
    n_splits = 5

    # List of iteration(s) to be used as the validation set(s) in cross validation.
    #   If cross validation is not used, first iteration index in list is used
    # Available options: [0, 1, 2, ..., n_splits-1]
    # Set to an empty list ([]) if no specific iteration(s) are desired. In this case,
    #   first iteration is used when cross validation is not used, and all iterations are used in cross validation
    crossValIndex = []

    # ---------

    # Path to store current text files and model
    text_path = text_folder_path + '/' + text_folder_name

    # Make folder to store text files and model
    try:
        os.makedirs(text_path, exist_ok=allow_textFile_overwrite)
    except OSError:
        print('\nThe folder: ' + text_path + ', already exists.'
                                             '\nChange to another folder, or allow text files in this folder to be overwritten.')
        exit()

    # Make List with information about the settings of this run
    if tilePerWSI <= 0:
        real_tilePerWSI = 'all'
    else:
        real_tilePerWSI = tilePerWSI
    info = [
        'feature_extract = ' + str(feature_extract),
        'num_classes = ' + str(num_classes),
        '',
        'batch_size = ' + str(batch_size),
        'num_epochs = ' + str(num_epochs),
        'patience = ' + str(patience),
        '',
        'lr = ' + str(lr),
        'momentum = ' + str(momentum),
        '',
        'coord_path = ' + coord_path,
        'crossVal = ' + str(crossVal),
        'magnification_levels = ' + str(magnification_levels),
        'n_splits = ' + str(n_splits),
        'crossValIndex = ' + str(crossValIndex),
        'tilePerWSI = ' + str(real_tilePerWSI),
        ''
    ]

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Make train- and validation-datasets
    dataset_x, dataset_y = MyMethods.makeTrainValDataset(coord_path=coord_path, xlsx_file=xlsx_file, n_splits=n_splits,
                                                         xlsx_ant_elements=xlsx_ant_elements, xlsx_min_row=xlsx_min_row,
                                                         positive_value=xlsx_value, crossVal=crossVal,
                                                         crossValIndex=crossValIndex)

    # --- Train each dataset in dataset_x, with true labels in dataset_y ---

    histories = []
    allModels = []
    allPredValues = []

    for i in range(len(dataset_x)):
        # Initialize the model for this run
        if len(magnification_levels) == 1:
            model_ft = OtherMethods.make_Model(num_classes, feature_extract, use_pretrained=True)
        else:
            model_ft = MyMethods.make_multiModel(num_classes, feature_extract, num_img_inputs=len(magnification_levels),
                                                 use_pretrained=True)

        # Print the instantiated model
        print(model_ft)

        # Create training and validation datasets
        print("\nInitializing Datasets and Dataloaders...")
        print('\nWSIs in training-set: ' + str(len(dataset_x[i]['train'])))
        print('WSIs in validation-set: ' + str(len(dataset_x[i]['val'])))
        image_datasets = {x: MyMethods.tile_datasets(transform=data_transforms[x], coord_path=coord_path,
                                                     dataset_x=dataset_x[i][x], dataset_y=dataset_y[i][x],
                                                     tilePerWSI=tilePerWSI, magnification_levels=magnification_levels)
                          for x in ['train', 'val']}
        print('\nTiles in training-set: ' + str(image_datasets['train'].len()))
        print('Tiles in validation-set: ' + str(image_datasets['val'].len()) + '\n')

        # Save information about WSI and tile amount in info file
        info.append('WSIs in training-set: ' + str(len(dataset_x[i]['train'])))
        info.append('WSIs in validation-set: ' + str(len(dataset_x[i]['val'])))
        info.append('Tiles in training-set: ' + str(image_datasets['train'].len()))
        info.append('Tiles in validation-set: ' + str(image_datasets['val'].len()))
        info.append('')

        # Create training and validation dataloaders
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x
            in
            ['train', 'val']}

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)
        print()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=momentum)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()

        # Train and evaluate
        # Histories contain (in order) : training accuracy, validation accuracy, training loss, validation loss
        model_ft, hist, conf_mtx, predValues = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device,
                                                           magnification_levels, num_epochs, patience)

        allModels.append(model_ft)
        histories.append(hist)
        allPredValues.append(predValues)

    # ---------

    # Save information about this run to text file
    MyMethods.save_listedObject_as_text(info, text_path + '/' + text_info_name)

    # Save prediction values to text file
    MyMethods.save_listedObject_as_text(allPredValues, text_path + '/' + predictValues_name)

    # Save history to text file
    MyMethods.save_listedObject_as_text(histories, text_path + '/' + hist_name)

    # Save model
    for i, model in enumerate(allModels):
        if crossVal:
            torch.save(model.state_dict(), text_path + '/' + model_path_name + '_' + str(i))
        else:
            torch.save(model.state_dict(), text_path + '/' + model_path_name)
