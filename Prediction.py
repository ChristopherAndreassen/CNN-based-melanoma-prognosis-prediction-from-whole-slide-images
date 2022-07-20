import matplotlib.pyplot as plt
import seaborn as sns
import MyMethods
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np


def Predict_validationData(text_folder_path, text_folder_name, predictValues_name, hist_name):
    # The class with bad prognosis
    # Available options: 0 (class 1) or 1 (class 2)
    class_bad = 0

    # Threshold to predict bad prognosis of WSI
    # WSIs with proportion of predicted bad-prognosis tiles over threshold is predicted to have bad prognosis
    threshold = 0.4720

    # Path to stored text files
    text_path = text_folder_path + '/' + text_folder_name

    # Get prediction values from stored text file
    predictValues = MyMethods.read_listedObject_from_text(text_path + '/' + predictValues_name)

    # Get history from stored text file
    histories = MyMethods.read_listedObject_from_text(text_path + '/' + hist_name)

    # Plot accuracy and loss
    for i, hist in enumerate(histories, start=1):
        fig, (ax1, ax2) = plt.subplots(1, 2, num='History plot ' + str(i))
        # Plot accuracy
        ax1.plot(range(0, len(hist['accuracy'])), hist['accuracy'], label='training accuracy')
        ax1.plot(range(0, len(hist['val_accuracy'])), hist['val_accuracy'], label='validation accuracy')
        ax1.set_title('Accuracy')
        ax1.set_ylim((0, 1))
        ax1.legend(loc="upper left")
        # Plot Loss
        ax2.plot(range(0, len(hist['loss'])), hist['loss'], label='training loss')
        ax2.plot(range(0, len(hist['val_loss'])), hist['val_loss'], label='validation loss')
        ax2.set_title('Loss')
        ax2.set_ylim((0, 2 * max(hist['val_loss'])))
        ax2.legend(loc="upper left")

        val_loss = min(hist['val_loss'])
        val_index = hist['val_loss'].index(val_loss)
        val_acc = hist['val_accuracy'][val_index]

        print('Best epoch: ' + str(val_index))
        print('Minimum validation loss: ' + str(val_loss))
        print('Accuracy: ' + str(val_acc))
        print('')

    # Show performance for WSIs
    all_positiveValues = []
    all_trueValues = []
    for j, predictValue in enumerate(predictValues, start=1):
        prediction = []
        positiveValues = []

        for i, name in enumerate(predictValue['names']):
            true_label = predictValue['labels'][i]
            predTiles = predictValue['predict'][i]

            numb_correct = 0
            numb_bad = 0

            # Count number of tiles truly predicted and tiles predicted as bad prognosis
            for pred in predTiles:
                if pred == true_label:
                    numb_correct += 1
                if pred == class_bad:
                    numb_bad += 1

            # Proportion of tiles predicted as bad prognosis
            proportion_bad = numb_bad / len(predTiles)
            positiveValues.append(proportion_bad)

            # Find correct class
            if true_label == class_bad:
                correct_class = 'Bad prognosis'
            else:
                correct_class = 'Good prognosis'

            # Find predictions
            if proportion_bad >= threshold:
                prediction.append(class_bad)
                pred_class = 'Bad prognosis'
            else:
                prediction.append(1 - class_bad)
                pred_class = 'Good prognosis'

            print('WSI: ' + name)
            print('Number of tiles predicted as bad prognosis: ' + str(numb_bad) + '/' + str(len(predTiles)))
            print('Proportion of tiles predicted as bad prognosis: ' + str(proportion_bad))
            print('Predicted class: ' + pred_class)
            print('Correct class: ' + correct_class)
            print('')

        all_positiveValues.extend(positiveValues)
        all_trueValues.extend(predictValue['labels'])

        # Make confusion matrix
        conf_mtx = confusion_matrix(predictValue['labels'], prediction)

        TP = conf_mtx[0][0]
        FP = conf_mtx[1][0]
        TN = conf_mtx[1][1]
        FN = conf_mtx[0][1]

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        F1_score = (2 * TP) / (2 * TP + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (FP + TN)

        print('Accuracy: ' + str(accuracy))
        print('F1 score: ' + str(F1_score))
        print('Sensitivity: ' + str(sensitivity))
        print('Specificity: ' + str(specificity))
        print('')

        # Plot confusion matrix
        f, ax = plt.subplots(figsize=(4, 4), num='WSI predictions ' + str(j))
        labels = ['Bad prognosis', 'Good prognosis']
        sns.heatmap(conf_mtx, annot=True, linewidths=0.01, cmap='Oranges', linecolor='gray', fmt='d',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted class')
        plt.ylabel('True class')

    # Make receiver operating characteristic
    fpr, tpr, thres = roc_curve(np.array(all_trueValues), np.array(all_positiveValues), pos_label=0)
    roc_auc = auc(fpr, tpr)

    print('')
    print('Thresholds Sensitivity 1-Specificity F1_score Accuracy: ')
    for i, testTheshold in enumerate(thres):
        prediction = []

        # Find predictions with different thresholds
        for posValue in all_positiveValues:
            if posValue >= testTheshold:
                prediction.append(class_bad)
            else:
                prediction.append(1 - class_bad)

        conf_mtx = confusion_matrix(all_trueValues, prediction)

        TP = conf_mtx[0][0]
        FP = conf_mtx[1][0]
        TN = conf_mtx[1][1]
        FN = conf_mtx[0][1]

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        F1_score = (2 * TP) / (2 * TP + FP + FN)

        print('{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(thres[i], tpr[i], fpr[i], F1_score, accuracy))

    plt.figure('ROC')
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")

    plt.show()


def Predict_inference(text_folder_path, text_folder_name, predictValues_name, prediction_name):
    # The class with bad prognosis
    # Available options: 0 (class 1) or 1 (class 2)
    class_bad = 0

    # Threshold to predict bad prognosis of WSI
    # WSIs with proportion of predicted bad-prognosis tiles over threshold is predicted to have bad prognosis
    threshold = 0.3720

    # Path to stored text files
    text_path = text_folder_path + '/' + text_folder_name

    # Get prediction values from stored text file
    predictValues = MyMethods.read_listedObject_from_text(text_path + '/' + predictValues_name)

    predictonString = ''

    # Show performance for WSIs
    for j, predictValue in enumerate(predictValues, start=1):
        prediction = []
        positiveValues = []

        for i, name in enumerate(predictValue['names']):
            predTiles = predictValue['predict'][i]

            numb_bad = 0

            # Count number of tiles truly predicted and tiles predicted as bad prognosis
            for pred in predTiles:
                if pred == class_bad:
                    numb_bad += 1

            # Proportion of tiles predicted as bad prognosis
            proportion_bad = numb_bad / len(predTiles)
            positiveValues.append(proportion_bad)

            # Find predictions
            if proportion_bad >= threshold:
                prediction.append(class_bad)
                pred_class = 'Bad prognosis'
            else:
                prediction.append(1 - class_bad)
                pred_class = 'Good prognosis'

            print('WSI: ' + name)
            print('Number of tiles predicted as bad prognosis: ' + str(numb_bad) + '/' + str(len(predTiles)))
            print('Proportion of tiles predicted as bad prognosis: ' + str(proportion_bad))
            print('Predicted class: ' + pred_class + '\n')

            predictonString += 'WSI: ' + name + '\n' \
            + 'Number of tiles predicted as bad prognosis: ' + str(numb_bad) + '/' + str(len(predTiles)) + '\n' \
            + 'Proportion of tiles predicted as bad prognosis: ' + str(proportion_bad) + '\n' \
            + 'Predicted class: ' + pred_class + '\n\n'

    with open(text_path + '/' + prediction_name + '.txt', 'w') as f:
        f.write(predictonString)
