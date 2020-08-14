import numpy as np
import pandas as pd
from dataset import *
from model import solar_classifier
import torch
from torch.utils.data import DataLoader

class ConfusionMatrix():
    '''
    A class to store the confusion matrix, its features and the associated statistics that go along with it.

    Parameters
    ----------
    val_set : SolarDataset
        A SolarDataset instance for the data to calculate the confusion matrix for.
    model : ??
        The model to calculate the confusion matrix for.
    model_pth : str
        The path to the trained weights of the model to calculate the confusion matrix for.
    labels : list
        A names for the class labels.
    '''

    def __init__(self, val_set, model, model_pth, labels):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)
        model.load_state_dict(torch.load(model_pth)["model"])
        model.eval()

        label_sets = self.parse_dataset(val_set)
        self.labels = labels
        self.feature_dict = {self.labels[j] : j for j in range(len(self.labels))}

        misclassified = []
        cm_diag = []
        for x in label_sets:
            loader = DataLoader(dataset=x, batch_size=1)
            correct = 0
            with torch.no_grad():
                for j, (image, label) in enumerate(loader):
                    image, label = image.float().to(self.device), label.long().to(self.device)
                    output = model(image)
                    _, predicted = torch.max(output.data, 1)
                    correct += (predicted == label).sum().item()
                    if predicted != label:
                        misclassified.append((j, predicted.item(), label.item()))
                        #save the index of the misclassification and what was predicted and what the true label is
                cm_diag.append(correct)
        
        cm = torch.diag(torch.tensor(cm_diag)) #this creates a square tensor with the diagonal elements filled
        for m in misclassified:
            # in confusion matrix, y-axis is true class, x-axis is predicted class
            # thus, elements with true class l and predicted class k will add one count to cm[l,k]
            cm[m[2], m[1]] += 1

        self.cm_pd = pd.DataFrame(data=cm.numpy(), index=self.labels, columns=self.labels, dtype=np.uint16)
        self.cm = cm

    def true_pos(self, feature):
        '''
        A function for obtaining the true positives of a specific feature from the confusion matrix.

        Parameters
        ----------
        feature : str
            The name of the feature to get the true positive result for.

        Returns
        -------
        int
            The true positive of the desired feature.
        '''

        feature = feature.lower()
        idx = self.feature_dict[feature]

        return self.cm[idx, idx].item()

    def false_pos(self, feature):
        '''
        A function for obtaining the false positives of a specific feature from the confusion matrix. The false positives are indicated by the numbers down a column in the data that are not from the diagonal element of that column (i.e. where the row and column index are equal).

        Parameters
        ----------
        feature : str
            The name of the feature to get the false positive result for.

        Returns
        -------
        int
            The false positive of the desired feature.
        '''

        feautre = feature.lower()
        idx = self.feature_dict[feature]

        tp = self.true_pos(feature=feature)
        return self.cm[:, idx].sum().item() - tp

    def false_neg(self, feature):
        '''
        A function for obtaining the false negatives of a specific feature from the confusion matrix. The false negatives are indicated by the numbers across a row in the data that are not from the diagonal element of that row (i.e. where the row and column index are equal).

        Parameters
        ----------
        feature : str
            The name of the feature to get the false negative result for.

        Returns
        -------
        int
            The false negative of the desired feature.
        '''

        feature = feature.lower()
        idx = self.feature_dict[feature]

        tp = self.true_pos(feature=feature)
        return self.cm[idx].sum().item() - tp

    def true_neg(self, feature):
        '''
        A function for obtaining the true negatives of a specific feature from the confusion matrix. The true negatives are the number of samples that are not attributed to the feature in question and are not classified as that feature.

        Parameters
        ----------
        feature : str
            The name of the feature to get the true negative for.

        Returns
        -------
        int
            The true negative of the desired feature.
        '''

        feature = feature.lower()
        idx = self.feature_dict[feature]

        tp = self.true_pos(feature=feature)
        fp = self.false_pos(feature=feature)
        fn = self.false_neg(feature=feature)
        return cm.sum().item() - tp - fp - fn

    def precision(self, feature):
        '''
        The precision of the classifier is the ratio

        TP_j / (TP_j + FP_j)

        i.e. it is the ratio of correct predictions for a certain feature to the total number of predictions for a certain feature.

        Parameters
        ----------
        feature : str
            The name of the feature to get the precision value for.

        Returns
        -------
        int
            The precision of the desired feature.
        '''

        feature = feature.lower()
        idx = self.feature_dict[feature]

        tp = self.true_pos(feature=feature)
        fp = self.false_pos(feature=feature)

        return tp / (tp + fp)

    def recall(self, feature):
        '''
        The recall of the classifier is the ratio

        TP_j / (TP_j + FN_j)

        i.e. it is the ratio of the correct predictions for a certain feature to the total number of instances containing that feature regardless of correct classifications.

        Parameters
        ----------
        feature : str
            The name of the feature to get the recall value for.

        Returns
        -------
        int
            The recall of the desired feature.
        '''

        feature = feature.lower()
        idx = self.feature_dict[feature]

        tp = self.true_pos(feature=feature)
        fn = self.false_neg(feature=feature)

        return tp / (tp + fn)

    def F1_score(self, feature):
        '''
        The F1 score of the classifier describes the balance between the precision and recall. It is given by

        2 * P * R / (P + R)

        Parameters
        ----------
        feature : str
            The name of the feature to get the F1 score value for.

        Returns
        -------
        int
            The F1 score of the desired feature.
        '''

        feature = feature.lower()
        idx = self.feature_dict[feature]

        p = self.precision(feature=feature)
        r = self.recall(feature=feature)

        return 2 * p * r / (p + r)

    @staticmethod
    def parse_dataset(dataset):
        '''
        A function for splitting the dataset into the different classes for use in the confusion matrix.

        Parameters
        ----------
        dataset : EODataset
            A EODataset object containing the images and labels for classification.

        Returns
        -------
        tens_list : list
            A list of EODataset objects with dependent length on the length of torch.unique(dataset.label).
        '''

        tens_list = []

        for l in torch.unique(dataset.labels):
            args = torch.nonzero(dataset.labels == l).squeeze()

            tens_list.append(EODataset(dataset.data[args], dataset.labels[args], transform=dataset.transform, transform_prob=dataset.transform_prob))

        return tens_list

    def plot_confusion_matrix(self, title=None, cmap=plt.cm.Blues, normalise=False):
        '''
        This function plots the confusion matrix.

        Parameters
        ----------
        title : str or None
            The title of the plot.
        cmap : str or cm
            The colour map to use for the plot of the confusion matrix.
        normalise : bool
            Whether or not to normalise the data in the confusion matrix based on the number of samples from each label. Default is False.
        '''

        if normalise:
            cm = self.cm.numpy() / self.cm.numpy().sum(axis=1)[:, np.newaxis]
        else:
            cm = self.cm

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(self.cm.size(1)),
               yticks=np.arange(self.cm.size(0)),
               xticklabels=self.labels, yticklabels=self.labels,
               title=title,
               ylabel="True label",
               xlabel="Predicted label")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        fmt = ".3f" if normalise else "d"
        thresh = self.cm.max() / 2
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        return ax
