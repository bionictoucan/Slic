import numpy as np
import pandas as pd
from dataset import *
from model import solar_classifier
import torch
from torch.utils.data import DataLoader

class confusion_matrix():
    '''
    A class to store the confusion matrix, its features and the associated statistics that go along with it.
    '''

    def __init__(self,val_set=None,model=None):
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        solar_c = solar_classifier().to(dev)
        solar_c.load_state_dict(torch.load(model))
        solar_c.eval()

        filaments, flares, prominences, quiet, sunspots = parse_dataset(val_set)

        label_sets = [filaments,flares,prominences,quiet,sunspots]

        self.labels = ["Filaments","Flares","Prominences","Quiet","Sunspots"]

        misclassified = []
        cm_diag = []
        for x in label_sets:
            loader = DataLoader(dataset=x,batch_size=1)
            correct = 0
            with torch.no_grad():
                for i, (image, labels) in enumerate(loader):
                    image = image.float().to(dev)
                    labels = labels.long().to(dev)
                    output = solar_c(image)
                    _, predicted = torch.max(output.data,1)
                    correct += (predicted == labels).sum().item()
                    if predicted != labels:
                        misclassified.append((i,predicted.item(),labels.item()))
                cm_diag.append(correct)
        cm = np.zeros((5,5))
        cm[np.diag_indices(5)] = cm_diag
        for m in misclassified:
            cm[m[2],m[1]] += 1

        self.cm = pd.DataFrame(data=cm,index=self.labels,columns=self.labels,dtype=np.uint16)

    def __eq__(self,other):
        return (self.cm.equals(other.cm)) and (self.labels == other.labels)

    def true_pos(self,cm,feature):
        '''
        A function for obtaining the true positives of a specific feature from the confusion matrix.
        
        Parameters
        ----------
        cm : pandas DataFrame
            A table of the confusion matrix where the top axis refers to the calculated class labels and the left axis corresponds to the true class.
        feature : str
            The name of the feature to get the true positive value for.
        
        Returns
        -------
        int
            The true positive of the desired feature.
        '''
        
        feature_dict = {
            "Filaments" : 0,
            "Flares" : 1,
            "Prominences" : 2,
            "Quiet" : 3,
            "Sunspots" : 4
        }
        
        return cm[feature][feature_dict[feature]]

    def false_pos(self,cm,feature):
        '''
        A function for obtaining the false positives of a specific feature from the confusion matrix. The false positives are indicated by the numbers down a column in the data that are not from the diagonal element of that column (i.e. where the row and column index are equal).
        
        Parameters
        ----------
        cm : pandas DataFrame
            A table of the confusion matrix where the top axis refers to the calculated class labels and the left axis corresponds to the true class.
        feature : str
            The name of the feature to get the false negative value for.
        
        Returns
        -------
        int
            The false positives of the desired feature.
        '''
        
        tp = self.true_pos(cm=cm,feature=feature)
        return cm[feature].sum() - tp

    def false_neg(self,cm,feature):
        '''
        A function for obtaining the false negatives of a specific feature from the confusion matrix. The false negatives are indicated by the numbers across a row in the data that are not from the diagonal element of that row (i.e. where the row and column index are equal).
        
        Parameters
        ----------
        cm : pandas DataFrame
            A table of the confusion matrix where the top axis refers to the calculated class labels and the left axis corresponds to the true class.
        feature : str
            The name of the feature to get the false negative for.
        
        Returns
        -------
        int
            The false negatives of the desired feature.
        '''
        
        tp = self.true_pos(cm=cm,feature=feature)
        return cm.loc[feature].sum() - tp

    def true_neg(self,cm,feature):
        '''
        A function for obtaining the true negatives of a specific feature from the confusion matrix. The true negatives are the number of samples that are not attributed to the feature in question and are not classified as that feature.
        
        Parameters
        ----------
        cm : pandas DataFrame
            A table of the confusion matrix where the top axis refers to the calculated class labels and the left axis corresponds to the true class.
        feature : str
            The name of the feature to get the true negative value for.
        
        Returns
        -------
        int
            The true negatives of the desired feature.
        '''
        
        tp = self.true_pos(cm=cm,feature=feature)
        fp = self.false_pos(cm=cm,feature=feature)
        fn = self.false_neg(cm=cm,feature=feature)
        return int(cm.values.sum() - tp - fp - fn)

    def precision(self,cm,feature):
        '''
        The precision of the classifier is the ratio
        
        TP_i / (TP_i + FP_i)
        
        i.e. it is the ratio of correct predictions for a certain feature to the total number of predictions for a certain feature.
        
        Parameters
        ----------
        cm : pandas DataFrame
            A table of the confusion matrix where the top axis refers to the calculated class labels and the left axis corresponds to the true class.
        feature : str
            The name of the feature to get the precision value for.
        
        Returns
        -------
        int
            The precision of the desired feature.
        '''
        
        tp = self.true_pos(cm=cm,feature=feature)
        fp = self.false_pos(cm=cm,feature=feature)
        
        return tp / (tp + fp)

    def recall(self,cm,feature):
        '''
        The recall of the classifier is the ratio
        
        TP_i / (TP_i + FN_i)
        
        i.e. it is the ratio of correct predictions for a certain feature to the total number of instances containing that feature regardless of correct classifications.
        
        Parameters
        ----------
        cm : pandas DataFrame
            A table of the confusion matrix where the top axis refers to the calculated class labels and the left axis corresponds to the true class.
        feature : str
            The name of the feature to get the recall value for.
        
        Returns
        -------
        int
            The recall of the desired feature.
        '''
        
        tp = self.true_pos(cm=cm,feature=feature)
        fn = self.false_neg(cm=cm,feature=feature)
        
        return tp / (tp + fn)

    def F1_score(self,cm,feature):
        '''
        The F1 score of the classifier describes the balanace between the precision and recall. It is given by
        
        2 * P * R / (P + R)
        
        Parameters
        ----------
        cm : pandas DataFrame
            A table of the confusion matrix where the top axis refers to the calculated class labels and the left axis corresponds to the true class.
        feature : str
            The name of the feature to get the F1 score value for.
        
        Returns
        -------
        int
            The F1 score of the desired feature.
        '''
        
        p = self.precision(cm=cm,feature=feature)
        r = self.recall(cm=cm,feature=feature)
        
        return 2 * p * r / (p + r)