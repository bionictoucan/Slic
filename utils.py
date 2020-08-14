import torch
from dataset import SolarDataset
from model import SolarClassifier
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sunpy.cm as cm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os, html
from astropy.io import fits
import sunpy.map as m
from skimage.transform import resize
        
class Classification:
    def __init__(self,fits_pth,weights):
        '''
        This is a class for classifying solar images quickly to be used with the pre-trained Slic network.

        Parameters
        ----------
        fits_pth : str
            This is the path to the fits files to be classified.
        weights : str
            This is the path to the pretrained model.

        Attributes
        ----------
        files : list
            A list of the image files to be classified.
        weights : str
            The path to the pretrained model.
        class_dict : dict
            A dictionary to store the classifications of the image in. The numbers in the entries of the dictionary correspond to the index in the file list meaning that file contains its associated feature.
        hist_dict : dict
            A dictionary to store the probability histograms in the same format as class_dict.
        label_dict : dict
            A dictionary to store the corresponding numerical class labels with what they correspond to physically.
        '''
        self.files = sorted([fits_pth+x for x in os.listdir(fits_pth)]) #this assumes that the fits files are named sensibly
        self.weights = weights
        self.class_dict = {}
        self.hist_dict = {}
        self.label_dict = {
            "filaments" : 0,
            "flares" : 1,
            "prominences" : 2,
            "quiet" : 3,
            "sunspots" : 4
        }

    def solar_classification(self,features=None,freedom=False):
        '''
        This class method does the classificiation of the images in the files attribute and updates the class_dict and hist_dict attributes with the classification of the images and their probability histograms respectively.

        Parameters
        ----------
        features : None, list
            The features we would like to keep after classifying. The default is None which will lead to all feature data being saved.
        freedom : bool
            This garbage collects the model after usage to free up memory. Default is false.
        '''
        im_arr = np.zeros((len(self.files),1,256,256)) #this sets up the array of images to be classified

        for i, image in enumerate(self.files):
            tmp = fits.getdata(image).astype(np.float64)
            tmp = resize(tmp,(256,256),anti_aliasing=True)
            tmp = tmp.reshape(1,256,256)
            im_arr[i] = tmp

        dataset = SolarDataset(source="numpy",data_arr=im_arr,test=True)
        idxs = np.zeros(dataset.__len__())
        labels = np.zeros(dataset.__len__())
        hists = np.zeros((dataset.__len__(),5))
        data_loader = DataLoader(dataset,batch_size=1)
        device = ("cuda:0" if torch.cuda.is_available() else "cpu")

        model = SolarClassifier()
        model.to(device)
        model.load_state_dict(torch.load(self.weights,map_location=device))
        model.eval()

        with torch.no_grad():
            for idx, images in tqdm(enumerate(data_loader),desc="Classifying images"):
                images = images.float().to(device)
                output = model(images)
                _, predicted = torch.max(output.data,1)
                idxs[idx] = idx
                labels[idx] = predicted.item()
                hists[idx] = F.softmax(output.cpu()).numpy()

        if features == None:
            self.class_dict.update({
                "filaments" : idxs[np.where(labels==0)].astype(np.int16),
                "flares" : idxs[np.where(labels==1)].astype(np.int16),
                "prominences" : idxs[np.where(labels==2)].astype(np.int16),
                "quiet" : idxs[np.where(labels==3)].astype(np.int16),
                "sunspots" : idxs[np.where(labels==4)].astype(np.int16)
            })

            self.hist_dict.update({
                "filaments" : hists[np.where(labels==0)],
                "flares" : hists[np.where(labels==1)],
                "prominences" : hists[np.where(labels==2)],
                "quiet" : hists[np.where(labels==3)],
                "sunspots" : hists[np.where(labels==4)]
            })
        else:
            for f in features:
                self.class_dict.update({f : idxs[np.where(labels==self.label_dict[f])].astype(np.int16)})
                self.hist_dict.update({f : hists[np.where(labels==self.label_dict[f])]})

        if freedom:
            del(model)

    def plot_image(self,feature,idx=None):
        '''
        This is a class method to plot the images that we are interested in after classification.

        Parameters
        ----------
        feature : str
            The feature we want to look at.
        idx : None, list
            A list of the indices we want to look at. These are the numerical values assigned to the files via their index in the files attribute. Default is None which means the code will plot all of a single class.
        '''
        fig = plt.figure()
        if idx == None:
            idx = list(self.class_dict[feature])
        if type(idx) == list:
            fig_side = np.sqrt(len(idx))
            for j, i in enumerate(idx):
                ax = fig.add_subplot(np.floor(fig_side),np.ceil(fig_side),j+1)
                im = m.Map(self.files[i])
                im.plot_settings["title"] = im.meta["detector"] + " " + im.meta["wave"][3:] + html.unescape("&#8491;") +" " + im.meta["date-obs"][:10] + " " + im.meta["date-obs"][11:-4]
                im.plot()
                ax.set_ylabel("Solar-Y [arcsec]")
                ax.set_xlabel("Solar-X [arcsec]")
        else:
            raise TypeError("Indices should be a list.")

        fig.tight_layout()

    def plot_hist(self,feature,idx):
        '''
        This is a class method to plot the probability distributions that we are interested in after classification.

        Parameters
        ----------
        feature : str
            The feature we want to see the distribution for.
        idx : None, list
            A list of the indices we want to look at. These are the numerical values assigned to the files via their index in the files attribute. Default is None which means the code will plot all of the probability distributions for a single class.
        '''
        fig = plt.figure()
        if idx == None:
            idx = list(self.class_dict[feature])
        if type(idx) == list:
            fig_side = np.sqrt(len(idx))
            for j, i in enumerate(idx):
                hist = np.where(self.class_dict[feature] == i)[0]
                ax = fig.add_subplot(np.floor(fig_side),np.ceil(fig_side),j+1)
                ax.bar(range(5),self.hist_dict[feature][hist][0],tick_label=list(self.label_dict.keys()))
                ax.set_yscale("log")
                ax.tick_params(axis="x",labelrotation=90)
        else:
            raise TypeError("Indices should be a list.")

        fig.tight_layout()
