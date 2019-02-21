import torch
from dataset import solar_dataset
from model import solar_classifier
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sunpy.cm as cm
import matplotlib.pyplot as plt

def softmax(x):
    '''
    A function for calculating the softmax probability an output vector of the network.
    
    Parameters
    ----------
    x : numpy array
        A numpy array containing the pre-softmax probabilities i.e. the output of the network.
    '''
    ex = np.exp(x)
    return ex / ex.sum()

def solar_classification(weights,data,features=None):
    dataset = solar_dataset(source="numpy",data_arr=data,test=True)
    idxs = np.zeros(dataset.__len__())
    labels = np.zeros(dataset.__len__())
    hists = np.zeros((dataset.__len__(),5))
    features_dict = {}
    hist_dict = {}
    label_dict = {
        "filaments" : 0,
        "flares" : 1,
        "prominences" : 2,
        "quiet" : 3,
        "sunspots" : 4
    }
    data_loader = DataLoader(dataset,batch_size=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = solar_classifier()
    model.to(device)
    model.load_state_dict(torch.load(weights,map_location=device))
    model.eval()

    with torch.no_grad():
        for idx, images in tqdm(enumerate(data_loader),desc="Classifying images"):
            images = images.float().to(device)
            output = model(images)
            _, predicted = torch.max(output.data,1)
            idxs[idx] = idx
            labels[idx] = predicted.item()
            hists[idx] = softmax(output.cpu().numpy())

    if features==None:
        features_dict.update({
            "filaments" : idxs[np.where(labels==0)].astype(np.int16),
            "flares" : idxs[np.where(labels==1)].astype(np.int16),
            "prominences" : idxs[np.where(labels==2)].astype(np.int16),
            "quiet" : idxs[np.where(labels==3)].astype(np.int16),
            "sunspots" : idxs[np.where(labels==4)].astype(np.int16)
        })

        hist_dict.update({
            "filaments" : hists[np.where(labels==0)],
            "flares" : hists[np.where(labels==1)],
            "prominences" : hists[np.where(labels==2)],
            "quiet" : hists[np.where(labels==3)],
            "sunspots" : hists[np.where(labels==4)]
        })
    else:
        for f in features:
            features_dict.update({f : idxs[np.where(labels==label_dict[f])].astype(np.int16)})
            hist_dict.update({f : hists[np.where(labels==label_dict[f])]})

    return features_dict, hist_dict

def plot_image(data,feature_dict,feature,idx,cmap="hinodesotintensity"):
    if len(data.shape) == 4:
        data = data.squeeze()
    plt.imshow(data[feature_dict[feature][idx]],origin="bottom",cmap=cmap)

def plot_hist(hist_dict,feature,idx):
    hist_dict[feature][idx][np.isnan(hist_dict[feature][idx])] = 1
    plt.bar(np.arange(5),hist_dict[feature][idx],tick_label=["Filaments","Flare ribbon","Prominence","Quiet Sun","Sunspot"])
    plt.yscale("log")
    plt.ylim(plt.ylim()[0],1)
    plt.ylabel("Normalised probability for each class label")
