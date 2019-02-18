import torch
from dataset import solar_dataset
from model import solar_classifier
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

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
            hists[idx] = output.cpu().numpy()

    if features==None:
        features_dict.update({
            "filaments" : idxs[np.where(labels==0)],
            "flares" : idxs[np.where(labels==1)],
            "prominences" : idxs[np.where(labels==2)],
            "quiet" : idxs[np.where(labels==3)],
            "sunspots" : idxs[np.where(labels==4)]
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
            features_dict.update({f : idxs[np.where(labels==label_dict[f])]})
            hist_dict.update({f : hists[np.where(labels==label_dict[f])]})

    return features_dict, hist_dict