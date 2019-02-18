import torch
from dataset import solar_dataset
from model import solar_classifier
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def solar_classification(model,device,weights,data,features=None):
    dataset = solar_dataset(source="numpy",data_arr=data)
    labels = np.zeros(dataset.__len__())
    data_loader = DataLoader(dataset,batch_size=1)

    model = solar_classifier()
    model.to(device)
    model.load_state_dict(torch.load(weights),map_location=device)
    model.eval()

    with torch.no_grad():
        for i, images in enumerate(data_loader):
            images = images.float().to(device)
            output = model(images)
            _, predicted = torch.max(output.data,1)
            labels[i] = predicted.item()

    if features==None:
        return labels
    else:
        