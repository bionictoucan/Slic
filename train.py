import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dataset import SolarDataset
from model import SolarClassifier
import argparse
from tqdm import tqdm

def train(model,device,data_loader,optimiser,epoch,criterion):
    model.to(device)
    model.train()

    for i, (images, labels) in tqdm(enumerate(data_loader),desc="Epoch no."+str(epoch)):
        images, labels = images.float().to(device), labels.long().to(device) #casts the tensors to the GPU if available
        optimiser.zero_grad() #must zero the gradients in the optimiser since backward() accumulates gradients and this stops mixing of values between batches
        output = model(images) #feeds the data through the network
        loss = criterion(output,labels) #finds the distance in the loss space between predicted values and actual values
        loss.backward()
        optimiser.step()

def validate(model,device,data_loader,epoch,test_losses):
    model.to(device)
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.float().to(device), labels.long().to(device)
            output = model(images)
            _, predicted = torch.max(output.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    print("Test Accuracy of the model on the test images: %f %% on epoch %d" % (100 * correct / total, epoch))
    test_losses.append(correct / total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",help="The learning rate for the network.",default=0.0001,type=float)
    parser.add_argument("--n_epochs",help="The number of epochs to train for.",default=100,type=int)
    parser.add_argument("--batch_size",help="The batch size to use for training and validation.",default=2,type=int)
    parser.add_argument("--no_gpu",help="Don't use the GPU.",dest='use_gpu',action='store_false')
    parser.add_argument("--use_dataparallel",help="Use DataParallel to parallelise across multiple GPUs.",dest='use_dataparallel',action='store_true')
    parser.add_argument("--train_data",help="The path to the training data.",default="./solar_train_data.npz")
    parser.add_argument("--val_data",help="The path to the validation data.",default="./solar_test_data.npz")
    parser.add_argument("--save_dir",help="The directory to save the models from each epoch.",default="./")
    parser.set_defaults(use_gpu=True, use_dataparallel=False)
    args = parser.parse_args()

<<<<<<< HEAD
    device = torch.device("cuda:0" if torch.cuda.is_available and args.use_gpu else "cpu")
    sol_clas = SolarClassifier() #creates an instance of the solar classification network
=======
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
    sol_clas = Solar_Classifier() #creates an instance of the solar classification network
    if args.use_gpu and torch.cuda.device_count() > 1 and args.use_dataparallel:
        print("Using %d GPUs!" % torch.cuda.device_count())
        sol_clas = nn.DataParallel(sol_clas)

>>>>>>> 93fdf38f917daec0d142d0be8c9ecbd38377703e
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(sol_clas.parameters(),args.lr,momentum=0.9,nesterov=True)

    test_losses = []
    train_dataset = SolarDataset(dat_file=args.train_data)
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True)
    val_dataset = SolarDataset(dat_file=args.val_data)
    val_loader = DataLoader(dataset=val_dataset,batch_size=args.batch_size,shuffle=True)
    del(train_dataset,val_dataset)

    for i in tqdm(range(1,args.n_epochs+1)):
        train(sol_clas,device,train_loader,optimiser,i,criterion)
        torch.save(sol_clas.state_dict(),args.save_dir+"sol_class_"+str(i)+".pth")
        validate(sol_clas,"cpu",val_loader,i,test_losses)
    np.save("loss"+str(args.lr)+".npy",np.array(test_losses))
