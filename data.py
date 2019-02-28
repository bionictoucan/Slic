import numpy as np
from astropy.io.fits import getdata
import os,argparse
from scipy.misc import imresize
from tqdm import tqdm

def train_test_data(dataset,percentage_split=10,save_dir="./"):
    '''
    Parameters
    ----------
    dataset : str
        The path to the dataset to be prepped.
    percentage_split : int
        The percentage to be used in the validation. Default is 10.
    save_dir : str
        The directory to save the files to. Default is the current working directory.
    '''

    if dataset is None:
        raise TypeError("Please tell us what dataset you would like prepped!")
    
    class bad_data(Exception): pass

    dir_list = sorted([dataset+x for x in os.listdir(dataset) if not x.startswith(".")]) #generates a list of the dataset class folders each containing the images pertaining to the eponymous class
    train_list, test_list = [], []

    for (i,direc) in enumerate(dir_list):
        data_list = sorted([direc+"/"+x for x in os.listdir(direc) if not x.startswith(".")])

        for (j, image) in tqdm(enumerate(data_list),desc=str(dir_list[i])):
            tmp = getdata(image).astype(np.float64)
            try:
                for (x,y), pixel in np.ndenumerate(tmp):
                    if tmp[x,y] == 0:
                        raise bad_data() #skip over images with faults in the data
            except bad_data:
                continue
            tmp = imresize(tmp,(256,256),interp="bicubic") #resizes the images to 256x256 pixels using bicubic interpolation
            tmp = tmp.flatten() #flatten the image to a 1D vector for easier storage in the .npz file
            tmp = np.insert(tmp,0,i)
            if (j % percentage_split) == 0:
                test_list.append(tmp)
            else:
                train_list.append(tmp)

    print("There are %d training images." % len(train_list))
    print("There are %d test images." % len(test_list))

    train_arr = np.array(train_list)
    test_arr = np.array(test_list)
    del(train_list,test_list)
    np.savez_compressed(save_dir+"solar_train_data.npz",data=train_arr)
    np.savez_compressed(save_dir+"solar_test_data.npz",data=test_arr)
    print("The training and testing data files have been created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",help="The path to the dataset to be loaded in.",default=None)
    parser.add_argument("--percent_split",help="The percentage of the dataset to put in the validation.",default=10)
    parser.add_argument("--save_dir",help="The directory to save the prepped datasets to.",default="./")
    args = parser.parse_args()

    train_test_data(dataset=args.dataset,percentage_split=args.percent_split,save_dir=args.save_dir)