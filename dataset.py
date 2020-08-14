import numpy as np
from torch.utils.data import Dataset

class SolarDataset(Dataset):
    def __init__(self,source="from_file",dat_file=None,data_arr=None,label_arr=None,test=False):
        super().__init__()
        self.test = test
        if not self.test:
            if source == "from_file":
                if dat_file is None:
                    raise TypeError("dat_file should not be done when initialising a class instance from a file!")
                f = np.load(dat_file)
                if len(f.files) == 2:
                    self.header = f["header"]
                self.label = f["data"][:,0]
                self.length = f["data"].shape[0]
                self.data = f["data"][:,1:].reshape(self.length,1,256,256)
                del f
            elif source == "numpy":
                self.label = label_arr
                self.length = label_arr.shape[0]
                self.data = data_arr.reshape(self.length,1,256,256)
            else:
                raise TypeError("Invalid source format.")
        else:
            self.data = data_arr

    def __len__(self):
        if not self.test:
            return len(self.label)
        else:
            return self.data.shape[0]

    def __getitem__(self,idx):
        if not self.test:
            if hasattr(self,"header"):
                item = self.data[idx]
                label = self.label[idx]
                header = self.header[idx]

                return (item,label,header)
            else:
                item = self.data[idx]
                label = self.label[idx]

                return (item,label)
        else:
            item = self.data[idx]

            return item

def parse_dataset(dataset):
    '''
    A function for splitting the dataset into the different classes for use in the confusion matrix.

    Parameters
    ----------
    dataset : solar_dataset or str
        A solar dataset class instance or path to the file containing images of the different classes.

    Returns
    -------
    sol_list : list
        A list of solar_dataset instances with dependent length on the length of np.unique(dataset.label).
    '''

    sol_list = []
    
    if type(dataset) == solar_dataset:
        for l in np.unique(dataset.label):
            args = np.argwhere(dataset.label == l) #finds all of the indices where the given label is l and returns an array of these indices
    
            sol_list.append(solar_dataset(source="numpy",data_arr=dataset.data[args],label_arr=dataset.label[args]))

        return sol_list
    elif type(dataset) == str:
        s_dataset = solar_dataset(source="from_file",dat_file=dataset)

        for l in np.unique(s_dataset.label):
            args = np.argwhere(s_dataset.label == l)

            sol_list.append(solar_dataset(source="numpy",data_arr=s_dataset.data[args],label_arr=s_dataset.label[args]))

        return sol_list
