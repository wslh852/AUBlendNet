import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa
from collections import defaultdict
from torch.utils import data 
import pandas as pd
import re 


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train",read_audio=False):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.read_audio = read_audio
        #self.facs2onehot = {'1':0,'2':1,'4':2,'5':3,'6':4,'7':5,'9':6,'10':7,'11':8,'12':9,'14':10,'15':11,'16':12,'17':13,'18':14,'20':15,'22':16,'23':17,'24':18,'25':19,'26':20,'27':21,'28':22,'29':23,'30R':24,'30L':25,'33':26,'45':27,'61':28,'62':29,'63':30,'64':31}
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
    
        vertice = self.data[index]["vertice"]
        template = self.data[index]["temp"]
        name = self.data[index]['name']
        onehot = np.eye(33)

        return torch.FloatTensor(vertice), torch.FloatTensor(template),name ,torch.FloatTensor(onehot)

    def __len__(self):
        return self.len
    
def create_dict():
    return defaultdict(create_dict)

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []
    for file in tqdm(sorted(os.listdir(args.data_root))):
        file_path = os.path.join(args.data_root,file)
        #au_dic={'AU1':0,'AU2':1,'AU4':2,'AU5':3,'AU6':4,'AU7':5,'AU9':6,'AU10':7,'AU11':8,'AU12':9,'AU14':10,'AU15':11,'AU16':12,'AU17':13,'AU18':14,'AU20':15,'AU22':16,'AU23':17,'AU24':18,'AU25':19,'AU26':20,'AU27':21,'AU28':22,'AU29':23,'AU30L':24,'AU30R':25,'AU33':26,'AU45':27,'AU61':28,'AU62':29,'AU63':30,'AU64':31}
        key = file
        vertice_path = os.path.join(file_path,'new_AU_vertice32.npy')
        vertice = np.load(vertice_path).reshape(-1,15069)
        temp_path = os.path.join(file_path,'template.npy')
        temp = np.load(temp_path).reshape(-1,15069)[0:1]
        vertice = np.concatenate((temp,vertice),axis=0)
        data[key]['temp'] = temp[0]
        data[key]['vertice'] = vertice
        data[key]['name'] = file

    subjects_dict = {}
    subjects_dict["train"] = sorted(os.listdir(args.data_root))[:400]
    subjects_dict["val"] = sorted(os.listdir(args.data_root))[400:450]
    subjects_dict["test"] = sorted(os.listdir(args.data_root))[1]

    for k, v in data.items():
        subject_id = re.findall(r'\d+', k)[0]
       # sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"]:
            train_data.append(v)
        if subject_id in subjects_dict["val"]:
            valid_data.append(v)
        if subject_id in subjects_dict["test"]:
       # if subject_id  == subjects_dict["test"]:
            test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))

    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data,subjects_dict,"train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_data = Dataset(valid_data,subjects_dict,"val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False, num_workers=args.workers)
    test_data = Dataset(test_data,subjects_dict,"test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=args.workers)
    return dataset

if __name__ == "__main__":
    get_dataloaders()