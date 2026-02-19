import numpy as np
import os
import tensorflow as tf
import re

cc_train_dataset = 1
cc_val_dataset = 1

def get_filenames(data_dir):
    data_list = os.listdir(data_dir)                            
    data_list.sort(key=lambda x: int(re.split('[._]', x)[-2])) 
    name_list = []
    for example in data_list:
        name_list.append(os.path.join(data_dir, example))

    return name_list


def datasetflow_reader(batch_size=128, shuffer_1=10000, shuffer_2=3000, layers=12):
    cc_1, cc_2 = [], []    
    for i in range(layers):
        cc_1.append(cc_train_dataset[i].shuffle(shuffer_1).batch(batch_size))
        cc_2.append(cc_val_dataset[i].shuffle(shuffer_2).batch(batch_size))
    return cc_1, cc_2

