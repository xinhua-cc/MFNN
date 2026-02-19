import os
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split

def get_filenames(data_path):
    data_list = os.listdir(data_path) # Get a list of file names
    data_list.sort(key=lambda x: int(re.split('[._]', x)[-2])) # Split the list, sort it in ascending order by serial number
    name_list = []
    for m_name in data_list:
        name_list.append(os.path.join(data_path, m_name))
    return name_list


def read_sensi(sensi_path, AreaNum, AreaName, Number):
    sensi = []
    for i in range(AreaNum):
        fieldname = AreaName[i]
        for j in range(Number):
            sensi_ = np.loadtxt(sensi_path + fieldname + '_' + str(j + 1) + '.txt')
            sensi.append(sensi_)
    return sensi


def read_files(inputDir, labelDir, sensiDir, denDir, vpDir, AreaNum, AreaName, Number, layers, fvNum):
    labels = np.zeros((layers, Number * AreaNum))
    inputs = np.zeros((layers, Number * AreaNum, fvNum, 5))
    VS = np.zeros((AreaNum, layers, 5))
    for h in range(layers):
        input_ = np.zeros((Number * AreaNum, fvNum, 5))
        label_ = np.zeros((Number * AreaNum))
        for i in range(AreaNum):
            field_name = AreaName[i]
            path1 = os.path.join(inputDir, field_name)
            path2 = os.path.join(labelDir, field_name)
            path3 = os.path.join(sensiDir, field_name)
            path4 = os.path.join(denDir, field_name)
            path5 = os.path.join(vpDir, field_name)
            for j in range(Number):
                p1 = path1 + '_' + str(j + 1) + '.txt'
                p2 = path2 + '_' + str(j + 1) + '.txt'
                p3 = path3 + '_' + str(j + 1) + '.txt'
                p4 = path4 + '_' + str(j + 1) + '.txt'
                p5 = path5 + '_' + str(j + 1) + '.txt'
                fv = np.loadtxt(p1)
                vs = np.loadtxt(p2)
                sensi = np.loadtxt(p3)
                den = np.loadtxt(p4)
                vp = np.loadtxt(p5)
                input_[i * Number + j, :, 0:2] = fv.copy()
                input_[i * Number + j, :, 2] = sensi[:, h].copy()
                input_[i * Number + j, :, 3] = den.copy()
                input_[i * Number + j, :, 4] = vp.copy()   
                label_[i * Number + j] = vs[h].copy();            
        inputs[h, :, :, :] = input_
        labels[h, :] = label_
     
    m_max0 = np.max(inputs[:, :, :, 0])
    m_min0 = np.min(inputs[:, :, :, 0])
    m_dif0 = m_max0 - m_min0
    m_max1 = np.max(inputs[:, :, :, 1])
    m_min1 = np.min(inputs[:, :, :, 1])
    m_dif1 = m_max1 - m_min1
    m_max2 = np.max(inputs[:, :, :, 2])
    m_min2 = np.min(inputs[:, :, :, 2])
    m_dif2 = m_max2 - m_min2
    m_max3 = np.max(inputs[:, :, 0:layers, 3])
    m_min3 = np.min(inputs[:, :, 0:layers, 3])
    m_dif3 = m_max3 - m_min3
    m_max4 = np.max(inputs[:, :, 0:layers, 4])
    m_min4 = np.min(inputs[:, :, 0:layers, 4])
    m_dif4 = m_max4 - m_min4
    
    data=open("Parameters_all.txt",'w+')
    print(m_min0, file = data)
    print(m_dif0, file = data)
    print(m_min1, file = data)
    print(m_dif1, file = data)
    print(m_min2, file = data)
    print(m_dif2, file = data)
    print(m_min3, file = data)
    print(m_dif3, file = data)
    print(m_min4, file = data)
    print(m_dif4, file = data)
    data.close()  
    
    inputs[:, :, :, 0] = (inputs[:, :, :, 0] - m_min0) / m_dif0
    inputs[:, :, :, 1] = (inputs[:, :, :, 1] - m_min1) / m_dif1 
    inputs[:, :, :, 2] = (inputs[:, :, :, 2] - m_min2) / m_dif2
    inputs[:, :, 0:layers, 3] = (inputs[:, :, 0:layers, 3] - m_min3) / m_dif3 
    inputs[:, :, 0:layers, 4] = (inputs[:, :, 0:layers, 4] - m_min4) / m_dif4         
    labels = labels.transpose([1, 0])
    inputs = inputs.transpose([1, 0, 2, 3])
    # shuffle data
    state = np.random.get_state()
    np.random.shuffle(inputs)
    np.random.set_state(state)
    np.random.shuffle(labels)
    inputs = inputs.transpose([1, 0, 2, 3])
    return inputs, labels


def read_files_one(inputDir, labelDir, sensi, index, Number, layers, fvNum):
    AreaNum = 1
    labels = np.zeros((layers, Number * AreaNum))
    inputs = np.zeros((layers, Number * AreaNum, fvNum, 5))
    VS = np.zeros((AreaNum, layers, 5))
    for i in range(AreaNum):
        VS[i, :, :] = np.loadtxt('/data/field_work/data/vs_type/VS_' + str(index) + '.txt')
    for h in range(layers):
        input_ = np.zeros((Number * AreaNum, fvNum, 5))
        label_ = np.zeros((Number * AreaNum))
        for i in range(AreaNum):
            path1 = os.path.join(inputDir, str(index))
            path2 = os.path.join(labelDir, str(index))
            for j in range(Number):
                p1 = path1 + '_' + str(j + 1) + '.txt'
                p2 = path2 + '_' + str(j + 1) + '.txt'
                T_input = np.loadtxt(p1)
                TT_label = np.loadtxt(p2)
                T_label = TT_label[h]
                input_[i * Number + j, :, 0:2] = T_input
                input_[i * Number + j, :, 2] = sensi[index-1][:, h]
                for k in range(layers):
                    input_[i * Number + j, k, 3:5] = VS[i, k, 3:5]
                label_[i * Number + j] = T_label;            
        inputs[h, :, :, :] = input_
        labels[h, :] = label_
     
    m_max0 = np.max(inputs[:, :, :, 0])
    m_min0 = np.min(inputs[:, :, :, 0])
    m_dif0 = m_max0 - m_min0
    m_max1 = np.max(inputs[:, :, :, 1])
    m_min1 = np.min(inputs[:, :, :, 1])
    m_dif1 = m_max1 - m_min1
    m_max2 = np.max(inputs[:, :, :, 2])
    m_min2 = np.min(inputs[:, :, :, 2])
    m_dif2 = m_max2 - m_min2
    m_max3 = np.max(inputs[:, :, 0:layers, 3])
    m_min3 = np.min(inputs[:, :, 0:layers, 3])
    m_dif3 = m_max3 - m_min3
    m_max4 = np.max(inputs[:, :, 0:layers, 4])
    m_min4 = np.min(inputs[:, :, 0:layers, 4])
    m_dif4 = m_max4 - m_min4
    
    data=open("Parameters_all.txt",'w+')
    print(m_min0, file = data)
    print(m_dif0, file = data)
    print(m_min1, file = data)
    print(m_dif1, file = data)
    print(m_min2, file = data)
    print(m_dif2, file = data)
    print(m_min3, file = data)
    print(m_dif3, file = data)
    print(m_min4, file = data)
    print(m_dif4, file = data)
    data.close()  
    
    inputs[:, :, :, 0] = (inputs[:, :, :, 0] - m_min0) / m_dif0
    inputs[:, :, :, 1] = (inputs[:, :, :, 1] - m_min1) / m_dif1 
    inputs[:, :, :, 2] = (inputs[:, :, :, 2] - m_min2) / m_dif2
    inputs[:, :, 0:layers, 3] = (inputs[:, :, 0:layers, 3] - m_min3) / m_dif3
    if(m_dif4 != 0):
        inputs[:, :, 0:layers, 4] = (inputs[:, :, 0:layers, 4] - m_min4) / m_dif4         
    labels = labels.transpose([1, 0])
    inputs = inputs.transpose([1, 0, 2, 3])
    # shuffle data
    state = np.random.get_state()
    np.random.shuffle(inputs)
    np.random.set_state(state)
    np.random.shuffle(labels)
    inputs = inputs.transpose([1, 0, 2, 3])
    return inputs, labels