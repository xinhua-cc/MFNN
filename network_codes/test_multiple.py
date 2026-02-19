import numpy as np
import tensorflow as tf
import os
from network import model_simulate

weights_dir = "/data/field_work/multiple-Withsensi_2/data/checkpoint_125_4/"
pre_dir = "/data/field_work/multiple-Withsensi_2/data/prediction_multiple/"

input_dir = "/data/field_work/data/test_fv/"
sensi_dir = "/data/field_work/data/sensi/"
vs_dir = "/data/field_work/data/vs_type/"
layers = 8

parameter_ = np.loadtxt('/data/field_work/multiple-Withsensi_2/network_combine/Parameters_all.txt')

# load model
model = model_simulate()
print("-------model weights load success-----")

weights_list = os.listdir(weights_dir)
for weights_name in weights_list:
    pre_path = os.path.join(pre_dir, weights_name)

    Exists = os.path.exists(pre_path)
    if not Exists:
        os.makedirs(pre_path)
    
    model.load_weights(os.path.join(weights_dir, weights_name))
    data_list = os.listdir(input_dir)
    for example in data_list:
        input_raw = np.loadtxt(os.path.join(input_dir, example))
        [fvNum, c] = np.shape(input_raw)
        p_list = example.split('_', 1)
        sensi_path = sensi_dir + 'sensi_' + p_list[0] + '.txt'
        sensi = np.loadtxt(sensi_path)
        vs_path = vs_dir + 'VS_' + p_list[0] + '.txt'
        vs = np.loadtxt(vs_path)
        input_ = np.zeros((layers, fvNum, c+3))
        for k in range(layers):
            input_tep = np.copy(input_raw)
            input_tep[:, 0] = (input_tep[:, 0] - parameter_[0]) / parameter_[1]
            input_tep[:, 1] = (input_tep[:, 1] - parameter_[2]) / parameter_[3]
            sensi[:, k] = (sensi[:, k] - parameter_[4]) / parameter_[5]
            input_[k, :, 0:2] = input_tep
            input_[k, :, 2] = sensi[:, k]
            input_[k, 0:layers, 3] = vs[:, 4]
            input_[k, 0:layers, 4] = vs[:, 3]
            # input_[k, 0:layers, 3:5] = vs[:, 3:5]
            input_[k, 0:layers, 3] = (input_[k, 0:layers, 3]- parameter_[6]) / parameter_[7]
            if(parameter_[9] != 0):
                input_[k, 0:layers, 4] = (input_[k, 0:layers, 4]- parameter_[8]) / parameter_[9]
        fv = np.expand_dims(input_, 1)
        m_train_input = fv
        pre_ = model.predict({"t1": m_train_input[0, :, :, 0:3], "t2": m_train_input[1, :, :, 0:3], 
                              "t3": m_train_input[2, :, :, 0:3], "t4": m_train_input[3, :, :, 0:3],
                              "t5": m_train_input[4, :, :, 0:3], "t6": m_train_input[5, :, :, 0:3],
                              "t7": m_train_input[6, :, :, 0:3], "t8": m_train_input[7, :, :, 0:3],
                              "cons": m_train_input[0, :, 0:8, 3:5]})
        pre_ = tf.squeeze(pre_)
        np.savetxt(pre_path + '/' + example, pre_, fmt = '%s')
print("-------model predict success-----")