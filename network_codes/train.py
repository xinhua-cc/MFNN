import tensorflow as tf
import numpy as np
import os
import argparse
from network import model_simulate
from txt2npy import read_files, read_sensi, read_files_one

parser = argparse.ArgumentParser()

parser.add_argument("--epoch", default=10000000)
parser.add_argument("--batch_size", default=128)
parser.add_argument("--ratio", default=0.2)
parser.add_argument("--traininput_dir", default="/data/field_work/data/train_input_3/")
parser.add_argument("--trainlabel_dir", default="/data/field_work/data/train_label_3/")
parser.add_argument("--sensi_dir", default="/data/field_work/data/constraintdata_3/sensi/")
parser.add_argument("--den_dir", default="/data/field_work/data/constraintdata_3/den/")
parser.add_argument("--vp_dir", default="/data/field_work/data/constraintdata_3/vp/")

parser.add_argument("--cpt_dir", default=("/data/field_work/multiple-Withsensi_2/data/checkpoint_125_4" ))
parser.add_argument("--callbacks_dir", default=("/data/field_work/multiple-Withsensi_2/data/callbacks_125_4"))

opt = parser.parse_args()
LayerNum = 8
fvNum = 180
AreaName = ["1", "2", "3", "4", "5", "6"]
AreaNum = 6
Number = 40000

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
# tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
# strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2"])
m_batch_size = opt.batch_size * strategy.num_replicas_in_sync

def scheduler(epoch):
    if epoch <= 10000:
        return 0.001
    else:
        return 0.001

def cc_loss_2(y_true, y_cal):
    cc_loss = 100000 * 0.5 * tf.reduce_mean(tf.square((y_true - y_cal) / y_true) * layer_sensi)
    return cc_loss

Exists1 = os.path.exists(opt.cpt_dir)
if not Exists1:
    os.makedirs(opt.cpt_dir)

Exists2 = os.path.exists(opt.callbacks_dir)
if not Exists2:
    os.makedirs(opt.callbacks_dir)

sensi = read_sensi(opt.sensi_dir, AreaNum, AreaName, Number)
pre_sensi = np.mean(sensi, axis = 1)
layer_sensi = np.mean(pre_sensi, axis = 0)


print("load all_data begin !!!!!!!!!")
m_train_input, m_train_label = read_files(opt.traininput_dir, opt.trainlabel_dir, opt.sensi_dir, opt.den_dir, opt.vp_dir,
                                          AreaNum, AreaName, Number, LayerNum, fvNum)
print("load all_data success !!!!!!!!!")

checkpoint_prefix = os.path.join(opt.cpt_dir, "weights.{epoch:02d}-{loss:.5f}.h5")

'''
if os.path.exists(opt.opt_dir + 'variables' + 'variables.index'):
    print('------------------load weights-------------------')
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
'''

#--------GPU并行-------------#
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(scheduler),
    tf.keras.callbacks.TensorBoard(opt.callbacks_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, 
                                       monitor='loss', # applied for quit the training process
                                       save_wights_only=True, save_best_only=False, save_freq = 10 * m_batch_size,
                                       verbose=1)
    ]

with strategy.scope():
    m_model = model_simulate()

    m_model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss = cc_loss_2)

    m_model.fit({"t1": m_train_input[0, :, :, 0: 3], "t2": m_train_input[1, :, :, 0: 3], "t3": m_train_input[2, :, :, 0: 3], 
                 "t4": m_train_input[3, :, :, 0: 3], "t5": m_train_input[4, :, :, 0: 3], "t6": m_train_input[5, :, :, 0: 3],
                 "t7": m_train_input[6, :, :, 0: 3], "t8": m_train_input[7, :, :, 0: 3], "cons": m_train_input[0, :, 0:8, 3: 5],},
                m_train_label,
            validation_split=opt.ratio, shuffle=True, batch_size = opt.batch_size,
            epochs=opt.epoch, callbacks=callbacks, validation_freq=10)
