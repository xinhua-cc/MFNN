import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras import Model, Input
from tensorflow.python.keras.layers import ReLU


def model_child(m_input, key):
    
    x = Conv1D(256, kernel_size=3, padding='same')(m_input)
    x = ReLU()(x)        
    y = Flatten()(x)        
    # x = Dense(256, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # y = Dense(32, activation='relu')(x)
    # x = Dense(16, activation='relu')(x)
    # x = Dense(8)(x)
    # y = Dense(1, name=key)(x) 
    return y

def model_simulate():
    
    m_input1 = Input(shape=(180, 3), name = "t1")
    m_input2 = Input(shape=(180, 3), name = "t2")
    m_input3 = Input(shape=(180, 3), name = "t3")
    m_input4 = Input(shape=(180, 3), name = "t4")
    m_input5 = Input(shape=(180, 3), name = "t5")
    m_input6 = Input(shape=(180, 3), name = "t6")
    m_input7 = Input(shape=(180, 3), name = "t7")
    m_input8 = Input(shape=(180, 3), name = "t8")
    m_fv = Input(shape=(8, 2), name = 'cons')
    fv = Flatten()(m_fv)

    y1 = model_child(m_input1, "f1")
    y2 = model_child(m_input2, "f2")
    y3 = model_child(m_input3, "f3")
    y4 = model_child(m_input4, "f4")
    y5 = model_child(m_input5, "f5")
    y6 = model_child(m_input6, "f6")
    y7 = model_child(m_input7, "f7")
    y8 = model_child(m_input8, "f8")
    
    y = Concatenate()([y1, y2, y3, y4, y5, y6, y7, y8])
    # y = BatchNormalization(y)
    y = Dense(256, activation = 'relu')(y)
    y = Dense(512, activation='relu')(y)
    y = Dense(256, activation='relu')(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(64, activation='relu')(y)
    y = Dense(32, activation='relu')(y)
    # y = Dense(16, activation='relu')(y)
    y = Dense(24)(y)
    y = Concatenate()([y, fv,])
    # y = BatchNormalization(y)
    y = Dense(64)(y)
    y = Dense(24)(y)
    y = Dense(8)(y)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    m_model = tf.keras.Model(inputs=[m_input1, m_input2, m_input3, m_input4,
                                     m_input5, m_input6, m_input7, m_input8, m_fv],
                             outputs=y)
    return m_model

