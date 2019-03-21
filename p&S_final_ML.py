
# coding: utf-8

# In[6]:


import keras
import segyio
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Activation, Flatten, Dropout
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy import signal
from random import sample


# import and process the raw data 
# state: data processed for different CNN models. 
#            'RF' for random forest on P-wave
#            '2D' for 2D-CNN
#            '3D' for 3D-CNN

# In[7]:


def import_process_data(state='2D'):
   
    # load 3 event sets
    path = [
            
            '/Waveform_events_5/',
            '/Waveform_events_4/',
             '/Waveform_events_B/'
           ]

    # create a map to track unique event
    count = 0
    d = {}
    for i in range(len(path)):
        for file in os.listdir(path[i]):
            if file.endswith(".sgy"):
                name = file[:-6]
                if name not in d:
                    d[name] = count
                    count += 1

    labels_P = np.zeros((count,12))
    labels_S = np.zeros((count,12))
    data = np.zeros((count, 3, 12, 2000))

    for j in range(len(path)):
        for file in os.listdir(path[j]):
            if file.endswith('.sgy'):
                fn = os.path.join(path[j], file)
                with segyio.open(fn) as f:
                    name = file[:-6]
                    if file[-5] == 'P':
                        for i in range(12):
                            labels_P[d[name], i] = 2000 - np.sum(f.trace[i])                    #P wave label
                    elif file[-5] == 'S':
                        for i in range(12):
                            labels_S[d[name], i] = 2000 - np.sum(f.trace[i])                    #S wave label
                    elif file[-5] == 'x':
                        for i in range(12):
                            data[d[name], 0, i] = f.trace[i] /np.amax(np.abs(f.trace[i]))       #normalize data                 
                    elif file[-5] == 'y':
                        for i in range(12):
                            data[d[name], 1, i] = f.trace[i] /np.amax(np.abs(f.trace[i]))       #normalize data
                    elif file[-5] == 'z':
                        for i in range(12):
                            data[d[name], 2, i] = f.trace[i] /np.amax(np.abs(f.trace[i]))       #normalize data

    data = np.transpose(data, (0, 3, 2, 1))
    
    if state == '2D':
        return data, labels_P, labels_S
    
    if state == 'RF':
        data = np.transpose(data, (0, 2, 3, 1))
        data = data.reshape((data.shape[0], data.shape[1], data.shape[2]*data.shape[3]))
        data = data.reshape((-1, data.shape[2]))
        labels_P = labels_P.flatten()
        labels_S = labels_S.flatten()
        return data, labels_P, labels_S
    
    if state == '3D':
        data = np.transpose(data, (0, 2, 3, 1))

        dt = .25e-3 # s
        labels_P = labels_P * dt
        labels_S = labels_S * dt

        fs = 1/dt
        window_size = 200 #filter window size for spectrogram 
        overlap = 195     #number of overlap for spectrogram

        f, t, Sxx = signal.spectrogram(data[0, 0, 0, :], fs, nperseg=window_size, window=('hamming'), noverlap=overlap)

        spec_data = np.zeros((data.shape[0], data.shape[1], data.shape[2], Sxx.shape[1], Sxx.shape[0]))
        spec_P = np.zeros((data.shape[0], data.shape[1]))
        spec_S = np.zeros_like(spec_P)
        for i in range(data.shape[0]): # 1665
            for j in range(data.shape[1]): # 12
                for k in range(data.shape[2]): # 3
                    f, t, Sxx = signal.spectrogram(data[i, j, k, :], fs, nperseg=window_size, window=('hamming'), noverlap=overlap)
                    spec_P[i, j] = np.argmax(t >= labels_P[i, j])
                    spec_S[i, j] = np.argmax(t >= labels_S[i, j])
                    spec_data[i, j, k, :, :] = Sxx.T / np.amax(Sxx)

        spec_data = spec_data[:,:,:,:,0:spec_data.shape[4]//4]
        spec_data = np.transpose(spec_data, (0, 1, 3, 4, 2))

        data = spec_data
        labels_P = spec_P
        labels_S = spec_S

    return data, labels_P, labels_S


# train random forest model
# n: number of traces to be blocked for sensitivity analysis
# state: 'P' for P-wave
#        'S' for S-wave

# In[8]:


def train_RF(data, labels_P, labels_S, state='P', n=1):
   
    if state == 'P':
        trainX, testX, trainY, testY = train_test_split(data, labels_P, test_size = 0.20)
    if state == 'S':
        trainX, testX, trainY, testY = train_test_split(data, labels_S, test_size = 0.20)
        
    model = RandomForestRegressor(n_estimators = 100, max_features = 80, min_samples_split = 12, min_samples_leaf = 5, bootstrap = True)
    model.fit(trainX, trainY)
    
    predictY = model.predict(testX).flatten()
    testY = testY.flatten()

    print('test statistics:')
    acc = np.abs(predictY - testY) <= 20
    acc = 1. * np.sum(acc == True) / predictY.shape[0]
    print(acc)
    acc1 = np.abs(predictY - testY) <= 40
    acc1 = 1. * np.sum(acc1 == True) / predictY.shape[0]
    print(acc1)
    acc2 = np.abs(predictY - testY) <= 80
    acc2 = 1. * np.sum(acc2 == True) / predictY.shape[0]
    print(acc2)
    mse = np.sum(((predictY-testY)**2))/predictY.shape[0]
    print(mse)
    
    print('sensitivity analysis:')
    analysis(X, y, model, n, state='RF')
    
    return model


# train 2D-CNN model
# n: number of traces to be blocked for sensitivity analysis
# state: 'P' for P-wave
#        'S' for S-wave
# 

# In[9]:


def train_2D(data, labels_P, labels_S, state='P', n=1):
    
    model = Sequential()
    inputShape = (2000, 12, 3)

    model.add(Conv2D(16, (5, 2), strides = (1, 1), padding="valid", input_shape = inputShape, data_format="channels_last"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    
    model.add(Conv2D(32, (4, 2), strides = (1, 1), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2,1), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (2, 2), strides = (1, 1), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))

    model.add(Conv2D(128, (2, 2), strides = (1, 1), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 1), strides=(2,1), padding='same'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(108))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dense(36))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dense(12))

    EPOCHS = 120
    INIT_LR = 1e-2
    BS = 128

    opt = keras.optimizers.adam(lr = INIT_LR)
    model.compile(optimizer = opt, loss = 'mse')

    if state == 'P':
        trainX, testX, trainY, testY = train_test_split(data, labels_P, test_size = 0.15)
    if state == 'S':
        trainX, testX, trainY, testY = train_test_split(data, labels_S, test_size = 0.15)

    keras.callbacks.CSVLogger(filename, separator=',', append=False)
    csv_logger = CSVLogger('training.log')

    model.fit(x=trainX, y=trainY, batch_size=BS,
                                    validation_split=0.18,
                                    callbacks=[csv_logger],
                                    epochs=EPOCHS,
                                    shuffle=True)

    print(model.summary)
    predictY = model.predict(testX).flatten()
    testY = testY.flatten()

    print('test statistics:')
    acc = np.abs(predictY - testY) <= 20
    acc = 1. * np.sum(acc == True) / predictY.shape[0]
    print(acc)
    acc1 = np.abs(predictY - testY) <= 40
    acc1 = 1. * np.sum(acc1 == True) / predictY.shape[0]
    print(acc1)
    acc2 = np.abs(predictY - testY) <= 80
    acc2 = 1. * np.sum(acc2 == True) / predictY.shape[0]
    print(acc2)
    mse = np.sum(((predictY-testY)**2))/predictY.shape[0]
    print(mse)
    
    print('sensitivity analysis:')
    analysis(X, y, model, n, state='2d')
    
    return model


# train 3D-CNN model
# n: number of traces to be blocked for sensitivity analysis
# state: 'P' for P-wave
#        'S' for S-wave
# 

# In[10]:


def train_3D(data, labels_P, labels_S, state='P', n=1):
    
    model = Sequential()
    inputShape = (data.shape[1], data.shape[2], data.shape[3], data.shape[4])
    
    model.add(Conv3D(16, (2, 4, 4), strides = (1, 1, 1), padding="valid", input_shape = inputShape, data_format="channels_last"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv3D(32, (2, 3, 3), strides = (1, 1, 1), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv3D(64, (2, 2, 2), strides = (1, 1, 1), padding="valid"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(108))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dense(36))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dense(12))

    EPOCHS = 120
    INIT_LR = 1e-2
    BS = 128
    
    opt = keras.optimizers.adam(lr = INIT_LR)
    model.compile(optimizer = opt, loss = 'mse')

    if state == 'P':
        trainX, testX, trainY, testY = train_test_split(data, labels_P, test_size = 0.15)
    if state == 'S':
        trainX, testX, trainY, testY = train_test_split(data, labels_S, test_size = 0.15)


    keras.callbacks.CSVLogger(filename, separator=',', append=False)
    csv_logger = CSVLogger('TF_12_S.log')

    model.fit(x=trainX, y=trainY, batch_size=BS,
                                    validation_split=0.18,
                                    callbacks=[csv_logger],
                                    epochs=EPOCHS,
                                    shuffle=True)

    predictY = model.predict(testX).flatten()
    testY = testY.flatten()
    
    print('test statistics:')
    acc = np.abs(predictY - testY) <= 8 /2000 * 361
    acc = 1. * np.sum(acc == True) / predictY.shape[0]
    print(acc)
    acc1 = np.abs(predictY - testY) <= 20 /2000 * 361
    acc1 = 1. * np.sum(acc1 == True) / predictY.shape[0]
    print(acc1)
    acc2 = np.abs(predictY - testY) <= 80 /2000 * 361
    acc2 = 1. * np.sum(acc2 == True) / predictY.shape[0]
    print(acc2)
    mse = np.sum(((predictY-testY)**2))/predictY.shape[0]
    print(mse)
    
    print('sensitivity analysis:')
    analysis(X, y, model, n, state='3d')
    
    return model


# sensitivity analysis
# only for 2D or 3D CNN
# n: the number of traces to be blocked
# state: '2D' for 2D-CNN
#        '3D' for 3D-CNN
# 

# In[11]:


def analysis(X, y, model, n, state='2D'):

    traces = [i for i in range(12)]
    block = sample(traces, n)
    k = 1   #coefficient for different time scale
    if state == '2D':
        for i in range(12):
            if i in block:
                X[:,:,i,:] = 0
    if state == '3D':
        k = 361/2000
        for i in range(12):
            if i in block:
                X[i,:,:,:,:] = 0
    
    predictY = model.predict(testX).flatten()
    testY = y.flatten()

    acc = np.abs(predictY - testY) <= 20 * k
    acc = np.sum(acc == True) / predictY.shape[0] *1.0
    print(acc)
    acc1 = np.abs(predictY - testY) <= 40 * k
    acc1 = 1. * np.sum(acc1 == True) / predictY.shape[0]
    print(acc1)
    acc2 = np.abs(predictY - testY) <= 80 * k
    acc2 = 1. * np.sum(acc2 == True) / predictY.shape[0]
    print(acc2)
    mse = np.sum(((predictY-testY)**2))/predictY.shape[0]
    print(mse)


# load and process data, train and test model, do sensitivity analysis
# state: 'RF' for random forest
#        '2D' for 2D-CNN
#        '3D' for 3D-CNN
# wave: 'P' for P-wave
#       'S' for S-wave
# n: the number of traces to be blocked
# 

# In[12]:


def main(state, wave, n):
    state='2D'
    wave='P'
    
    data, labels_P, labels_S = import_process_data(state)
    if state == 'RF':
        model = train_RF(data, labels_P, labels_S, wave, n)
    if state == '2D':
        model = train_2D(data, labels_P, labels_S, wave, n)
    if state == '3D':
        model = train_3D(data, labels_P, labels_S, wave, n)
    
    main(state, wave, n)

