#Write by Chiru Ge, contact: gechiru@126.com

# stack (HSI+HSI_EPLBP+LiDAR_EPLBP) ++ Resnet
# 3 deep feature fusion

### use CPU only
#import os 
#import sys 
#os.environ["CUDA_DEVICE_ORDER"]="PCA_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

## use GPU
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='0'
config=tf.ConfigProto()
config.gpu_options.allow_growth= True
sess=tf.Session(config=config)

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.utils.np_utils import to_categorical
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
from keras.regularizers import l2
import time
import collections
from sklearn import metrics, preprocessing
from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy, ssrn_SS_Houston_3FF_F1
from keras.utils.vis_utils import plot_model
import h5py
from keras.models import load_model

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length 
        assign_1 = value % Col + pad_length 
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
#        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices

def sampling1(trainlabels, testlabels, verification_set):
    labels_loc = {}
    train_indices=[]
    test_indices=[]
    verification_indices=[]
    m={}
    m=np.max(trainlabels[:])
    for i in range(m):
        indices = [j for j, x in enumerate(trainlabels.ravel().tolist()) if x == i + 1] 
        labels_loc[i] = indices
        train_indices += labels_loc[i]
    for i in range(m):
        indices = [j for j, x in enumerate(testlabels.ravel().tolist()) if x == i + 1] 
        labels_loc[i] = indices
        test_indices += labels_loc[i]
    for i in range(m):
        indices = [j for j, x in enumerate(verification_set.ravel().tolist()) if x == i + 1] 
        labels_loc[i] = indices
        verification_indices += labels_loc[i]    
    return train_indices, test_indices, verification_indices

def res4_model_ss():
    model_res4 = ssrn_SS_Houston_3FF_F1.ResnetBuilder.build_resnet_fusion_2_2((1, img_rows, img_cols, img_channels_HSI),(1, img_rows, img_cols, img_channels_HSIEPLBP),(1, img_rows, img_cols, img_channels_LiDAREPLBP), nb_classes)

    RMS = RMSprop(lr=0.0003)
    # Let's train the model using RMSprop
    model_res4.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])

    return model_res4

######### save the best validated model ##########
    # 3FF is 3 features deep fusion
best_weights_RES_path_ss4 = ('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/models/Houston/3DFF/Houston_3FF_7-7_2-2_24_0.0003_2.hdf5')
 
######### Load data HSI ########
mat_LiDAR = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/HSI.mat')
data_Houston_HSI = mat_LiDAR['HSI']
mat_data = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/Houston_labels.mat')
trainlabels1 = mat_data['trainlabels']
testlabels1 = mat_data['testlabels']
verification_set1 = mat_data['verification_set']
del mat_data, mat_LiDAR

######### Load data HSI_EPLBP ########
file=h5py.File('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/HSI_EPLBP.mat','r')
file.keys()
data = file['HSI_EPLBP'][:]
data_Houston_HSIEPLBP=data.transpose(2,1,0);
file.close()
mat_data = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/Houston_labels.mat')
trainlabels = mat_data['trainlabels']
testlabels = mat_data['testlabels']
verification_set = mat_data['verification_set']
del mat_data, data

######### Load data LiDAR_EPLBP ########
mat_LiDAR = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/LiDAR_DSM_EPLBP.mat')
data_Houston_LiDAREPLBP = mat_LiDAR['LiDAR_DSM_EPLBP']
mat_data = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/Houston_labels.mat')
trainlabels = mat_data['trainlabels']
testlabels = mat_data['testlabels']
verification_set = mat_data['verification_set']
del mat_data, mat_LiDAR

######### Show the image ########
plt.imshow(data_Houston_HSI[:,:,10])
plt.show()
plt.imshow(trainlabels[:,], cmap="jet")
plt.show()
plt.imshow(testlabels[:,],cmap="jet")
plt.show()
np.max(testlabels[:])#find the max value of the groundtruth
np.min(testlabels[:])
######### Training parameter setting ##########
batch_size = 16 #sample number of each batch
nb_classes = 15 #class number
nb_epoch = 200  #epoch
img_rows, img_cols = 7, 7        
patience = 200
PATCH_LENGTH = 3 #Patch_size 
TEST_SIZE = 12197
TRAIN_SIZE = 2832
VERIFICATION_SIZE = 1500
TOTAL_SIZE = TRAIN_SIZE+TEST_SIZE                     
img_channels_HSI = 144
img_channels_HSIEPLBP = 815
img_channels_LiDAREPLBP = 134
CATEGORY = 15

######### Data normalization ########
data = data_Houston_HSI.reshape(np.prod(data_Houston_HSI.shape[:2]),np.prod(data_Houston_HSI.shape[2:]))# 3D to 2D
data = preprocessing.scale(data) #normalization
whole_data_HSI = data.reshape(data_Houston_HSI.shape[0], data_Houston_HSI.shape[1],data_Houston_HSI.shape[2])
padded_data_HSI = zeroPadding.zeroPadding_3D(whole_data_HSI, PATCH_LENGTH)
del data,data_Houston_HSI

data = data_Houston_HSIEPLBP.reshape(np.prod(data_Houston_HSIEPLBP.shape[:2]),np.prod(data_Houston_HSIEPLBP.shape[2:]))# 3维矩阵转换为2维矩阵
data = preprocessing.scale(data) #normalization
whole_data_HSIEPLBP = data.reshape(data_Houston_HSIEPLBP.shape[0], data_Houston_HSIEPLBP.shape[1],data_Houston_HSIEPLBP.shape[2])
padded_data_HSIEPLBP = zeroPadding.zeroPadding_3D(whole_data_HSIEPLBP, PATCH_LENGTH)
del data,data_Houston_HSIEPLBP

data = data_Houston_LiDAREPLBP.reshape(np.prod(data_Houston_LiDAREPLBP.shape[:2]),np.prod(data_Houston_LiDAREPLBP.shape[2:]))# 3维矩阵转换为2维矩阵
data = preprocessing.scale(data) #normalization
whole_data_LiDAREPLBP = data.reshape(data_Houston_LiDAREPLBP.shape[0], data_Houston_LiDAREPLBP.shape[1],data_Houston_LiDAREPLBP.shape[2])
padded_data_LiDAREPLBP = zeroPadding.zeroPadding_3D(whole_data_LiDAREPLBP, PATCH_LENGTH)
del data,data_Houston_LiDAREPLBP

######### Make the labels of the training samples and testing samples ########
## 3D to 2D       
trainl = trainlabels.reshape(np.prod(trainlabels.shape[:2]),)
testl = testlabels.reshape(np.prod(testlabels.shape[:2]),)
verificationl = verification_set.reshape(np.prod(verification_set.shape[:2]),)

## Find the index of training samples and testing samples
train_indices, test_indices, verification_indices = sampling1(trainlabels, testlabels, verification_set)

## Training sample label
y_train = trainl[train_indices] - 1
y_train = to_categorical(np.asarray(y_train))#to_categorical convert the category number to binary

## Testing sample label
y_test = testl[test_indices] - 1
y_test = to_categorical(np.asarray(y_test))

## Validation sample label
y_verification = verificationl[verification_indices] - 1
y_verification = to_categorical(np.asarray(y_verification))
   
## HSI training samples, testing samples, validation samples
train_data_HSI = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_HSI))
test_data_HSI = np.zeros((TEST_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_HSI))
verification_data_HSI = np.zeros((VERIFICATION_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_HSI))

train_assign_HSI = indexToAssignment(train_indices, whole_data_HSI.shape[0], whole_data_HSI.shape[1], PATCH_LENGTH)
for i in range(len(train_assign_HSI)):
    train_data_HSI[i] = selectNeighboringPatch(padded_data_HSI, train_assign_HSI[i][0], train_assign_HSI[i][1], PATCH_LENGTH) 

test_assign_HSI = indexToAssignment(test_indices, whole_data_HSI.shape[0], whole_data_HSI.shape[1], PATCH_LENGTH)
for i in range(len(test_assign_HSI)):
    test_data_HSI[i] = selectNeighboringPatch(padded_data_HSI, test_assign_HSI[i][0], test_assign_HSI[i][1], PATCH_LENGTH) 

verification_assign_HSI = indexToAssignment(verification_indices, whole_data_HSI.shape[0], whole_data_HSI.shape[1], PATCH_LENGTH)
for i in range(len(verification_assign_HSI)):
    verification_data_HSI[i] = selectNeighboringPatch(padded_data_HSI, verification_assign_HSI[i][0], verification_assign_HSI[i][1], PATCH_LENGTH) 

x_train_HSI = train_data_HSI.reshape(train_data_HSI.shape[0], train_data_HSI.shape[1], train_data_HSI.shape[2], img_channels_HSI)
x_test_HSI = test_data_HSI.reshape(test_data_HSI.shape[0], test_data_HSI.shape[1], test_data_HSI.shape[2], img_channels_HSI)
x_verification_HSI = verification_data_HSI.reshape(verification_data_HSI.shape[0], verification_data_HSI.shape[1], verification_data_HSI.shape[2], img_channels_HSI)
 
## HSIEPLBP training samples, testing samples, validation samples
train_data_HSIEPLBP = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_HSIEPLBP))
test_data_HSIEPLBP = np.zeros((TEST_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_HSIEPLBP))
verification_data_HSIEPLBP = np.zeros((VERIFICATION_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_HSIEPLBP))

train_assign_HSIEPLBP = indexToAssignment(train_indices, whole_data_HSIEPLBP.shape[0], whole_data_HSIEPLBP.shape[1], PATCH_LENGTH)
for i in range(len(train_assign_HSIEPLBP)):
    train_data_HSIEPLBP[i] = selectNeighboringPatch(padded_data_HSIEPLBP, train_assign_HSIEPLBP[i][0], train_assign_HSIEPLBP[i][1], PATCH_LENGTH) 

test_assign_HSIEPLBP = indexToAssignment(test_indices, whole_data_HSIEPLBP.shape[0], whole_data_HSIEPLBP.shape[1], PATCH_LENGTH)
for i in range(len(test_assign_HSIEPLBP)):
    test_data_HSIEPLBP[i] = selectNeighboringPatch(padded_data_HSIEPLBP, test_assign_HSIEPLBP[i][0], test_assign_HSIEPLBP[i][1], PATCH_LENGTH) 

verification_assign_HSIEPLBP = indexToAssignment(verification_indices, whole_data_HSIEPLBP.shape[0], whole_data_HSIEPLBP.shape[1], PATCH_LENGTH)
for i in range(len(verification_assign_HSIEPLBP)):
    verification_data_HSIEPLBP[i] = selectNeighboringPatch(padded_data_HSIEPLBP, verification_assign_HSIEPLBP[i][0], verification_assign_HSIEPLBP[i][1], PATCH_LENGTH) 

x_train_HSIEPLBP = train_data_HSIEPLBP.reshape(train_data_HSIEPLBP.shape[0], train_data_HSIEPLBP.shape[1], train_data_HSIEPLBP.shape[2], img_channels_HSIEPLBP)
x_test_HSIEPLBP = test_data_HSIEPLBP.reshape(test_data_HSIEPLBP.shape[0], test_data_HSIEPLBP.shape[1], test_data_HSIEPLBP.shape[2], img_channels_HSIEPLBP)
x_verification_HSIEPLBP = verification_data_HSIEPLBP.reshape(verification_data_HSIEPLBP.shape[0], verification_data_HSIEPLBP.shape[1], verification_data_HSIEPLBP.shape[2], img_channels_HSIEPLBP)

## LiDAREPLBP training samples, testing samples, validation samples
train_data_LiDAREPLBP = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_LiDAREPLBP))
test_data_LiDAREPLBP = np.zeros((TEST_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_LiDAREPLBP))
verification_data_LiDAREPLBP = np.zeros((VERIFICATION_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_LiDAREPLBP))

train_assign_LiDAREPLBP = indexToAssignment(train_indices, whole_data_LiDAREPLBP.shape[0], whole_data_LiDAREPLBP.shape[1], PATCH_LENGTH)
for i in range(len(train_assign_LiDAREPLBP)):
    train_data_LiDAREPLBP[i] = selectNeighboringPatch(padded_data_LiDAREPLBP, train_assign_LiDAREPLBP[i][0], train_assign_LiDAREPLBP[i][1], PATCH_LENGTH) 

test_assign_LiDAREPLBP = indexToAssignment(test_indices, whole_data_LiDAREPLBP.shape[0], whole_data_LiDAREPLBP.shape[1], PATCH_LENGTH)
for i in range(len(test_assign_LiDAREPLBP)):
    test_data_LiDAREPLBP[i] = selectNeighboringPatch(padded_data_LiDAREPLBP, test_assign_LiDAREPLBP[i][0], test_assign_LiDAREPLBP[i][1], PATCH_LENGTH) 

verification_assign_LiDAREPLBP = indexToAssignment(verification_indices, whole_data_LiDAREPLBP.shape[0], whole_data_LiDAREPLBP.shape[1], PATCH_LENGTH)
for i in range(len(verification_assign_LiDAREPLBP)):
    verification_data_LiDAREPLBP[i] = selectNeighboringPatch(padded_data_LiDAREPLBP, verification_assign_LiDAREPLBP[i][0], verification_assign_LiDAREPLBP[i][1], PATCH_LENGTH) 

x_train_LiDAREPLBP = train_data_LiDAREPLBP.reshape(train_data_LiDAREPLBP.shape[0], train_data_LiDAREPLBP.shape[1], train_data_LiDAREPLBP.shape[2], img_channels_LiDAREPLBP)
x_test_LiDAREPLBP = test_data_LiDAREPLBP.reshape(test_data_LiDAREPLBP.shape[0], test_data_LiDAREPLBP.shape[1], test_data_LiDAREPLBP.shape[2], img_channels_LiDAREPLBP)
x_verification_LiDAREPLBP = verification_data_LiDAREPLBP.reshape(verification_data_LiDAREPLBP.shape[0], verification_data_LiDAREPLBP.shape[1], verification_data_LiDAREPLBP.shape[2], img_channels_LiDAREPLBP)

############ Evaluation index ############ 
KAPPA_RES_SS4 = []
OA_RES_SS4 = []
AA_RES_SS4 = []
TRAINING_TIME_RES_SS4 = []
TESTING_TIME_RES_SS4 = []
#ELEMENT_ACC_RES_SS4 = np.zeros((ITER, CATEGORY))
      
############ Model training and result evaluation ############
model_res4_SS_BN = res4_model_ss()
plot_model(model_res4_SS_BN, to_file='/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/model_show/test.png', show_shapes=True, show_layer_names=True) # imageshow the Residual Network with BN

earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_acc', patience=patience, verbose=1, mode='max')
saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_RES_path_ss4, monitor='val_acc', verbose=1,
                                            save_best_only=True,
                                            mode='max')

tic6 = time.clock()
#print(x_train.shape, x_test.shape)
history_res4_SS_BN = model_res4_SS_BN.fit(x=[x_train_HSI.reshape(x_train_HSI.shape[0], x_train_HSI.shape[1], x_train_HSI.shape[2], x_train_HSI.shape[3], 1),
                                          x_train_HSIEPLBP.reshape(x_train_HSIEPLBP.shape[0], x_train_HSIEPLBP.shape[1], x_train_HSIEPLBP.shape[2], x_train_HSIEPLBP.shape[3], 1), 
                                          x_train_LiDAREPLBP.reshape(x_train_LiDAREPLBP.shape[0], x_train_LiDAREPLBP.shape[1], x_train_LiDAREPLBP.shape[2], x_train_LiDAREPLBP.shape[3], 1)], 
                                          y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, callbacks=[earlyStopping6, saveBestModel6], 
                                          validation_data=([x_verification_HSI.reshape(x_verification_HSI.shape[0], x_verification_HSI.shape[1], x_verification_HSI.shape[2], x_verification_HSI.shape[3], 1),
                                                            x_verification_HSIEPLBP.reshape(x_verification_HSIEPLBP.shape[0], x_verification_HSIEPLBP.shape[1], x_verification_HSIEPLBP.shape[2], x_verification_HSIEPLBP.shape[3], 1),
                                                            x_verification_LiDAREPLBP.reshape(x_verification_LiDAREPLBP.shape[0], x_verification_LiDAREPLBP.shape[1], x_verification_LiDAREPLBP.shape[2], x_verification_LiDAREPLBP.shape[3], 1)], y_verification), 
                                          shuffle=True)
toc6 = time.clock()

# load  best  model
model=load_model(best_weights_RES_path_ss4)
tic7 = time.clock()
loss_and_metrics = model.evaluate(x=[x_test_HSI.reshape(x_test_HSI.shape[0], x_test_HSI.shape[1], x_test_HSI.shape[2], x_test_HSI.shape[3], 1),
                                          x_test_HSIEPLBP.reshape(x_test_HSIEPLBP.shape[0], x_test_HSIEPLBP.shape[1], x_test_HSIEPLBP.shape[2], x_test_HSIEPLBP.shape[3], 1), 
                                          x_test_LiDAREPLBP.reshape(x_test_LiDAREPLBP.shape[0], x_test_LiDAREPLBP.shape[1], x_test_LiDAREPLBP.shape[2], x_test_LiDAREPLBP.shape[3], 1)], 
                                          y=y_test,
                                          batch_size=batch_size)
toc7 = time.clock()

print('3D RES_SS4 without BN Training Time: ', toc6 - tic6)

print('3D RES_SS4 without BN Test time:', toc7 - tic7)
print('3D RES_SS4 without BN Test score:', loss_and_metrics[0])
print('3D RES_SS4 without BN Test accuracy:', loss_and_metrics[1])

Probability = model.predict([x_test_HSI.reshape(x_test_HSI.shape[0], x_test_HSI.shape[1], x_test_HSI.shape[2], x_test_HSI.shape[3], 1),
                                                            x_test_HSIEPLBP.reshape(x_test_HSIEPLBP.shape[0], x_test_HSIEPLBP.shape[1], x_test_HSIEPLBP.shape[2], x_test_HSIEPLBP.shape[3], 1),
                                                            x_test_LiDAREPLBP.reshape(x_test_LiDAREPLBP.shape[0], x_test_LiDAREPLBP.shape[1], x_test_LiDAREPLBP.shape[2], x_test_LiDAREPLBP.shape[3], 1)])
pred_test_res4=Probability.argmax(axis=1)

collections.Counter(pred_test_res4)
gt_test = testl[test_indices] - 1
overall_acc_res4 = metrics.accuracy_score(pred_test_res4, gt_test)
confusion_matrix_res4 = metrics.confusion_matrix(pred_test_res4, gt_test)
each_acc_res4, average_acc_res4 = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_res4)
kappa = metrics.cohen_kappa_score(pred_test_res4, gt_test)
testing_time=toc7 - tic7

training_time=toc6 - tic6

# Save the data to "**.mat" format
adict={}
adict['OA']=overall_acc_res4
adict['AA']=average_acc_res4
adict['testing_time']=testing_time
adict['kappa']=kappa
adict['each_acc']=each_acc_res4
adict['confusion_matrix']=confusion_matrix_res4

adict['Probability_DF']=Probability
adict['maxPro_DF']=pred_test_res4
adict['testlabel']=y_test

adict['training_time']=training_time
sio.savemat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/records/Houston/3DFF/Houston_3FF_7-7_2-2_24_0.0003_2.mat',adict)


