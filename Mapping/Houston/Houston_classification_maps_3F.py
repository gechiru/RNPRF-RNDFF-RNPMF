#Write by Chiru Ge, contact: gechiru@126.com

# -*- coding: utf-8 -*-
## use GPU
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='0'
config=tf.ConfigProto()
config.gpu_options.allow_growth= True
sess=tf.Session(config=config)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
from sklearn import metrics, preprocessing
from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy, ssrn_SS_Houston_3FF_F1
import h5py
from keras.models import load_model
from keras.utils.vis_utils import plot_model

def sampling1(trainlabels, testlabels):
    labels_loc = {}
    train_indices=[]
    test_indices=[]
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
    return train_indices, test_indices

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

def classification_map(map, groundTruth, dpi, savePath):

    fig = plt.figure(frameon=False)
    fig.set_size_inches(groundTruth.shape[1]*2.0/dpi, groundTruth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map, aspect='equal')
    fig.savefig(savePath, dpi = dpi)

    return 0

def res4_model_ss():
    model_res4 = ssrn_SS_Houston_3FF_F1.ResnetBuilder.build_resnet_2_2((1, img_rows, img_cols, img_channels), nb_classes)

    RMS = RMSprop(lr=0.0003)
    # Let's train the model using RMSprop
    model_res4.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])

    return model_res4

######### Load data HSI ########
mat_LiDAR = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/HSI.mat')
data_Houston_HSI = mat_LiDAR['HSI']
mat_data = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/Houston_gt.mat')
trainlabels = mat_data['trainlabels']
testlabels = mat_data['testlabels']
del mat_data, mat_LiDAR

######### Load data HSI_EPLBP ########
file=h5py.File('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/HSI_EPLBP.mat','r')
file.keys()
data = file['HSI_EPLBP'][:]
data_Houston_HSIEPLBP=data.transpose(2,1,0);
file.close()
mat_data = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/Houston_gt.mat')
trainlabels = mat_data['trainlabels']
testlabels = mat_data['testlabels']
del mat_data, data

######### Load data LiDAR_EPLBP ########
mat_LiDAR = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/LiDAR_DSM_EPLBP.mat')
data_Houston_LiDAREPLBP = mat_LiDAR['LiDAR_DSM_EPLBP']
mat_data = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/Houston_gt.mat')
trainlabels = mat_data['trainlabels']
testlabels = mat_data['testlabels']
del mat_data, mat_LiDAR

######### Training parameter setting ##########
batch_size = 16 #sample number of each batch
nb_classes = 15 #class number
nb_epoch = 200  #epoch
img_rows, img_cols = 7, 7   
patience = 200
PATCH_LENGTH = 3 #Patch_size 
TEST_SIZE = 12197
TRAIN_SIZE = 2832
TOTAL_SIZE = TRAIN_SIZE+TEST_SIZE                     
img_channels_HSI = 144
img_channels_HSIEPLBP = 815
img_channels_LiDAREPLBP = 134
CATEGORY = 15
ALL_SIZE = data_Houston_HSI.shape[0] * data_Houston_HSI.shape[1]
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

############ Full image mapping and model reading ############ 

best_weights_RES_path_ss4 = ('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/models/Houston/3DFF/Houston_3FF_7-7_2-2_24_0.0003.hdf5')
model=load_model(best_weights_RES_path_ss4)

##Grouping the testing samples
n=60
group=ALL_SIZE//n
group_last=ALL_SIZE%n
Group=[]

for i in range(n+1):
   if i==0:
       Group.append(range(group))
   elif i!= n and i > 0:
       Group.append(range(group*i,group*(i+1)))
   elif i==n:
       Group.append(range(group*i,group*i+group_last))     
GROUP=[]
for i in range(n+1):
   if i!= n:
       GROUP.append(group)
   elif i==n:
       GROUP.append(group_last)
       
##Predict each set of test samples. imagemap is the final map.
imagemap=[]
imageprob=[]
for i in range(len(Group)):
   print(i)
   all_data_HSI = np.zeros((GROUP[i], 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_HSI))
   all_data_HSIEPLBP = np.zeros((GROUP[i], 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_HSIEPLBP))
   all_data_LiDAREPLBP = np.zeros((GROUP[i], 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels_LiDAREPLBP))
   all_assign = indexToAssignment(Group[i], whole_data_HSI.shape[0], whole_data_HSI.shape[1], PATCH_LENGTH)
   for j in range(len(all_assign)):
       all_data_HSI[j] = selectNeighboringPatch(padded_data_HSI, all_assign[j][0], all_assign[j][1], PATCH_LENGTH)
       all_data_HSIEPLBP[j] = selectNeighboringPatch(padded_data_HSIEPLBP, all_assign[j][0], all_assign[j][1], PATCH_LENGTH)
       all_data_LiDAREPLBP[j] = selectNeighboringPatch(padded_data_LiDAREPLBP, all_assign[j][0], all_assign[j][1], PATCH_LENGTH)      
   prob_image = model.predict(
           [all_data_HSI.reshape(all_data_HSI.shape[0], all_data_HSI.shape[1], all_data_HSI.shape[2], all_data_HSI.shape[3], 1),
            all_data_HSIEPLBP.reshape(all_data_HSIEPLBP.shape[0], all_data_HSIEPLBP.shape[1], all_data_HSIEPLBP.shape[2], all_data_HSIEPLBP.shape[3], 1),
            all_data_LiDAREPLBP.reshape(all_data_LiDAREPLBP.shape[0], all_data_LiDAREPLBP.shape[1], all_data_LiDAREPLBP.shape[2], all_data_LiDAREPLBP.shape[3], 1)])
   pred_image=prob_image.argmax(axis=1)
   imageprob=imageprob+[prob_image]
   imagemap=imagemap+[pred_image]

adict={}
adict['imageprob']=imageprob
adict['imagemap']=imagemap
sio.savemat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/records/Houston/map/3FF_map.mat',adict)
