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
from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy, ssrn_SS_Houston_1_nbfilter
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
    model_res4 = ssrn_SS_Houston_1_nbfilter.ResnetBuilder.build_resnet_1_1((1, img_rows, img_cols, img_channels), nb_classes)

    RMS = RMSprop(lr=0.0003)
    # Let's train the model using RMSprop
    model_res4.compile(loss='categorical_crossentropy', optimizer=RMS, metrics=['accuracy'])

    return model_res4

######### Load data ########
mat_LiDAR = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/stack_HSI_LiDAR.mat')
data_Houston = mat_LiDAR['stack']
mat_data = sio.loadmat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/datasets/Houston/Houston_gt.mat')
trainlabels = mat_data['trainlabels']
testlabels = mat_data['testlabels']
del mat_data, mat_LiDAR

######### Show the image ########
plt.imshow(data_Houston[:,:,10])
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
img_rows, img_cols = 11, 11      
patience = 200
PATCH_LENGTH = 5 #Patch_size 
TEST_SIZE = 12197
TRAIN_SIZE = 2832
TOTAL_SIZE = TRAIN_SIZE+TEST_SIZE                     
CATEGORY = 15
ALL_SIZE = data_Houston.shape[0] * data_Houston.shape[1]
img_channels = 145

######### Data normalization ########
data = data_Houston.reshape(np.prod(data_Houston.shape[:2]),np.prod(data_Houston.shape[2:]))# 3D to 2D
data = preprocessing.scale(data) #normalization
whole_data = data.reshape(data_Houston.shape[0], data_Houston.shape[1],data_Houston.shape[2])
padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)
del data,data_Houston

############ Full image mapping and model reading ############ 

best_weights_RES_path_ss4 = '/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/models/Houston/stack/Houston_stack_2-2_32_11x11_0.0003.hdf5'
model=load_model(best_weights_RES_path_ss4)

##Grouping the testing samples
n=10
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
   all_data = np.zeros((GROUP[i], 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, img_channels))
   all_assign = indexToAssignment(Group[i], whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
   for j in range(len(all_assign)):
       all_data[j] = selectNeighboringPatch(padded_data, all_assign[j][0], all_assign[j][1], PATCH_LENGTH)
   prob_image = model.predict(
           all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], all_data.shape[3], 1))
   pred_image=prob_image.argmax(axis=1)
   imageprob=imageprob+[prob_image]
   imagemap=imagemap+[pred_image]

adict={}
adict['imageprob']=imageprob
adict['imagemap']=imagemap
sio.savemat('/home/amax/Documents/GCR/RNPRF-RNDFF-RNPMF/records/Houston/map/stack_map.mat',adict)
