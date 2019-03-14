# -*- coding: utf-8 -*-
"""
LandSurfaceClustering.py

Landon Halloran 
07.03.2019 
www.ljsh.ca

Land use clustering using multi-band remote sensing data.

This is a rough first version. Modifications will need to be made in order to properly 
treat other datasets.

Demo data is Sentinel-2 data (bands 2, 3, 4, 5, 6, 7, 8, 11 & 12) at 10m resolution 
(some bands are upsampled) in PNG format, exported from L1C_T32TLT_A007284_20180729T103019.
"""

# import these modules:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
import glob
import seaborn as sns; sns.set(style="ticks", color_codes=True)

########## USER MUST DEFINE THESE ###########
image_folder_name = 'example_png' # subfolder where images are located
image_format = 'png' # format of image files (the exact suffix of the filenames)
Nsamples = 15000 #  number of random samples used to "train" k-means here (for faster execution)
NUMBER_OF_CLUSTERS = 8 # the number of independent clusters for k-means
colour_map = 'terrain' # cmap, see matplotlib.org/examples/color/colormaps_reference.html
#############################################

# import images to dictionary:
images = dict();
for image_path in glob.glob(image_folder_name+'/*.'+image_format):
    print('reading ',image_path)
    temp = imageio.imread(image_path)
    temp = temp[:,:,0].squeeze()
    images[image_path[18:21]] = temp
print('images have ', np.size(temp),' pixels each')


# make a 3D numpy array of data...
imagecube = np.zeros([images['B2_'].shape[0],images['B2_'].shape[1],9])
imagecube[:,:,0] = images['B2_'] # 
imagecube[:,:,1] = images['B3_'] # 
imagecube[:,:,2] = images['B4_'] #
imagecube[:,:,3] = images['B5_'] #
imagecube[:,:,4] = images['B6_'] #
imagecube[:,:,5] = images['B7_'] # 
imagecube[:,:,6] = images['B8_'] # 
imagecube[:,:,7] = images['B11'] # 
imagecube[:,:,8] = images['B12'] # 
imagecube=imagecube/256 #  scaling to between 0 and 1

# display an RGB or false colour image
thefigsize = (10,8)# set figure size
#plt.figure(figsize=thefigsize)
#plt.imshow(imagecube[:,:,0:3])


# sample random subset of images
imagesamples = []
for i in range(Nsamples):
    xr=np.random.randint(0,imagecube.shape[1]-1)
    yr=np.random.randint(0,imagecube.shape[0]-1)
    imagesamples.append(imagecube[yr,xr,:])
# convert to pandas dataframe
imagessamplesDF=pd.DataFrame(imagesamples,columns = ['B2_','B3_','B4_','B5_','B6_','B7_','B8_','B11','B12'])


# make pairs plot (each band vs. each band)
seaborn_params_p = {'alpha': 0.15, 's': 20, 'edgecolor': 'k'}
#pp1=sns.pairplot(imagessamplesDF, plot_kws = seaborn_params_p)#, hist_kws=seaborn_params_h)

# fit kmeans to to samples:
from sklearn.cluster import KMeans

KMmodel = KMeans(n_clusters=NUMBER_OF_CLUSTERS) 
KMmodel.fit(imagessamplesDF)
KM_train = list(KMmodel.predict(imagessamplesDF)) 
i=0
for k in KM_train:
    KM_train[i] = str(k) 
    i=i+1
imagessamplesDF2=imagessamplesDF
imagessamplesDF2['group'] = KM_train
# pair plots with clusters coloured:
pp2=sns.pairplot(imagessamplesDF,vars=['B2_','B3_','B4_','B5_','B6_','B7_','B8_','B11','B12'], hue='group',plot_kws = seaborn_params_p)

# 
imageclustered=np.empty((imagecube.shape[0],imagecube.shape[1]))
i=0
for row in imagecube:
    temp = KMmodel.predict(row) 
    imageclustered[i,:]=temp
    i=i+1
# plot the map of the clustered data
plt.figure(figsize=thefigsize)
plt.imshow(imageclustered, cmap=colour_map) 
