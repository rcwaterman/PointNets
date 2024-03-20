import numpy as np
import glob
import os
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#directory
directory = os.getcwd()

#folder path and extension
folder = os.path.join(directory, r'ExpressionDetection\UnNormData')
extension = '/*.npy'

spatial_filename = os.path.join(folder, 'AggregatedSpatialData.npy')
label_filename = os.path.join(folder, 'AggregatedLabelData.npy')

for file in glob.glob(folder + extension):
    if file == spatial_filename or file == label_filename:
        continue
    dataset = np.load(file)
    # Extract the first column and reformat it to be a 22x22 array
    #data = np.append(data, np.nan_to_num(dataset[:, :, :, 1:]), axis = 0)
    count = 0
    for sample in dataset:

        x = sample[:, :, 1][0]
        y = -sample[:, :, 2][0]
        z = sample[:, :, 3][0]

        ind = np.argsort(-y, axis=0)

        x = x[ind]
        y = y[ind]
        z = z[ind]

        x = x.reshape(478,1)
        y = y.reshape(478,1)
        z = z.reshape(478,1)

        x = np.pad(np.append(x, 
                            np.zeros((math.ceil(np.sqrt(len(x))))**2 
                            -len(x))).
                            reshape(math.ceil(np.sqrt(len(x))), 
                            math.ceil(np.sqrt(len(x)))),
                            pad_width=1,
                            mode='constant',
                            constant_values=0)
        y = np.pad(np.append(y, 
                            np.zeros((math.ceil(np.sqrt(len(y))))**2 
                            -len(y))).
                            reshape(math.ceil(np.sqrt(len(y))), 
                            math.ceil(np.sqrt(len(y)))),
                            pad_width=1,
                            mode='constant',
                            constant_values=0)
        z = np.pad(np.append(z, 
                            np.zeros((math.ceil(np.sqrt(len(z))))**2 
                            -len(z))).
                            reshape(math.ceil(np.sqrt(len(z))), 
                            math.ceil(np.sqrt(len(z)))),
                            pad_width=1,
                            mode='constant',
                            constant_values=0)
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
        z = np.nan_to_num(z)

        data_point = np.stack((x, y, z))

        if not count:
            labels = dataset[:, 0, 0, 0].reshape(dataset.shape[0], 1)
            data = np.nan_to_num(dataset[:, :, :, 1:])
        else:
            labels = np.append(labels, dataset[:, 0, 0, 0].reshape(dataset.shape[0], 1), axis=0)
            data = np.append(data, np.nan_to_num(dataset[:, :, :, 1:]), axis = 0)
    

exit()

# Existing 2D NumPy array
existing_array = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

# Value to be added in the new column
new_column_value = 10

# Add a new column with the same value to the existing array
new_array = np.append(existing_array, np.full((existing_array.shape[0], 1), new_column_value), axis=1)

# Print the updated array
print(new_array)