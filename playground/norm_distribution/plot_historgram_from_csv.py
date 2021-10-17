#!/usr/bin/env python3
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import seaborn as sns

# Input the file name
path = input("Please type in the input file input_path generated by C++ program: ")
df = pd.read_csv(path, sep = r'\s*,\s*', engine = "python")
# print(df)
# headers = ['norm value range', 'counter', 'percentage']
# print(df.columns.tolist())

df_GT = df[df['HistogramType'] == 'GT']
df_CEN = df[df['HistogramType'] == 'CENTROID']

df_GT_0NN = df_GT[df_GT['nth'] == 0]
x_GT_0NN = df_GT_0NN['value range']
y_GT_0NN = df_GT_0NN['percentage']
# print(x_GT_0NN)
# print(y_GT_0NN)

df_GT_9NN = df_GT[df_GT['nth'] == 9]
x_GT_9NN = df_GT_9NN['value range']
y_GT_9NN = df_GT_9NN['percentage']
# print(x_GT_9NN)
# print(y_GT_9NN)

df_CEN_1NN = df_CEN[df_CEN['nth'] == 1]
x_CEN_1NN = df_CEN_1NN['value range']
y_CEN_1NN = df_CEN_1NN['percentage']

df_CEN_9NN = df_CEN[df_CEN['nth'] == 9]
x_CEN_9NN = df_CEN_9NN['value range']
y_CEN_9NN = df_CEN_9NN['percentage']

df_CEN_49NN = df_CEN[df_CEN['nth'] == 49]
x_CEN_49NN = df_CEN_49NN['value range']
y_CEN_49NN = df_CEN_49NN['percentage']

df_CEN_99NN = df_CEN[df_CEN['nth'] == 99]
x_CEN_99NN = df_CEN_99NN['value range']
y_CEN_99NN = df_CEN_99NN['percentage']

plt.plot(x_GT_0NN, y_GT_0NN, label='GT 0NN', color='green', linewidth=3)
plt.plot(x_GT_9NN, y_GT_9NN, label='GT 9NN', color='blue', linewidth=3)
plt.plot(x_CEN_1NN, y_CEN_1NN, label='CENTROID 1NN', color='red', linestyle='dashed')
plt.plot(x_CEN_9NN, y_CEN_9NN, label='CENTROID 9NN', color='cyan', linestyle='dashed')
plt.plot(x_CEN_49NN, y_CEN_49NN, label='CENTROID 49NN', color='black', linestyle='dashed')
plt.plot(x_CEN_99NN, y_CEN_99NN, label='CENTROID 99NN', color='pink', linestyle='dashed')

plt.legend(fontsize = 'xx-large', markerscale = 2.0)
plt.tick_params(axis='x', which='major', labelsize=23)
plt.tick_params(axis='y', which='major', labelsize=35)

# zip joins x and y coordinates in pairs
for x,y in zip(x_GT_0NN, y_GT_0NN):
    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 fontsize=30,
                 color='green',
                 ha='center') # horizontal alignment can be left, right or center

for x,y in zip(x_GT_9NN, y_GT_9NN):
    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-10), # distance from text to points (x,y)
                 fontsize=30,
                 color='blue',
                 ha='center') # horizontal alignment can be left, right or center

for x,y in zip(x_CEN_1NN, y_CEN_1NN):
    label = "{:.2f}".format(y)
    plt.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 fontsize=30,
                 color='red',
                 ha='center') # horizontal alignment can be left, right or center

# for x,y in zip(x_CEN_9NN, y_CEN_9NN):
#     label = "{:.2f}".format(y)
#     plt.annotate(label, # this is the text
#                  (x,y), # these are the coordinates to position the label
#                  textcoords="offset points", # how to position the text
#                  xytext=(10,0), # distance from text to points (x,y)
#                  fontsize=30,
#                  color='cyan',
#                  ha='center') # horizontal alignment can be left, right or center

# for x,y in zip(x_CEN_49NN, y_CEN_49NN):
#     label = "{:.2f}".format(y)
#     plt.annotate(label, # this is the text
#                  (x,y), # these are the coordinates to position the label
#                  textcoords="offset points", # how to position the text
#                  xytext=(0,-10), # distance from text to points (x,y)
#                  fontsize=30,
#                  color='black',
#                  ha='center') # horizontal alignment can be left, right or center

# TODO: Norm Distribution or Norm Bias or N-TH Distance
plt.xlabel("L2-Distance-SQR Value Range", size = 40)
plt.ylabel("Percentage", size = 40)
# plt.title("MSSPACEV 1B GT: Distance SQR between Query and " + str(df['nth'][0]) + "-th NN ", size = 30)
# plt.title("BIGANN 1B Index: L2-Distance-SQR between Each Centroid and its Top-0 NN Centroid.", size = 30)
plt.title("BIGANN 1B Index: L2-Distance-SQR Distribution", size = 60)
plt.show()


