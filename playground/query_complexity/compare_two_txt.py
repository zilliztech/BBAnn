#!/usr/bin/env python3
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import seaborn as sns

# Input the file name
df_query_10Centriod = pd.read_csv("/home/jigao/Workspace/BigAnn/playground/query_complexity/query_to_10NN_centriod_distance.txt", sep = r'\s*:=\s*', engine = "python")
df_query_20Centriod = pd.read_csv("/home/jigao/Workspace/BigAnn/playground/query_complexity/query_to_20NN_centriod_distance.txt", sep = r'\s*:=\s*', engine = "python")
df_query_30Centriod = pd.read_csv("/home/jigao/Workspace/BigAnn/playground/query_complexity/query_to_30NN_centriod_distance.txt", sep = r'\s*:=\s*', engine = "python")
df_query_40Centriod = pd.read_csv("/home/jigao/Workspace/BigAnn/playground/query_complexity/query_to_40NN_centriod_distance.txt", sep = r'\s*:=\s*', engine = "python")
df_query_50Centriod = pd.read_csv("/home/jigao/Workspace/BigAnn/playground/query_complexity/query_to_50NN_centriod_distance.txt", sep = r'\s*:=\s*', engine = "python")
df_gt_9NN = pd.read_csv("/home/jigao/Workspace/BigAnn/playground/query_complexity/query_to_9thGT_distance.txt", sep = r'\s*:=\s*', engine = "python")

query_10Centriod = df_query_10Centriod['query_10Centriod']
query_20Centriod = df_query_20Centriod['query_20Centriod']
query_30Centriod = df_query_30Centriod['query_30Centriod']
query_40Centriod = df_query_40Centriod['query_40Centriod']
query_50Centriod = df_query_50Centriod['query_50Centriod']
gt_9NN = df_gt_9NN['gt_9NN']
# print(query_50Centriod)
# print(gt_9NN)

num_query = 10000
# True  := Distance(query, nthCentriod) > Distance(query, 9NN GT)
# False := Distance(query, nthCentriod) < Distance(query, 9NN GT)

print("========================================")
print("Distance(query, 10th Centriod) >= Distance(query, 9NN GT)")
bool_df_10 = query_10Centriod >= gt_9NN
print(bool_df_10.describe())
print("========================================")

print("Distance(query, 20th Centriod) >= Distance(query, 9NN GT)")
bool_df_20 = query_20Centriod >= gt_9NN
print(bool_df_20.describe())
print("========================================")

print("Distance(query, 30th Centriod) >= Distance(query, 9NN GT)")
bool_df_30 = query_30Centriod >= gt_9NN
print(bool_df_30.describe())
print("========================================")

print("Distance(query, 40th Centriod) >= Distance(query, 9NN GT)")
bool_df_40 = query_40Centriod >= gt_9NN
print(bool_df_40.describe())
print("========================================")

print("Distance(query, 50th Centriod) >= Distance(query, 9NN GT)")
bool_df_50 = query_50Centriod >= gt_9NN
print(bool_df_50.describe())
print("========================================")

for column_name, item in bool_df_20.iteritems():
    if item == False:
        # print(str(column_name) + " having false")
        print(str(column_name))



# for index, row in bool_df.iterrows():
#     if row == False:
#         print(index)