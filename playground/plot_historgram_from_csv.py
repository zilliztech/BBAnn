#!/usr/bin/env python3
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import seaborn as sns

# Input the file name
path = "/tmp/h.csv"
# path = input("Please type in the input file input_path generated by C++ program: ")
df = pd.read_csv(path, sep = r'\s*,\s*')
# print(df)
# headers = ['norm value range', 'counter', 'percentage']
# print(df.columns.tolist())

x = df['norm value range']
y = df['percentage']
# plt.barh(x, y)
sns.set_theme(style = "whitegrid", font_scale= 0.8)
plots = sns.barplot(x = 'norm value range', y = 'percentage', data = df)

# Iterrating over the bars one-by-one
for bar in plots.patches:
    # Using Matplotlib's annotate function and
    # passing the coordinates where the annotation shall be done
    # x-coordinate: bar.get_x() + bar.get_width() / 2
    # y-coordinate: bar.get_height()
    # free space to be left to make graph pleasing: (0, 8)
    # ha and va stand for the horizontal and vertical alignment
    plots.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')

plt.xlabel("Norm Value Range", size = 20)
plt.ylabel("Percentage", size = 20)
plt.title("Yandex Text To Image 1B: Norm Distribution", size = 30)
plt.show()


