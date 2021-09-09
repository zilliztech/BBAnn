import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from vecs_io import *
from matplotlib.ticker import EngFormatter
import os


# fontsize = 20
# ticksize = 20

def delete_dir_if_exist(dire):
    if os.path.isdir(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


# datasets = ['audio', 'imagenet', 'movielens', 'music100', 'netflix', 'normal-64', 'tiny5m', 'word2vec', 'yahoomusic']
datasets = ['text-to-image']


def plot(dataset):
    # path = '../data/{}/{}_base.fvecs'.format(dataset, dataset)
    path = "/home/jigao/Desktop/Yandex.TexttoImage.base.10M.fdata"
    # X, dim = fvecs_read(path)
    X, num_points, dim = fbin_read(path)
    norms = np.linalg.norm(X, axis=1)
    norms[:] = norms[:] / np.max(norms)
    norms = np.sort(norms)
    median = int(len(norms) / 2)
    print(norms[median])

    ax = plt.gca()
    # mkfunc = lambda x, pos: '%1.0fM' % (x * 1e-6) if x >= 1e6 \
    #     else '%1.0fK' % (x * 1e-3) if x >= 1e3 \
    #     else '%1.0f' % x if x > 0 \
    #     else '0 '

    formatter1 = EngFormatter(places=1, sep="\N{THIN SPACE}")  # U+2009
    ax.yaxis.set_major_formatter(formatter1)

    # mkformatter = matplotlib.ticker.FuncFormatter(mkfunc)
    # ax.yaxis.set_major_formatter(mkformatter)

    # plt.figure(figsize=(12.8, 9.25))
    # plt.axis([0, 1, 0, 2e6])
    plt.hist(norms, bins=20, color='dimgray')
    plt.legend(["%s, median %.6f" % (dataset, norms[median])], loc='upper right')
    plt.xlabel('Norm')
    plt.xticks()
    plt.ylabel('Frequency')
    plt.yticks()

    # plt.xlabel('Norm', fontsize=fontsize)
    # plt.xticks(fontsize=ticksize)
    # plt.ylabel('Frequency', fontsize=fontsize)
    # plt.yticks(fontsize=ticksize)

    # plt.tight_layout()

    plt.savefig('./norm-distribution/{}.png'.format(dataset))
    plt.close()
    # plt.show()


if __name__ == '__main__':
    # delete_dir_if_exist('norm-distribution')
    # os.mkdir('norm-distribution')
    for dataset in datasets:
        plot(dataset)
