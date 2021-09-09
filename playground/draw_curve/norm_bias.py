import numpy as np
import matplotlib.pyplot as plt
from vecs_io import loader
import os

# data_set = 'tinygist10million'
# data_set = 'netflix'
# data_set = 'yahoomusic'
# data_set = 'sift1m'
# data_set = 'imagenet'
# data_set = 'movielens'
# data_set = 'music100'

fontsize = 44
ticksize = 36


def delete_dir_if_exist(dire):
    if os.path.isdir(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def topk_distribution(data_set):
    split = 20
    top = 5

    X, T, Q, G = loader(data_set, 10)
    total_percents = np.zeros(shape=top + 1)
    norms = np.linalg.norm(X, axis=1)
    arg_norms = np.argsort(- norms)
    for tmp_gnd in G:
        # top_k_set = np.unique(G)
        # percents = [len(np.intersect1d(i, top_k_set)) / len(top_k_set) for i in np.array_split(arg_norms, split)]
        percents = [len(np.intersect1d(i, tmp_gnd)) / len(tmp_gnd) for i in np.array_split(arg_norms, split)]
        percents[top] = np.sum(percents[top:])
        percents = percents[:top + 1]
        # print(percents)
        total_percents += percents
    print(total_percents)
    total_percents /= len(G)
    print(total_percents)

    x = range(len(total_percents))
    plt.bar(x, total_percents, tick_label=['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-100%'], fc='darkgray')
    plt.xlabel('Norm Ranking')
    plt.ylabel('Percentage')
    # plt.xlabel('Norm Ranking', fontsize=fontsize)
    # plt.ylabel('Percentage', fontsize=fontsize)
    # plt.title('yahoo!music: Position of the ground truth top-10 in norm distribution', fontsize=24)
    plt.title('%s: Position of top-10 in norm distribution' % data_set)

    plt.xticks()
    plt.yticks()
    # plt.xticks(fontsize=ticksize - 8)
    # plt.yticks(fontsize=ticksize)
    plt.savefig('topk-norm-distribution/%s.png' % data_set)
    plt.close()


if __name__ == '__main__':
    data_set_l = ['text-to-image']
    # data_set_l = ['audio', 'imagenet', 'movielens', 'music100', 'netflix', 'normal-64', 'tiny5m', 'word2vec',
    #               'yahoomusic']
    # delete_dir_if_exist('topk-norm-distribution')
    # os.mkdir('topk-norm-distribution')
    for data_set in data_set_l:
        topk_distribution(data_set)
