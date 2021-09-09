import numpy as np
import struct


# to get the .vecs
# np.set_printoptions(threshold=np.inf)  # display all the content when print the numpy array


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy(), d


def fvecs_read(fname):
    data, d = ivecs_read(fname)
    return data.view('float32').astype(np.float32), d


def bvecs_read(fname):
    a = np.fromfile(fname, dtype='uint8')
    d = a[:4].view('uint8')[0]
    return a.reshape(-1, d + 4)[:, 4:].copy(), d


# store in format of vecs
def fvecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))  # *dimension就是int, dimension就是list
        f.write(struct.pack('f' * len(x), *x))

    f.close()


def ivecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('i' * len(x), *x))

    f.close()


def bvecs_write(filename, vecs):
    f = open(filename, "wb")
    dimension = [len(vecs[0])]

    for x in vecs:
        f.write(struct.pack('i' * len(dimension), *dimension))
        f.write(struct.pack('B' * len(x), *x))

    f.close()


def loader(data_set='audio', top_k=20, ground_metric='euclid', folder='../data/', data_type='fvecs'):
    """
    :param data_set: data set you wanna load , audio, sift1m, ..
    :param top_k: how many nearest neighbor in ground truth file
    :param ground_metric:
    :param folder:
    :return: X, T, Q, G
    """
    folder_path = folder + data_set
    base_file = folder_path + '/%s_base.%s' % (data_set, data_type)
    train_file = folder_path + '/%s_learn.%s' % (data_set, data_type)
    query_file = folder_path + '/%s_query.%s' % (data_set, data_type)
    ground_truth = '/home/zhengbian/ip-nsw/data/%s/gnd.ivecs' % data_set

    print("# load the base data {}, \n# load the queries {}, \n# load the ground truth {}".format(base_file, query_file,
                                                                                                  ground_truth))
    if data_type == 'fvecs':
        X = fvecs_read(base_file)[0]
        Q = fvecs_read(query_file)[0]
        try:
            T = fvecs_read(train_file)[0]
        except FileNotFoundError:
            T = None
    elif data_type == 'bvecs':
        X = bvecs_read(base_file)[0].astype(np.float32)
        Q = bvecs_read(query_file)[0].astype(np.float32)
        try:
            T = bvecs_read(train_file)[0]
        except FileNotFoundError:
            T = None
    else:
        assert False
    try:
        G = ivecs_read(ground_truth)[0]
    except FileNotFoundError:
        G = None
    return X, T, Q, G


if __name__ == '__main__':
    data, d = ivecs_read("/home/zhengbian/plus-ip-nsw/data/audio/audio_base.fvecs")
    print(data.shape)
    print(d)
