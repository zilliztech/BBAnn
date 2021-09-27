import numpy as np
import struct


# to get the .vecs
# np.set_printoptions(threshold=np.inf)  # display all the content when print the numpy array


def bbin_read(fname):
    a = np.fromfile(fname, dtype='uint8')
    num_vector = a[:4].view('int32')[0]
    dim = a[4:8].view('int32')[0]
    a = a[8:].reshape(-1, dim)
    return a.reshape(-1, dim), num_vector, dim


def ibin_read(fname):
    a = np.fromfile(fname, dtype='int32')
    n_data = a[0]
    dim = a[1]
    a = a[2:].reshape(-1, dim)
    return a.reshape(-1, dim), n_data, dim


def fbin_read(fname):
    data, n_data, dim = ibin_read(fname)
    return data.view('float32').astype(np.float32), n_data, dim


def bbin_write(filename, vecs):
    f = open(filename, "wb")
    length = [len(vecs)]
    dimension = [len(vecs[0])]
    f.write(struct.pack('I' * len(length), *length))
    f.write(struct.pack('I' * len(dimension), *dimension))

    for x in vecs:
        f.write(struct.pack('B' * len(x), *x))

    f.close()


def ibin_write(filename, vecs):
    f = open(filename, "wb")
    length = [len(vecs)]
    dimension = [len(vecs[0])]
    f.write(struct.pack('I' * len(length), *length))
    f.write(struct.pack('I' * len(dimension), *dimension))

    for x in vecs:
        f.write(struct.pack('I' * len(x), *x))

    f.close()


def fbin_write(filename, vecs):
    f = open(filename, "wb")
    length = [len(vecs)]
    dimension = [len(vecs[0])]
    f.write(struct.pack('I' * len(length), *length))
    f.write(struct.pack('I' * len(dimension), *dimension))

    for x in vecs:
        f.write(struct.pack('f' * len(x), *x))

    f.close()


# if __name__ == '__main__':
#     # data, n_data, dim = fbin_read("/home/bianzheng/NIPS-Competition/data/text-to-image/query.public.100K.fbin")
#     data, n_data, dim = fbin_read("/home/bianzheng/NIPS-Competition/script/tmp.fbin")
#     # fbin_write('tmp.fbin', data)
#     print(data.shape)
#     print(n_data, dim)
#     print(data[0])
