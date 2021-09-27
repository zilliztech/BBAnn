import os
import time
import numpy as np
import psutil
from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS, download_accelerated

import bbannpy


class BbANN(BaseANN):
    def __init__(self, metric, index_params):
        self.metric = metric
        self.index_params = index_params
        self.R = index_params.get("R")

    def done(self):
        pass

    def index_name(self):
        """
        File name for the index.
        """
        return f"R{self._R}"

    def create_index_dir(self, dataset):
        """
        Return a folder name, in which we would store the index.
        """
        index_dir = os.path.join(os.getcwd(), "bbann_index")
        return index_dir

    def set_index_type(self, ds_distance, ds_dtype):
        if ds_distance == "euclidean":
            metric = bbannpy.L2
        elif ds_distance == "ip":
            metric = bbannpy.INNER_PRODUCT
        else:
            print("Unsuported distance function.")
            return False

        if not hasattr(self, 'index'):
            if ds_dtype == "float32":
                self.index = bbannpy.FloatIndex(metric)
            elif ds_dtype == "int8":
                self.index = bbannpy.Int8Index(metric)
            elif ds_dtype == "uint8":
                self.index = bbannpy.UInt8Index(metric)
            else:
                print("Unsupported data type.")
                return False
        return True

    def fit(self, dataset):
        """
        Build the index for the data points given in dataset name.
        Assumes that after fitting index is loaded in memory.
        """
        ds = DATASETS[dataset]()
        d = ds.d
        index_dir = self.create_index_dir(ds)
        self.index_path = os.path.join(index_dir, self.index_name())
        if not self.set_index_type(ds.distance(), ds.dtype):
            return False

        start = time.time()
        self.index.build(
            ds.get_dataset_fn(),
            index_path=self.index_path)
        end = time.time()
        print("bbAnn index built in %.3f s" % (end - start))
        print(f"Loading index from {self.index_path}")
        self.index.load_index(self.index_path)

    def load_index(self, dataset):
        """
        Load the index for dataset. Returns False if index
        is not available, True otherwise.

        Checking the index usually involves the dataset name
        and the index build paramters passed during construction.
        """
        ds = DATASETS[dataset]()
        index_dir = self.create_index_dir(ds)
        if not (os.path.exists(index_dir)) and 'url' not in self._index_params:
            return False
        index_path = os.path.join(index_dir, self.index_name())
        print(f"Loading index from {self.index_path}")
        self.index.load_index(self.index_path)

    def query(self, X, k):
        """Carry out a batch query for k-NN of query set X."""
        nq, dim = (np.shape(X))
        self.res, self.query_dists = self.index.batch_search(X, dim, nq, k)
        print(self.res)

    def range_query(self, X, radius):
        """
        Carry out a batch query for range search with
        radius.
        """
        # TODO(!!!!!)
        pass

    def get_results(self):
        """
        Helper method to convert query results of k-NN search.
        If there are nq queries, returns a (nq, k) array of integers
        representing the indices of the k-NN for each query.
        """
        return self.res

    def get_range_results(self):
        """
        Helper method to convert query results of range search.
        If there are nq queries, returns a triple lims, I, D.
        lims is a (nq) array, such that

            I[lims[q]:lims[q + 1]] in int

        are the indiices of the indices of the range results of query q, and

            D[lims[q]:lims[q + 1]] in float

        are the distances.
        """
        return self.res

    def get_additional(self):
        """
        Allows to retrieve additional results.
        """
        return {}

    def __str__(self):
        return "BBANN"

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024
