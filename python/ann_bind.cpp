#include <fstream>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>

#include "bbann.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<unsigned>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

namespace py = pybind11;

struct BBAnnParameters {
  std::string dataFilePath;
  std::string indexPrefixPath;
  std::string queryPath;
  std::string groundTruthFilePath;
  MetricType metric;
  int K = 20; // top k.
  int hnswM = 32;
  int hnswefC = 500;
  int K1 = 20;
  int blockSize = 1;
  int nProbe = 2;
};

template <typename T> struct TypeWrapper;
template <> struct TypeWrapper<float> {
  using distanceT = float;
  using constDataT = const float;
};
template <> struct TypeWrapper<uint8_t> {
  using distanceT = uint32_t;
  using constDataT = const uint8_t;
};
template <> struct TypeWrapper<int8_t> {
  using distanceT = int32_t;
  using constDataT = const int8_t;
};

template <typename dataT> struct BBAnnIndex {
  BBAnnIndex(MetricType metric) : metric_(metric) {}
  MetricType metric_;

  bool LoadIndex(std::string &indexPathPrefix) {
    std::cout << "Loading: " << indexPathPrefix;

    std::string hnsw_index_file = indexPathPrefix + HNSW + INDEX + BIN;
    std::string bucket_centroids_file =
        indexPathPrefix + BUCKET + CENTROIDS + BIN;

    uint32_t bucket_num, dim;
    get_bin_metadata(bucket_centroids_file, bucket_num, dim);

    hnswlib::SpaceInterface<float> *space = nullptr;
    if (MetricType::L2 == metric_) {
      space = new hnswlib::L2Space(dim);
    } else if (MetricType::IP == metric_) {
      space = new hnswlib::InnerProductSpace(dim);
    }
    // load hnsw
    index_hnsw_ = std::make_shared<hnswlib::HierarchicalNSW<float>>(
        space, hnsw_index_file);
    indexPrefix_ = indexPathPrefix;
    return true;
  }

  auto BatchSearch(
      py::array_t<dataT, py::array::c_style | py::array::forcecast> &query,
      uint64_t dim, uint64_t numQuery, uint64_t knn,
      const BBAnnParameters para) {

    std::cout << "Query: ";

    using distanceT = typename TypeWrapper<dataT>::distanceT;
    using constDataT = typename TypeWrapper<dataT>::constDataT;
    py::array_t<unsigned> ids({numQuery, knn});
    py::array_t<float> dists({numQuery, knn});
    /*
    dataT *pquery = new dataT[numQuery * dim];
    for (int i = 0; i < numQuery; i++) {
      for (int j = 0; j < dim; j++) {
        pquery[i * dim + j] = query.data(i*dim+j);
      }
    }
    */
    constDataT *pquery = query.data();

    distanceT *answer_dists = new distanceT[(int64_t)numQuery * knn];
    uint32_t *answer_ids = new uint32_t[(int64_t)numQuery * knn];
    uint32_t *bucket_labels =
        new uint32_t[(int64_t)numQuery * para.nProbe]; // 400K * nprobe

    uint32_t nq, dq;

    switch (para.metric) {
    case MetricType::L2: {
      Computer<dataT, dataT, distanceT> dis_computer =
          L2sqr<constDataT, constDataT, distanceT>;
      // TODO(!!!!);
      /*
      search_bbann_exec<dataT, distanceT, CMax<distanceT, uint32_t>>(
          indexPrefix_, para.nProbe, para.hnswefC, knn, index_hnsw_, para.K1,
          para.blockSize, dis_computer, nq, dq, pquery, bucket_labels,
          answer_dists, answer_ids);*/

      break;
    }
    default:
      std::cerr << "not supported" << std::endl;
    }

    auto r = ids.mutable_unchecked();
    auto d = dists.mutable_unchecked();
    for (uint64_t i = 0; i < numQuery; ++i)
      for (uint64_t j = 0; j < knn; ++j) {
        r(i, j) = (unsigned)answer_ids[i * knn + j];
        d(i, j) = (float)answer_dists[i * knn + j];
      }

    return std::make_pair(ids, dists);
  }
  std::shared_ptr<hnswlib::HierarchicalNSW<float>> index_hnsw_;
  std::string indexPrefix_;
};

template <typename dataT> void BuildBBAnnIndex(const BBAnnParameters para) {
  std::cout << "Build start " << std::endl;
  using distanceT = typename TypeWrapper<dataT>::distanceT;
  switch (para.metric) {
  case MetricType::L2: {
    std::cout << "Build With L2" << std::endl;
    std::cout << "dataT" << typeid(dataT).name() << std::endl;
    std::cout << "distanceT" << typeid(distanceT).name() << std::endl;
    build_bbann<dataT, distanceT, CMax<distanceT, uint32_t>>(
        para.dataFilePath, para.indexPrefixPath, para.hnswM, para.hnswefC,
        para.metric, para.K1, para.blockSize);
    return;
  }
  case MetricType::IP: {
    build_bbann<dataT, distanceT, CMin<distanceT, uint32_t>>(
        para.dataFilePath, para.indexPrefixPath, para.hnswM, para.hnswefC,
        para.metric, para.K1, para.blockSize);
    return;
  }
  default:
    std::cerr << "not supported" << std::endl;
  }
}

template <typename dataT, class StringWrapper>
void IndexBindWrapper(py::module_ &m) {
  py::class_<BBAnnIndex<dataT>>(m, StringWrapper::Get())
      .def(py::init([](MetricType metric) {
        return std::unique_ptr<BBAnnIndex<dataT>>(
            new BBAnnIndex<dataT>(metric));
      }))
      .def("load_index", &BBAnnIndex<dataT>::LoadIndex,
           py::arg("index_path_prefix"))
      .def("batch_search", &BBAnnIndex<dataT>::BatchSearch, py::arg("query"),
           py::arg("dim"), py::arg("num_query"), py::arg("knn"),
           py::arg("para"))
      .def("build",
           [](BBAnnIndex<dataT> &self, BBAnnParameters para) {
             return BuildBBAnnIndex<dataT>(para);
           },
           py::arg("para"));

  m.def("load_aligned_bin",
        [](const std::string &path, std::vector<dataT> &data) {
          size_t num, dim, aligned_dims;
          // TODO(!!!!!!)  check disann implementaion, what is rounded dims?
          std::ifstream reader(path, std::ios::binary);
          reader.read((char *)&num, sizeof(uint32_t));
          reader.read((char *)&dim, sizeof(uint32_t));
          data.resize(num * dim);
          reader.read((char *)(&data[0]), sizeof(dataT) * (uint64_t)num * dim);
          reader.close();
          std::cout << "read binary file from " << path
                    << " done in ... seconds, n = " << num << ", dim = " << dim
                    << std::endl;
          aligned_dims = dim;
          auto l = py::list(3);
          l[0] = py::int_(num);
          l[1] = py::int_(dim);
          l[2] = py::int_(aligned_dims);
          return l;
        },
        py::arg("path"), py::arg("data"));
}

PYBIND11_MODULE(bbannpy, m) {
  m.doc() = "TBD"; // optional module docstring

  py::enum_<MetricType>(m, "Metric")
      .value("L2", MetricType::L2)
      .value("INNER_PRODUCT", MetricType::IP)
      .export_values();

  py::class_<BBAnnParameters>(m, "BBAnnParameters")
      .def(py::init<>())
      .def_readwrite("dataFilePath", &BBAnnParameters::dataFilePath)
      .def_readwrite("indexPrefixPath", &BBAnnParameters::indexPrefixPath)
      .def_readwrite("metric", &BBAnnParameters::metric)
      .def_readwrite("queryPath", &BBAnnParameters::queryPath)
      .def_readwrite("hnswM", &BBAnnParameters::hnswM)
      .def_readwrite("hnswefC", &BBAnnParameters::hnswefC)
      .def_readwrite("K1", &BBAnnParameters::K1)
      .def_readwrite("K", &BBAnnParameters::K)
      .def_readwrite("nProbe", &BBAnnParameters::nProbe)
      .def_readwrite("blockSize", &BBAnnParameters::blockSize);

  class FloatWrapper {
  public:
    static const char *Get() { return "FloatIndex"; }
  };
  class UInt8Wrapper {
  public:
    static const char *Get() { return "Uint8Index"; }
  };

  class Int8Wrapper {
  public:
    static const char *Get() { return "Int8Index"; }
  };

  IndexBindWrapper<float, FloatWrapper>(m);
  IndexBindWrapper<uint8_t, UInt8Wrapper>(m);
  IndexBindWrapper<int8_t, Int8Wrapper>(m);
}
