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

template <class TClass, typename paraT> class BuildIndexFactory {
public:
  static void BuildIndex(const paraT para) { TClass::BuildIndexImpl(para); }
};

// Interface for Ann index which the base data type of the vector space is
// dataT, and the index should be accessed via parameters/settings packed in
// paraT.
//    dataT should be one of [uint8_t, int8_t, float(32-bit)].
//    see BBAnnParameters for example paraT.
template <typename dataT, typename paraT> class AnnIndexInterface {
public:
  virtual ~AnnIndexInterface() = default;

  // Load the in-mem part of the index into memory.
  // Returns true if load success.
  virtual bool LoadIndex(std::string &indexPathPrefix) = 0;

  // To construct a index with parameters and settings given in *para*,
  // users shall call <>.BuildIndex().
  // The class that implements the interface must contain this method:
  // static void BuildIndexImpl(const paraT para);

  // Conduct a top-k Ann search for items in *query*, which is a numpy array of
  // dim*numQuery items converted to dataT type.
  virtual std::pair<py::array_t<unsigned>, py::array_t<float>> /* do batch */
  BatchSearch(
      py::array_t<dataT, py::array::c_style | py::array::forcecast> &query,
      uint64_t dim, uint64_t numQuery, uint64_t knn, const paraT para) = 0;
};
template <typename T> struct TypeWrapper;
template <> struct TypeWrapper<float> { using distanceT = float; };
template <> struct TypeWrapper<uint8_t> { using distanceT = uint32_t; };
template <> struct TypeWrapper<int8_t> { using distanceT = int32_t; };

template <typename dataT, typename paraT>
struct BBAnnIndex : public BuildIndexFactory<BBAnnIndex<dataT, paraT>, paraT>,
                    public AnnIndexInterface<dataT, paraT> {
public:
  using parameterType = paraT;
  using dataType = dataT;

  BBAnnIndex(MetricType metric) : metric_(metric) {
    std::cout << "BBAnnIndex constructor" << std::endl;
  }
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

  std::pair<py::array_t<unsigned>, py::array_t<float>> /* do batch */
  BatchSearch(
      py::array_t<dataT, py::array::c_style | py::array::forcecast> &query,
      uint64_t dim, uint64_t numQuery, uint64_t knn,
      const paraT para) override {

    std::cout << "Query: ";

    using distanceT = typename TypeWrapper<dataT>::distanceT;
    py::array_t<unsigned> ids({numQuery, knn});
    py::array_t<float> dists({numQuery, knn});
    const dataT *pquery = query.data();

    distanceT *answer_dists = new distanceT[(int64_t)numQuery * knn];
    uint32_t *answer_ids = new uint32_t[(int64_t)numQuery * knn];

    switch (para.metric) {
    case MetricType::L2: {
      Computer<dataT, dataT, distanceT> dis_computer =
          L2sqr<const dataT, const dataT, distanceT>;
      search_bbann_queryonly<dataT, distanceT, CMax<distanceT, uint32_t>>(
          indexPrefix_, para.nProbe, para.hnswefC, knn, index_hnsw_, para.K1,
          para.blockSize, dis_computer, pquery, answer_ids, answer_dists,
          numQuery, dim);
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

  static void BuildIndexImpl(const paraT para) {
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
};

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

template <class indexT, class TypeNameWrapper>
void IndexBindWrapper(py::module_ &m) {
  using paraT = typename indexT::parameterType;
  using dataT = typename indexT::dataType;
  py::class_<indexT>(m, TypeNameWrapper::Get())
      .def(py::init([](MetricType metric) {
        return std::unique_ptr<indexT>(new indexT(metric));
      }))
      .def("LoadIndex", &indexT::LoadIndex, py::arg("index_path_prefix"))
      .def("batch_search", &indexT::BatchSearch, py::arg("query"),
           py::arg("dim"), py::arg("num_query"), py::arg("knn"),
           py::arg("para"))
      .def("build",
           [](indexT &self, paraT para) { return indexT::BuildIndex(para); },
           py::arg("para"));

  m.def(TypeNameWrapper::ReaderName(),
        [](const std::string &path, std::vector<dataT> &data) {
          size_t num, dim, aligned_dims;
          // TODO(!!!!!!)  check disann implementaion, what is rounded dims?

          std::cout << "read binary file from " << path << std::endl;
          std::ifstream reader(path, std::ios::binary);
          num = 0; // not sure why but this one is necessary.
          dim = 0;
          reader.read((char *)&num, sizeof(uint32_t));
          reader.read((char *)&dim, sizeof(uint32_t));

          std::cout << num << ", dim = " << dim << std::endl;
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

  py::bind_vector<std::vector<unsigned>>(m, "VectorUnsigned");
  py::bind_vector<std::vector<float>>(m, "VectorFloat");

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
    static const char *ReaderName() { return "read_bin_float"; }
  };
  class UInt8Wrapper {
  public:
    static const char *Get() { return "Uint8Index"; }
    static const char *ReaderName() { return "read_bin_uint8"; }
  };

  class Int8Wrapper {
  public:
    static const char *Get() { return "Int8Index"; }
    static const char *ReaderName() { return "read_bin_int8"; }
  };

  IndexBindWrapper<BBAnnIndex<float, BBAnnParameters>, FloatWrapper>(m);
  IndexBindWrapper<uint8_t, BBAnnIndex<uint8_t>, UInt8Wrapper>(m);
  IndexBindWrapper<int8_t, BBAnnIndex<int8_t>, Int8Wrapper>(m);
}
