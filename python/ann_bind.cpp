#include <fstream>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>

#include "ann_interface.h"
#include "bbann.h"
#include "lib/bbannlib.h"
namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<unsigned>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

template <typename dataT, typename paraT>
struct BBAnnIndexPy : public BBAnnIndex<dataT, paraT> {
  BBAnnIndexPy(MetricType &metric) : BBAnnIndex<dataT, paraT>(metric){};
  /* do batch with numpy interface */
  std::pair<py::array_t<unsigned>, py::array_t<float>> BatchSearch(
      py::array_t<dataT, py::array::c_style | py::array::forcecast> &query,
      uint64_t dim, uint64_t numQuery, uint64_t knn, const paraT para) {
    using distanceT = typename TypeWrapper<dataT>::distanceT;

    py::array_t<unsigned> ids({numQuery, knn});
    py::array_t<float> dists({numQuery, knn});
    const dataT *pquery = query.data();
    distanceT *answer_dists = new distanceT[(int64_t)numQuery * knn];
    uint32_t *answer_ids = new uint32_t[(int64_t)numQuery * knn];

    BBAnnIndex<dataT, paraT>::BatchSearchCpp(pquery, dim, numQuery, knn, para,
                                             answer_ids, answer_dists);

    auto r = ids.mutable_unchecked();
    auto d = dists.mutable_unchecked();
    for (uint64_t i = 0; i < numQuery; ++i)
      for (uint64_t j = 0; j < knn; ++j) {
        r(i, j) = (unsigned)answer_ids[i * knn + j];
        d(i, j) = (float)answer_dists[i * knn + j];
      }
    return std::make_pair(ids, dists);
  }
};
template <class indexT, class TypeNameWrapper>
void IndexBindWrapper(py::module_ &m) {
  using paraT = typename indexT::parameterType;
  using dataT = typename indexT::dataType;
  py::class_<indexT>(m, TypeNameWrapper::Get())
      .def(py::init([](MetricType metric) {
        TestLib ann;
        std::cout << ann.Get();
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

  IndexBindWrapper<BBAnnIndexPy<float, BBAnnParameters>, FloatWrapper>(m);
  IndexBindWrapper<BBAnnIndexPy<uint8_t, BBAnnParameters>, UInt8Wrapper>(m);
  IndexBindWrapper<BBAnnIndexPy<int8_t, BBAnnParameters>, Int8Wrapper>(m);
}
