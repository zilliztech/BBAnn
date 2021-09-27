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

template <typename dataT> struct BBAnnIndex {
  BBAnnIndex(MetricType metric) : metric_(metric) {}
  MetricType metric_;

  void LoadIndex(std::string &indexPathPrefix) {
    std::cout << "Loading: " << indexPathPrefix;
  }

  auto BatchSearch(
      py::array_t<dataT, py::array::c_style | py::array::forcecast> &query,
      uint64_t dim, uint64_t numQuery, uint64_t knn) {
    py::array_t<unsigned> ids(knn);
    py::array_t<float> dists(knn);
    std::cout << "Query: ";
    // TODO(!!!!)

    return std::make_pair(ids, dists);
  }
};

template <typename dataT>
void BuildBBAnnIndex(const std::string &dataFilePath,
                     const std::string indexPrefixPath, MetricType metric) {
  std::cout << "Build from data type";
  // TODO(!!!!);
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
           py::arg("dim"), py::arg("num_query"), py::arg("knn"))
      .def("build",
           [](BBAnnIndex<dataT> &self, const char *data_file_path,
              const char *index_prefix_path) {
             BuildBBAnnIndex<dataT>(data_file_path, index_prefix_path,
                                    self.metric_);
           },
           py::arg("data_file_path"), py::arg("index_prefix_path"));
}

PYBIND11_MODULE(bbannpy, m) {
  m.doc() = "TBD"; // optional module docstring

  py::enum_<MetricType>(m, "Metric")
      .value("L2", MetricType::L2)
      .value("INNER_PRODUCT", MetricType::IP)
      .export_values();

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
