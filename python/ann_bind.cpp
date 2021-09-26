#include "util/defines.h"
#include <fstream>
#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>

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
  std::cout << "Build:";
  // TODO(!!!!);
}

static const char * getStr(){
	return "PyName";
}

template <typename dataT> void IndexBindWrapper(py::module_ &m) {
  py::class_<BBAnnIndex<dataT>> (m, getStr())
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

  IndexBindWrapper<float>(m);
  // IndexBindWrapper<uint8_t>("UInt8Index");
  // IndexBindWrapper<int8_t>("Int8Index");
}
