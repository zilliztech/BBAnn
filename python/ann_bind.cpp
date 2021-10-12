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
// #include "lib/bbannlib2.h"
namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<unsigned>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

template <class indexT, class TypeNameWrapper>
void IndexBindWrapper(py::module_ &m) {
  using paraT = typename indexT::parameterType;
  using dataT = typename indexT::dataType;
  py::class_<indexT>(m, TypeNameWrapper::Get())
      .def(py::init([](MetricType metric) {
        return std::unique_ptr<indexT>(new indexT(metric));
      }))
      .def("load_index", &indexT::LoadIndex, py::arg("index_path_prefix"))
      .def("batch_search",
           [](indexT &self,
              py::array_t<dataT, py::array::c_style | py::array::forcecast>
                  &query,
              uint64_t dim, uint64_t numQuery, uint64_t knn, const paraT para)
               -> std::pair<py::array_t<unsigned>, py::array_t<float>> {
             using distanceT = typename TypeWrapper<dataT>::distanceT;
             py::array_t<unsigned> ids({numQuery, knn});
             py::array_t<float> dists({numQuery, knn});
             const dataT *pquery = query.data();
             distanceT *answer_dists = new distanceT[(int64_t)numQuery * knn];
             uint32_t *answer_ids = new uint32_t[(int64_t)numQuery * knn];
             self.BatchSearchCpp(pquery, dim, numQuery, knn, para, answer_ids,
                                 answer_dists);
             auto r = ids.mutable_unchecked();
             auto d = dists.mutable_unchecked();
             for (uint64_t i = 0; i < numQuery; ++i)
               for (uint64_t j = 0; j < knn; ++j) {
                 r(i, j) = (unsigned)answer_ids[i * knn + j];
                 d(i, j) = (float)answer_dists[i * knn + j];
               }
             return std::make_pair(ids, dists);
           },
           py::arg("query"), py::arg("dim"), py::arg("num_query"),
           py::arg("knn"), py::arg("para"))
      .def("build",
           [](indexT &self, paraT para) { return indexT::BuildIndex(para); },
           py::arg("para"))
      .def("range_search",
           [](indexT &self,
              py::array_t<dataT, py::array::c_style | py::array::forcecast>
                  &query,
              uint64_t dim, uint64_t numQuery, double radius, const paraT para)
               -> std::pair<py::array_t<unsigned>, std::pair<py::array_t<unsigned>, py::array_t<float>>> {
              const dataT *pquery = query.data();
              std::vector<std::vector<uint32_t>> ids(numQuery);
              std::vector<std::vector<float>> dists(numQuery);
              std::vector<uint64_t> lims(numQuery+1);

              self.RangeSearchCpp(pquery, dim, numQuery, radius, para, ids, dists, lims);

              uint64_t total = lims.back();
              py::array_t<unsigned> res_ids(total);
              py::array_t<float> res_dists(total);
              py::array_t<unsigned> res_lims(numQuery+1);

              auto res_ids_mutable = res_ids.mutable_unchecked();
              auto res_dists_mutable = res_dists.mutable_unchecked();
              auto res_lims_mutable = res_lims.mutable_unchecked();
              size_t pos = 0;
              for (uint64_t i = 0; i < numQuery; ++i) {
                for (uint64_t j = 0; j < ids[i].size(); ++j) {
                  res_ids_mutable(pos) = (unsigned)ids[i][j];
                  res_dists_mutable(pos++) = dists[i][j];
                }
                res_lims_mutable(i) = lims[i];
              }
              res_lims_mutable(numQuery) = lims[numQuery];

              return std::make_pair(res_lims, std::make_pair(res_ids, res_dists));
           },
           py::arg("query"), py::arg("dim"), py::arg("num_query"),
           py::arg("radius"), py::arg("para"));
}

template <class dataT>
void DataReaderBindWrapper(py::module_ &m, const char * readerName) {
  m.def(readerName,
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
      .def_readwrite("efSearch", &BBAnnParameters::efSearch)
      .def_readwrite("K1", &BBAnnParameters::K1)
      .def_readwrite("K", &BBAnnParameters::K)
      .def_readwrite("nProbe", &BBAnnParameters::nProbe)
      .def_readwrite("blockSize", &BBAnnParameters::blockSize);
#define CLASSWRAPPER_DECL(className, index)                                    \
  class className {                                                            \
  public:                                                                      \
    static const char *Get() { return index; }                                 \
  }
  CLASSWRAPPER_DECL(FloatWrapper, "FloatIndex");
  CLASSWRAPPER_DECL(UInt8Wrapper, "UInt8Index");
  CLASSWRAPPER_DECL(Int8Wrapper, "Int8Index");
#undef CLASSWRAPPER_DECL

  IndexBindWrapper<BBAnnIndex<float, BBAnnParameters>, FloatWrapper>(m);
  IndexBindWrapper<BBAnnIndex<uint8_t, BBAnnParameters>, UInt8Wrapper>(m);
  IndexBindWrapper<BBAnnIndex<int8_t, BBAnnParameters>, Int8Wrapper>(m);
  DataReaderBindWrapper<float>(m, "read_bin_float");
  DataReaderBindWrapper<int8_t>(m, "read_bin_int8");
  DataReaderBindWrapper<uint8_t>(m, "read_bin_uint8");
}
