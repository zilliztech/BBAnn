// defines interface for AnnIndex
#pragma once
#include <stdint.h>
#include <string>
#include <vector>

template <typename T> struct TypeWrapper;
template <> struct TypeWrapper<float> { using distanceT = float; };
template <> struct TypeWrapper<uint8_t> { using distanceT = uint32_t; };
template <> struct TypeWrapper<int8_t> { using distanceT = int32_t; };

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
  using parameterType = paraT;
  using dataType = dataT;
  using distanceT = typename TypeWrapper<dataT>::distanceT;

public:
  virtual ~AnnIndexInterface() = default;

  // Load the in-mem part of the index into memory.
  // Returns true if load success.
  virtual bool LoadIndex(std::string &indexPathPrefix) = 0;

  // To construct a index with parameters and settings given in *para*,
  // users shall call <>.BuildIndex().
  // The class that implements the interface must contain this method:
  // static void BuildIndexImpl(const paraT para);

  // Conduct a top-k Ann search for items in *pquery*, which is an arry of dataT
  // type, converted from numpy array of dim*numQuery items .
  virtual void BatchSearchCpp(const dataT *pquery, uint64_t dim,
                              uint64_t numQuery, uint64_t knn, const paraT para,
                              uint32_t *answer_ids,
                              distanceT *answer_dists) = 0;

  //
  // Conduct a range search.
  // input:
  // *pquery, which is an arry of dataT type,  converted from numpy
  // array of dim*numQuery items.
  // returns: following three vectors
  // 

  // Range search with a radius, returns vectors's ids and distances within that
  // radius from each query, the lims array represents prefix sums of results'
  // lengths -- 
  // TODO(!!!!! -- updatedoc)
  virtual std::tuple<std::vector<uint32_t>, std::vector<float>, std::vector<uint64_t>>
 RangeSearchCpp(const dataT *pquery, uint64_t dim,
                              uint64_t numQuery, double radius,
                              const paraT para) = 0;
};
