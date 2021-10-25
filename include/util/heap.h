#pragma once
#include <limits>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

template <typename T_, typename TI_> struct CMax;

// traits of minheaps = heaps where the minimum value is stored on top
// useful to find the *max* values of an array
template <typename T_, typename TI_> struct CMin {
  typedef T_ T;
  typedef TI_ TI;
  typedef CMax<T_, TI_> Crev;
  inline static bool cmp(T a, T b) { return a < b; }
  // value that will be popped first -> must be smaller than all others
  // for int types this is not strictly the smallest val (-max - 1)
  inline static T neutral() { return -std::numeric_limits<T>::max(); }
};

template <typename T_, typename TI_> struct CMax {
  typedef T_ T;
  typedef TI_ TI;
  typedef CMin<T_, TI_> Crev;
  inline static bool cmp(T a, T b) { return a > b; }
  inline static T neutral() { return std::numeric_limits<T>::max(); }
};

template <class C>
inline void heap_heapify(size_t k, typename C::T *bh_val,
                         typename C::TI *bh_ids) {
  for (size_t i = 0; i < k; i++) {
    bh_val[i] = C::neutral();
    bh_ids[i] = -1;
  }
}

template <class C>
inline void heap_swap_top(size_t k, typename C::T *bh_val,
                          typename C::TI *bh_ids, typename C::T val,
                          typename C::TI ids) {
  bh_val--; /* Use 1-based indexing for easier node->child translation */
  bh_ids--;
  size_t i = 1, i1, i2;
  while (1) {
    i1 = i << 1;
    i2 = i1 + 1;
    if (i1 > k)
      break;
    if (i2 == k + 1 || C::cmp(bh_val[i1], bh_val[i2])) {
      if (C::cmp(val, bh_val[i1]))
        break;
      bh_val[i] = bh_val[i1];
      bh_ids[i] = bh_ids[i1];
      i = i1;
    } else {
      if (C::cmp(val, bh_val[i2]))
        break;
      bh_val[i] = bh_val[i2];
      bh_ids[i] = bh_ids[i2];
      i = i2;
    }
  }
  bh_val[i] = val;
  bh_ids[i] = ids;
}

template <class C>
inline void heap_pop(size_t k, typename C::T *bh_val, typename C::TI *bh_ids) {
  bh_val--; /* Use 1-based indexing for easier node->child translation */
  bh_ids--;
  typename C::T val = bh_val[k];
  size_t i = 1, i1, i2;
  while (1) {
    i1 = i << 1;
    i2 = i1 + 1;
    if (i1 > k)
      break;
    if (i2 == k + 1 || C::cmp(bh_val[i1], bh_val[i2])) {
      if (C::cmp(val, bh_val[i1]))
        break;
      bh_val[i] = bh_val[i1];
      bh_ids[i] = bh_ids[i1];
      i = i1;
    } else {
      if (C::cmp(val, bh_val[i2]))
        break;
      bh_val[i] = bh_val[i2];
      bh_ids[i] = bh_ids[i2];
      i = i2;
    }
  }
  bh_val[i] = bh_val[k];
  bh_ids[i] = bh_ids[k];
}

template <class C>
inline void heap_push(size_t k, typename C::T *bh_val, typename C::TI *bh_ids,
                      typename C::T val, typename C::TI ids) {
  bh_val--; /* Use 1-based indexing for easier node->child translation */
  bh_ids--;
  size_t i = k, i_father;
  while (i > 1) {
    i_father = i >> 1;
    if (!C::cmp(val, bh_val[i_father])) /* the heap structure is ok */
      break;
    bh_val[i] = bh_val[i_father];
    bh_ids[i] = bh_ids[i_father];
    i = i_father;
  }
  bh_val[i] = val;
  bh_ids[i] = ids;
}

template <typename C>
inline size_t heap_reorder(size_t k, typename C::T *bh_val,
                           typename C::TI *bh_ids) {
  size_t i, ii;

  for (i = 0, ii = 0; i < k; i++) {
    /* top element should be put at the end of the list */
    typename C::T val = bh_val[0];
    typename C::TI id = bh_ids[0];

    /* boundary case: we will over-ride this value if not a true element */
    heap_pop<C>(k - i, bh_val, bh_ids);
    bh_val[k - ii - 1] = val;
    bh_ids[k - ii - 1] = id;
    if (id != -1)
      ii++;
  }
  /* Count the number of elements which are effectively returned */
  size_t nel = ii;

  memmove(bh_val, bh_val + k - ii, ii * sizeof(*bh_val));
  memmove(bh_ids, bh_ids + k - ii, ii * sizeof(*bh_ids));

  for (; ii < k; ii++) {
    bh_val[ii] = C::neutral();
    bh_ids[ii] = -1;
  }
  return nel;
}