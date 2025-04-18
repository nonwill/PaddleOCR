// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "paddle/common/dim.h"
#include "paddle/common/enforce.h"
#include "paddle/common/exception.h"
#include "paddle/utils/test_macros.h"

namespace common {

#define PADDLE_VISIT_DDIM_BASE(rank, callback)                                 \
  case (rank): {                                                               \
    constexpr auto kRank = (rank);                                             \
    return (callback);                                                         \
  }

#define PADDLE_VISIT_DDIM(rank, callback)                                      \
  switch (rank) {                                                              \
    PADDLE_VISIT_DDIM_BASE(0, callback);                                       \
    PADDLE_VISIT_DDIM_BASE(1, callback);                                       \
    PADDLE_VISIT_DDIM_BASE(2, callback);                                       \
    PADDLE_VISIT_DDIM_BASE(3, callback);                                       \
    PADDLE_VISIT_DDIM_BASE(4, callback);                                       \
    PADDLE_VISIT_DDIM_BASE(5, callback);                                       \
    PADDLE_VISIT_DDIM_BASE(6, callback);                                       \
    PADDLE_VISIT_DDIM_BASE(7, callback);                                       \
    PADDLE_VISIT_DDIM_BASE(8, callback);                                       \
    PADDLE_VISIT_DDIM_BASE(9, callback);                                       \
  default:                                                                     \
    PD_THROW(                                                                  \
        "Unimplemented error. Invalid dimension to be accessed. Now only "     \
        "supports access to "                                                  \
        "dimension 0 to 9, but received dimension is ",                        \
        rank, ".");                                                            \
  }

template <typename T1, typename T2>
inline void dynamic_dim_assign(const T1 *in, T2 *out, int n) {
  if (n == -1) {
    return;
  }
  PADDLE_VISIT_DDIM(n, (static_dim_assign<kRank, T1, T2>(in, out)));
}

/**
 * \brief A dynamically sized dimension.
 *
 * The number of dimensions must be between [1, 9].
 */
class TEST_API DDim {
public:
  constexpr static int kMaxRank = 9;

  DDim();

  DDim(const DDim &ddim);

  DDim(const int *d, int n);

  DDim(const int64_t *d, int n);

  /*implicit*/ DDim(std::initializer_list<int64_t> init_list);

  template <int D>
  /*implicit*/ DDim(const Dim<D> &in) : rank_(D) { // NOLINT
    UnsafeCast<D>() = in;
  }

  inline DDim &operator=(const DDim &ddim) { return CopyFrom(ddim); }

  template <int D> inline DDim &operator=(const Dim<D> &dim) {
    rank_ = D;
    UnsafeCast<D>() = dim;
    return *this;
  }

  inline int64_t &operator[](int idx) { return dim_[idx]; }

  inline int64_t operator[](int idx) const { return dim_[idx]; }

  int64_t &at(int idx);

  int64_t at(int idx) const;

  template <typename Visitor>
  typename std::result_of<Visitor(Dim<0> &)>::type
  apply_visitor(Visitor &&visitor) {
    PADDLE_VISIT_DDIM(rank_, visitor(UnsafeCast<kRank>()));
  }

  template <typename Visitor>
  typename std::result_of<Visitor(const Dim<0> &)>::type
  apply_visitor(Visitor &&visitor) const {
    PADDLE_VISIT_DDIM(rank_, visitor(UnsafeCast<kRank>()));
  }

  bool operator==(const DDim &d) const;

  bool operator!=(const DDim &d) const;

  inline const int64_t *Get() const { return dim_.Get(); }

  inline int64_t *GetMutable() { return dim_.GetMutable(); }

  inline int size() const { return rank_; }

  std::string to_str() const;

  DDim reshape(std::vector<int> &shape) const; // NOLINT

  DDim transpose(const std::vector<int> &axis) const;

private:
  template <int D> inline Dim<D> &UnsafeCast() {
    static_assert(D >= 0 && D <= kMaxRank, "Invalid rank");
    auto *p = static_cast<void *>(&dim_);
    return *reinterpret_cast<Dim<D> *>(p);
  }

  template <int D> inline const Dim<D> &UnsafeCast() const {
    static_assert(D >= 0 && D <= kMaxRank, "Invalid rank");
    auto *p = static_cast<const void *>(&dim_);
    return *reinterpret_cast<const Dim<D> *>(p);
  }

  inline DDim &CopyFrom(const DDim &ddim) {
    if (ddim.rank_ == -1) {
      rank_ = -1;
      return *this;
    }
    PADDLE_VISIT_DDIM(ddim.rank_, (*this = ddim.UnsafeCast<kRank>()));
  }

  friend TEST_API DDim stride(const DDim &ddim);
  friend TEST_API DDim stride_numel(const DDim &ddim);

private:
  Dim<kMaxRank> dim_;
  int rank_;
};

#undef PADDLE_VISIT_DDIM_BASE
#undef PADDLE_VISIT_DDIM

/**
 * \brief Make a DDim from std::vector<int64_t>
 *
 * \param dims An vector of ints. Must be sized between [1, 9]
 */
TEST_API DDim make_ddim(const std::vector<int64_t> &dims);

TEST_API DDim make_ddim(const std::vector<int> &dims);

/**
 * \brief Make a DDim from an initializer list
 *
 * \param dims An initializer list of ints. Must be sized between [1, 9]
 *
 */
TEST_API DDim make_ddim(std::initializer_list<int64_t> dims);

template <typename T = int64_t> std::vector<T> vectorize(const DDim &ddim) {
  if (ddim.size() == -1) {
    return std::vector<T>({0});
  }
  std::vector<T> result(DDim::kMaxRank);
  dynamic_dim_assign(ddim.Get(), result.data(), ddim.size());
  result.resize(ddim.size());
  return result;
}

TEST_API int64_t product(const DDim &ddim);

TEST_API bool contain_unknown_dim(const DDim &ddim);

/**
 * \brief Slice a ddim
 *
 * Slice dim with [begin, end).
 * e.g.  DDim d = make_ddim({1,2,3,4,5});
 *       slice_ddim(d, 1, 3); ====> {2,3}
 */
TEST_API DDim slice_ddim(const DDim &dim, int begin, int end);

/**
 * \brief What is the length of this dimension?
 *
 * \param Dynamic dimension to inspect
 */

TEST_API int arity(const DDim &ddim);

TEST_API std::ostream &operator<<(std::ostream &, const DDim &);

/**
 * \brief Flatten dim to 3d
 * e.g., DDim d = make_ddim({1, 2, 3, 4, 5, 6})
 *       flatten_to_3d(d, 2, 4); ===> {1*2, 3*4, 5*6} ===> {2, 12, 30}
 */
TEST_API DDim flatten_to_3d(const DDim &src, int num_row_dims,
                            int num_col_dims);

// Reshape a tensor to a matrix. The matrix's first dimension(column length)
// will be the product of tensor's first `num_col_dims` dimensions.
TEST_API DDim flatten_to_2d(const DDim &src, int num_col_dims);

TEST_API DDim flatten_to_1d(const DDim &src);

TEST_API DDim stride(const DDim &ddim);

TEST_API DDim stride_numel(const DDim &ddim);

TEST_API bool AreDimsWithDynamicShapeCompatible(const DDim &dim1,
                                                const DDim &dim2);

TEST_API DDim ComputeCompatibleDim(const DDim &dim1, const DDim &dim2);

} // namespace common

namespace pir {
using DDim = common::DDim;
using LegacyLoD = std::vector<std::vector<size_t>>;
using LoD = LegacyLoD;
} // namespace pir

namespace std {
template <> struct TEST_API hash<common::DDim> {
  std::size_t operator()(common::DDim const &ddim) const;
};
} // namespace std
