#include "tensor.hpp"

#define restrict __restrict__

namespace linalg
{

/// Get offset for the first index
/// @param[in] N0 first dimension
/// @param[in] N1 second dimension
/// @param[in] N2 third dimension
template <int N0, int N1, int N2, TensorLayout layout>
constexpr int get_offset0()
{
  if constexpr (layout == TensorLayout::ijk)
    return N1 * N2;
  else if constexpr (layout == TensorLayout::ikj)
    return N1 * N2;
  else if constexpr (layout == TensorLayout::jik)
    return N2;
  else if constexpr (layout == TensorLayout::jki)
    return 1;
  else if constexpr (layout == TensorLayout::kij)
    return N1;
  else if constexpr (layout == TensorLayout::kji)
    return 1;
}

/// Get offset for the second index
/// @param[in] N0 first dimension
/// @param[in] N1 second dimension
/// @param[in] N2 third dimension
/// @param[in] layout tensor layout
template <int N0, int N1, int N2, TensorLayout layout>
constexpr int get_offset1()
{
  if constexpr (layout == TensorLayout::ijk)
    return N2;
  else if constexpr (layout == TensorLayout::ikj)
    return 1;
  else if constexpr (layout == TensorLayout::jik)
    return N0 * N2;
  else if constexpr (layout == TensorLayout::jki)
    return N0 * N2;
  else if constexpr (layout == TensorLayout::kij)
    return 1;
  else if constexpr (layout == TensorLayout::kji)
    return N0;
}

/// Get offset for the third index
/// @param[in] N0 first dimension
/// @param[in] N1 second dimension
/// @param[in] N2 third dimension
/// @param[in] layout tensor layout
template <int N0, int N1, int N2, TensorLayout layout>
constexpr int get_offset2()
{
  if constexpr (layout == TensorLayout::ijk)
    return 1;
  else if constexpr (layout == TensorLayout::ikj)
    return N1;
  else if constexpr (layout == TensorLayout::jik)
    return 1;
  else if constexpr (layout == TensorLayout::jki)
    return N0;
  else if constexpr (layout == TensorLayout::kij)
    return N0 * N1;
  else if constexpr (layout == TensorLayout::kji)
    return N0 * N1;
}

/// Compute tensor permutation B[P(I)] <- A[I] where P is a permutation.
/// @param[in] A tensor of shape (N0, N1, N2) store using lin layout.
/// @param[out] B tensor of shape (N0, N1, N2) stored in P(I) order,
/// lout layout.
/// Example: permute<double, 2, 3, 4, TensorLayout::ijk,
/// TensorLayout::kij >(A, B);
template <typename T, int N0, int N1, int N2, TensorLayout lin,
          TensorLayout lout>
void permute(const T* in, T* out)
{
  constexpr int offin0 = get_offset0<N0, N1, N2, lin>();
  constexpr int offin1 = get_offset1<N0, N1, N2, lin>();
  constexpr int offin2 = get_offset2<N0, N1, N2, lin>();

  constexpr int offout0 = get_offset0<N0, N1, N2, lout>();
  constexpr int offout1 = get_offset1<N0, N1, N2, lout>();
  constexpr int offout2 = get_offset2<N0, N1, N2, lout>();

  if constexpr (lin == lout)
  {
    constexpr int N = N0 * N1 * N2;
    for (int i = 0; i < N; i++)
      out[i] = in[i];
  }
  else if constexpr (lin == TensorLayout::ijk)
  {
    for (int i = 0; i < N0; i++)
      for (int j = 0; j < N1; j++)
        for (int k = 0; k < N2; k++)
          out[k * offout2 + j * offout1 + i * offout0]
              = in[i * offin0 + j * offin1 + k * offin2];
  }
  else if constexpr (lin == TensorLayout::ikj)
  {
    for (int i = 0; i < N0; i++)
      for (int k = 0; k < N2; k++)
        for (int j = 0; j < N1; j++)
          out[k * offout2 + j * offout1 + i * offout0]
              = in[i * offin0 + j * offin1 + k * offin2];
  }
  else if constexpr (lin == TensorLayout::jik)
  {
    for (int j = 0; j < N1; j++)
      for (int i = 0; i < N0; i++)
        for (int k = 0; k < N2; k++)
          out[k * offout2 + j * offout1 + i * offout0]
              = in[i * offin0 + j * offin1 + k * offin2];
  }
  else if constexpr (lin == TensorLayout::jki)
  {
    for (int j = 0; j < N1; j++)
      for (int k = 0; k < N2; k++)
        for (int i = 0; i < N0; i++)
          out[k * offout2 + j * offout1 + i * offout0]
              = in[i * offin0 + j * offin1 + k * offin2];
  }
  else if constexpr (lin == TensorLayout::kij)
  {
    for (int k = 0; k < N2; k++)
      for (int i = 0; i < N0; i++)
        for (int j = 0; j < N1; j++)
          out[k * offout2 + j * offout1 + i * offout0]
              = in[i * offin0 + j * offin1 + k * offin2];
  }
  else if constexpr (lin == TensorLayout::kji)
  {
    for (int k = 0; k < N2; k++)
      for (int j = 0; j < N1; j++)
        for (int i = 0; i < N0; i++)
          out[k * offout2 + j * offout1 + i * offout0]
              = in[i * offin0 + j * offin1 + k * offin2];
  }
}

} // namespace linalg
