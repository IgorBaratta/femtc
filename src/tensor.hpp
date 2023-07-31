/// @file
/// @brief This file contains classes for Tensor and Matrix, for multi-dimensional data storage.

#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <array>
#include <string>
#include <ostream>
#include <algorithm>

namespace linalg
{
    enum class Layout
    {
        RowMajor,
        ColMajor
    };

    enum class TensorLayout
    {
        ijk,
        ikj,
        jik,
        jki,
        kij,
        kji
    };

    constexpr TensorLayout default_layout = TensorLayout::ijk;

    // template <TensorLayout in, TensorLayout out, std::size_t Na, std::size_t Nb, std::size_t Nc, typename T>
    // void transpose(T *in_data, T *out_data)
    // {
    //     constexpr std::array<std::size_t, 3> offsets0 = get_offsets<in, Na, Nb, Nc>();
    //     constexpr std::array<std::size_t, 3> offsets1 = get_offsets<out, Na, Nb, Nc>();
    //     if constexpr (in == out)
    //         std::copy(in_data, in_data + Na * Nb * Nc, out_data);
    //     else if constexpr (in == Layout::ijk)
    //     {
    //         for (std::size_t i = 0; i < Na; ++i)
    //             for (std::size_t j = 0; j < Nb; ++j)
    //                 for (std::size_t k = 0; k < Nc; ++k)
    //                     out_data[offsets1[0] * i + offsets1[1] * j + offsets1[2] * k] = in_data[offsets0[0] * i + offsets0[1] * j + offsets0[2] * k];
    //     }
    //     else if constexpr (in == Layout::ikj)
    //     {
    //         for (std::size_t i = 0; i < Na; ++i)
    //             for (std::size_t j = 0; j < Nb; ++j)
    //                 for (std::size_t k = 0; k < Nc; ++k)
    //                     out_data[offsets1[0] * i + offsets1[1] * k + offsets1[2] * j] = in_data[offsets0[0] * i + offsets0[1] * j + offsets0[2] * k];
    //     }
    //     else if constexpr (in == Layout::jik)
    //     {
    //         for (std::size_t i = 0; i < Na; ++i)
    //             for (std::size_t j = 0; j < Nb; ++j)
    //                 for (std::size_t k = 0; k < Nc; ++k)
    //                     out_data[offsets1[0] * j + offsets1[1] * i + offsets1[2] * k] = in_data[offsets0[0] * i + offsets0[1] * j + offsets0[2] * k];
    //     }
    // }

    // /// @brief Returns the offsets for the three dimensions of a tensor.
    // /// @tparam layout The layout of the tensor.
    // /// @tparam Na The number of rows in the tensor.
    // /// @tparam Nb The number of columns in the tensor.
    // /// @tparam Nc The number of layers in the tensor.
    // template <TensorLayout layout, int Na, int Nb, int Nc>
    // constexpr std::array<std::size_t, 3> get_offsets()
    // {
    //     if constexpr (layout == TensorLayout::ijk)
    //     {
    //         return {Nb * Nc, Nc, 1};
    //     }
    //     else if constexpr (layout == TensorLayout::ikj)
    //     {
    //         return {Nb * Nc, 1, Nb};
    //     }
    //     else if constexpr (layout == TensorLayout::jik)
    //     {
    //         return {Nc, Na * Nc, 1};
    //     }
    //     else if constexpr (layout == TensorLayout::jki)
    //     {
    //         return {1, Na * Nc, Na};
    //     }
    //     else if constexpr (layout == TensorLayout::kij)
    //     {
    //         return {Nb, 1, Na * Nb};
    //     }
    //     else if constexpr (layout == TensorLayout::kji)
    //     {
    //         return {1, Na, Na * Nb};
    //     }
    // }

    // /// @brief  Three-dimensional tensor class.
    // /// @tparam T The data type of the tensor.
    // /// @tparam Na The number of rows in the tensor.
    // /// @tparam Nb The number of columns in the tensor.
    // /// @tparam Nc The number of layers in the tensor.
    // template <typename T, int Na, int Nb, int Nc, TensorLayout layout = default_layout>
    // class Tensor
    // {
    // public:
    //     ///< The data type of the tensor.
    //     using value_type = T;
    //     ///< The type of the tensor.
    //     using tensor_type = Tensor<T, Na, Nb, Nc, layout>;
    //     ///< The number of dimensions of the tensor.
    //     constexpr static int ndim = 3;
    //     ///< The total number of elements in the tensor.
    //     constexpr static int size = Na * Nb * Nc;
    //     ///< The offsets for the three dimensions of the tensor.
    //     constexpr static std::array<std::size_t, 3> offsets = get_offsets<layout, Na, Nb, Nc>();

    //     /// @brief Constructor for the Tensor class.
    //     /// @param pointer A pointer to the data to initialize the tensor with.
    //     Tensor(T *pointer) : data(pointer)
    //     {
    //     }

    //     /// @brief Fill the tensor with a specified value.
    //     /// @param val The value to fill the tensor with.
    //     void fill(T val)
    //     {
    //         for (int i = 0; i < size; ++i)
    //             data[i] = val;
    //     }

    //     /// @brief Fill the tensor with random values.
    //     void fill_random()
    //     {
    //         for (int i = 0; i < size; ++i)
    //             data[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    //     }

    //     /// @brief Returns the size of the tensor along a specified dimension.
    //     /// @tparam dim The dimension to return the size of.
    //     template <int dim>
    //     static constexpr std::size_t shape()
    //     {
    //         if constexpr (dim == 0)
    //             return Na;
    //         else if constexpr (dim == 1)
    //             return Nb;
    //         else if constexpr (dim == 2)
    //             return Nc;
    //         else
    //             return 0;
    //     }

    //     /// @brief Returns a pointer to the underlying data storage.
    //     /// @return A pointer to the underlying data storage.
    //     T *data_ptr()
    //     {
    //         return data;
    //     }

    //     /// @brief Returns a const pointer to the underlying data storage.
    //     /// @return A const pointer to the underlying data storage.
    //     const T *data_ptr() const
    //     {
    //         return data;
    //     }

    //     /// @brief Overloaded operator for accessing elements in the tensor.
    //     /// @param i The index of the element to access.
    //     /// @return A reference to the element at the specified index.
    //     inline constexpr T &operator[](std::size_t i)
    //     {
    //         return data[i];
    //     }

    //     /// @brief Overloaded operator for accessing elements in the tensor (const version).
    //     /// @param i The index of the element to access.
    //     /// @return A const reference to the element at the specified index.
    //     constexpr const T &operator[](std::size_t i) const
    //     {
    //         return data[i];
    //     }

    //     /// @brief Accesses elements in the tensor using row, column, and layer indices.
    //     /// @param i The row index of the element.
    //     /// @param j The column index of the element.
    //     /// @param k The layer index of the element.
    //     /// @return A reference to the element at the specified indices.
    //     inline constexpr T &operator()(std::size_t i, std::size_t j, std::size_t k)
    //     {
    //         return data[i * offsets[0] + j * offsets[1] + k * offsets[2]];
    //     }

    //     /// @brief Accesses elements in the tensor using row, column, and layer indices (const version).
    //     /// @param i The row index of the element.
    //     /// @param j The column index of the element.
    //     /// @param k The layer index of the element.
    //     /// @return A const reference to the element at the specified indices.
    //     inline constexpr const T &operator()(std::size_t i, std::size_t j, std::size_t k) const
    //     {
    //         return data[i * offsets[0] + j * offsets[1] + k * offsets[2]];
    //     }

    //     /// @brief Accesses elements in the tensor using row, and linear column and layer indices.
    //     /// @param i The row index of the element.
    //     /// @param j The linear column and layer index of the element.
    //     /// @return A reference to the element at the specified indices.
    //     constexpr T &operator()(std::size_t i, std::size_t j)
    //     {
    //         if constexpr (layout == TensorLayout::ijk)
    //             return data[i * offsets[0] + j];
    //         else if constexpr (layout == TensorLayout::ikj)
    //             return data[i * offsets[0] + j];
    //         else if constexpr (layout == TensorLayout::jik)
    //             return data[i * offsets[1] + j];
    //         else if constexpr (layout == TensorLayout::jki)
    //             return data[i * offsets[1] + j];
    //         else if constexpr (layout == TensorLayout::kij)
    //             return data[i * offsets[2] + j];
    //         else if constexpr (layout == TensorLayout::kji)
    //             return data[i * offsets[2] + j];
    //     }

    //     /// @brief Accesses elements in the tensor such the first index represents the row, and the
    //     /// second index represents the linear of the two remaining dimensions. E.g. for a tensor with
    //     /// layout ijk, the first index represents the row, and the second index represents the linear
    //     /// column and layer indices. For a tensor with layout ikj, the first index represents the row,
    //     /// and the second index represents the linear layer and column indices.
    //     /// @param i The row index of the element.
    //     /// @param j The linear column and layer index of the element.
    //     /// @return A const reference to the element at the specified indices.
    //     inline constexpr const T &operator()(std::size_t i, std::size_t j) const
    //     {
    //         if constexpr (layout == TensorLayout::ijk)
    //             return data[i * offsets[0] + j];
    //         else if constexpr (layout == TensorLayout::ikj)
    //             return data[i * offsets[0] + j];
    //         else if constexpr (layout == TensorLayout::jik)
    //             return data[i * offsets[1] + j];
    //         else if constexpr (layout == TensorLayout::jki)
    //             return data[i * offsets[1] + j];
    //         else if constexpr (layout == TensorLayout::kij)
    //             return data[i * offsets[2] + j];
    //         else if constexpr (layout == TensorLayout::kji)
    //             return data[i * offsets[2] + j];
    //     }

    //     /// @brief Return a new tensor with the same data but a different layout.
    //     /// @tparam new_layout The layout of the new tensor.
    //     /// @return A new tensor with the same data but a different layout.
    //     template <TensorLayout new_layout>
    //     inline constexpr void to_layout(Tensor<T, Na, Nb, Nc, new_layout> &other)
    //     {

    //         if constexpr (layout == new_layout)
    //         {
    //             std::copy_n(data, size, other.data);
    //         }

    //         else if constexpr (layout == TensorLayout::ijk) // ijk to other
    //         {
    //             for (std::size_t i = 0; i < Na; ++i)
    //                 for (std::size_t j = 0; j < Nb; ++j)
    //                     for (std::size_t k = 0; k < Nc; ++k)
    //                         other(i, j, k) = (*this)(i, j, k);
    //         }
    //         else if constexpr (layout == TensorLayout::ikj) // ikj to other
    //         {
    //             for (std::size_t i = 0; i < Na; ++i)
    //                 for (std::size_t k = 0; k < Nc; ++k)
    //                     for (std::size_t j = 0; j < Nb; ++j)
    //                         other(i, j, k) = (*this)(i, j, k);
    //         }
    //         else if constexpr (layout == TensorLayout::kij) // kij to other
    //         {
    //             for (std::size_t k = 0; k < Nc; ++k)
    //                 for (std::size_t i = 0; i < Na; ++i)
    //                     for (std::size_t j = 0; j < Nb; ++j)
    //                         other(i, j, k) = (*this)(i, j, k);
    //         }
    //         else if constexpr (layout == TensorLayout::kji) // kji to other
    //         {
    //             for (std::size_t k = 0; k < Nc; ++k)
    //                 for (std::size_t j = 0; j < Nb; ++j)
    //                     for (std::size_t i = 0; i < Na; ++i)
    //                         other(i, j, k) = (*this)(i, j, k);
    //         }
    //         else if constexpr (layout == TensorLayout::jik) // jik to other
    //         {
    //             for (std::size_t j = 0; j < Nb; ++j)
    //                 for (std::size_t i = 0; i < Na; ++i)
    //                     for (std::size_t k = 0; k < Nc; ++k)
    //                         other(i, j, k) = (*this)(i, j, k);
    //         }
    //         else if constexpr (layout == TensorLayout::jki) // jki to other
    //         {
    //             for (std::size_t j = 0; j < Nb; ++j)
    //                 for (std::size_t k = 0; k < Nc; ++k)
    //                     for (std::size_t i = 0; i < Na; ++i)
    //                         other(i, j, k) = (*this)(i, j, k);
    //         }
    //     }

    // private:
    //     T *data; ///< The underlying data storage for the tensor.
    // };

    // /// @brief Class for storing a matrix of shape (N0, N1).
    // /// @tparam T The data type of the matrix.
    // /// @tparam N0 The number of rows in the matrix.
    // /// @tparam N1 The number of columns in the matrix.
    // template <typename T, int N0, int N1, Layout layout = Layout::RowMajor>
    // class Matrix
    // {
    // public:
    //     using value_type = T; ///< The data type of the matrix.
    //     using size_type = std::size_t;
    //     using difference_type = std::ptrdiff_t;
    //     using pointer = value_type *;
    //     using const_pointer = const value_type *;
    //     using reference = value_type &;
    //     using const_reference = const value_type &;
    //     using iterator = pointer;
    //     using reverse_iterator = std::reverse_iterator<iterator>;

    //     /// @brief Constructor for the Matrix class.
    //     /// @param pointer A pointer to the data to initialize the tensor with.
    //     constexpr Matrix(T *pointer) noexcept : data(pointer)
    //     {
    //     }

    //     /// @brief Fill the matrix with a specified value.
    //     /// @param val The value to fill the matrix with.
    //     void fill(T val)
    //     {
    //         for (int i = 0; i < N0 * N1; ++i)
    //             data[i] = val;
    //     }

    //     /// @brief Fill the matrix with random values.
    //     void fill_random()
    //     {
    //         for (int i = 0; i < N0 * N1; ++i)
    //             data[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    //     }

    //     /// @brief Returns a pointer to the underlying data storage.
    //     /// @return A pointer to the underlying data storage.
    //     T *data_ptr()
    //     {
    //         return data;
    //     }

    //     /// @brief Returns a const pointer to the underlying data storage.
    //     /// @return A const pointer to the underlying data storage.
    //     const T *data_ptr() const
    //     {
    //         return data;
    //     }

    //     /// @brief Returns the total number of elements in the matrix.
    //     /// @return The total number of elements in the matrix.
    //     static constexpr std::size_t size()
    //     {
    //         return N0 * N1;
    //     }

    //     /// @brief Returns the size of the matrix along a specified dimension.
    //     /// @tparam dim The dimension to return the size of.
    //     template <int dim>
    //     static constexpr std::size_t size()
    //     {
    //         if constexpr (dim == 0)
    //             return N0;
    //         else if constexpr (dim == 1)
    //             return N1;
    //         else
    //             return 0;
    //     }

    //     /// @brief Overloaded operator for accessing elements in the matrix.
    //     /// @param i The index of the element to access.
    //     /// @return A reference to the element at the specified index.
    //     constexpr T &operator[](std::size_t i)
    //     {
    //         return data[i];
    //     }

    //     /// @brief Overloaded operator for accessing elements in the matrix (const version).
    //     /// @param i The index of the element to access.
    //     /// @return A const reference to the element at the specified index.
    //     constexpr const T &operator[](std::size_t i) const
    //     {
    //         return data[i];
    //     }

    //     /// @brief Accesses elements in the matrix using row and column indices.
    //     /// @param i The row index of the element.
    //     /// @param j The column index of the element.
    //     /// @return A reference to the element at the specified indices.
    //     constexpr T &operator()(std::size_t i, std::size_t j)
    //     {
    //         if constexpr (layout == Layout::RowMajor)
    //             return data[i * N1 + j];
    //         else
    //             return data[j * N0 + i];
    //     }

    //     /// @brief Accesses elements in the matrix using row and column indices (const version).
    //     /// @param i The row index of the element.
    //     /// @param j The column index of the element.
    //     /// @return A const reference to the element at the specified indices.
    //     constexpr const T &operator()(std::size_t i, std::size_t j) const
    //     {
    //         if constexpr (layout == Layout::RowMajor)
    //             return data[i * N1 + j];
    //         else
    //             return data[j * N0 + i];
    //     }

    //     /// overload the << operator to print the matrix
    //     friend std::ostream &operator<<(std::ostream &os, const Matrix<T, N0, N1, layout> &m)
    //     {
    //         for (int i = 0; i < N0; ++i)
    //         {
    //             for (int j = 0; j < N1; ++j)
    //                 os << m(i, j) << " ";
    //             os << std::endl;
    //         }
    //         return os;
    //     }

    //     /// @brief Returns the transpose of the matrix.
    //     /// @return The transposed matrix.
    //     auto transpose() const
    //     {
    //         Matrix<T, N1, N0, layout> B;
    //         for (int i = 0; i < N0; ++i)
    //             for (int j = 0; j < N1; ++j)
    //                 B(j, i) = (*this)(i, j);
    //         return B;
    //     }

    // private:
    //     T *data = nullptr; ///< The underlying data storage for the matrix.
    // };

} // namespace linalg

#endif // DATA_STRUCTURES_H