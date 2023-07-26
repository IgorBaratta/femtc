/// @file
/// @brief This file contains classes for Tensor and Matrix, for multi-dimensional data storage.

#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H

#include <array>

enum class Layout
{
    RowMajor,
    ColMajor
};

/// @brief  Three-dimensional tensor class.
/// @tparam T The data type of the tensor.
/// @tparam Na The number of rows in the tensor.
/// @tparam Nb The number of columns in the tensor.
/// @tparam Nc The number of layers in the tensor.
template <typename T, int Na, int Nb, int Nc>
class Tensor
{
public:
    using value_type = T; ///< The data type of the tensor.

    /// @brief Constructor for the Tensor class.
    /// @param val The default value to initialize all elements of the tensor (default is 0).
    Tensor(T val = 0)
    {
        data.fill(val);
    }

    /// @brief Fill the tensor with random values.
    void fill_random()
    {
        for (int i = 0; i < Na * Nb * Nc; ++i)
            data[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }

    /// @brief Returns the total number of elements in the tensor.
    /// @return The total number of elements in the tensor.
    constexpr std::size_t size() const
    {
        return Na * Nb * Nc;
    }

    /// @brief Returns the size of the tensor along a specified dimension.
    /// @tparam dim The dimension to return the size of.
    template <int dim>
    static constexpr std::size_t size()
    {
        if constexpr (dim == 0)
            return Na;
        else if constexpr (dim == 1)
            return Nb;
        else if constexpr (dim == 2)
            return Nc;
        else
            return 0;
    }

    /// @brief Overloaded operator for accessing elements in the tensor.
    /// @param i The index of the element to access.
    /// @return A reference to the element at the specified index.
    constexpr T &
    operator[](std::size_t i)
    {
        return data[i];
    }

    /// @brief Overloaded operator for accessing elements in the tensor (const version).
    /// @param i The index of the element to access.
    /// @return A const reference to the element at the specified index.
    constexpr const T &operator[](std::size_t i) const
    {
        return data[i];
    }

    /// @brief Accesses elements in the tensor using row, and linear column and layer indices.
    /// @param i The row index of the element.
    /// @param j The linear column and layer index of the element.
    /// @return A reference to the element at the specified indices.
    constexpr T &operator()(std::size_t i, std::size_t j)
    {
        return data[i * Nb * Nc + j];
    }

    /// @brief Accesses elements in the tensor using row, column, and layer indices.
    /// @param i The row index of the element.
    /// @param j The column index of the element.
    /// @param k The layer index of the element.
    /// @return A reference to the element at the specified indices.
    inline constexpr T &operator()(std::size_t i, std::size_t j, std::size_t k)
    {
        return data[i * Nb * Nc + j * Nc + k];
    }

    /// @brief Accesses elements in the tensor using row, column, and layer indices (const version).
    /// @param i The row index of the element.
    /// @param j The column index of the element.
    /// @param k The layer index of the element.
    /// @return A const reference to the element at the specified indices.
    inline constexpr const T &operator()(std::size_t i, std::size_t j, std::size_t k) const
    {
        return data[i * Nb * Nc + j * Nc + k];
    }

    /// @brief Accesses elements in the tensor using row, and linear column and layer indices (const version).
    /// @param i The row index of the element.
    /// @param j The linear column and layer index of the element.
    /// @return A const reference to the element at the specified indices.
    inline constexpr const T &operator()(std::size_t i, std::size_t j) const
    {
        return data[i * Nb * Nc + j];
    }

private:
    std::array<T, Na * Nb * Nc> data; ///< The underlying data storage for the tensor.
};

/// @brief Class for storing a matrix of shape (N0, N1).
/// @tparam T The data type of the matrix.
/// @tparam N0 The number of rows in the matrix.
/// @tparam N1 The number of columns in the matrix.
template <typename T, int N0, int N1, Layout layout = Layout::RowMajor>
class Matrix
{
public:
    using value_type = T; ///< The data type of the matrix.

    /// @brief Constructor for the Matrix class.
    /// @param val The default value to initialize all elements of the matrix (default is 0).
    Matrix(T val = 0)
    {
        data.fill(val);
    }

    /// @brief Fill the matrix with random values.
    void fill_random()
    {
        for (int i = 0; i < N0 * N1; ++i)
            data[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }

    /// @brief Returns a pointer to the underlying data storage.
    /// @return A pointer to the underlying data storage.
    T *data_ptr()
    {
        return data.data();
    }

    /// @brief Returns a const pointer to the underlying data storage.
    /// @return A const pointer to the underlying data storage.
    const T *data_ptr() const
    {
        return data.data();
    }

    /// @brief Returns the total number of elements in the matrix.
    /// @return The total number of elements in the matrix.
    static constexpr std::size_t size()
    {
        return N0 * N1;
    }

    /// @brief Returns the size of the matrix along a specified dimension.
    /// @tparam dim The dimension to return the size of.
    template <int dim>
    static constexpr std::size_t size()
    {
        if constexpr (dim == 0)
            return N0;
        else if constexpr (dim == 1)
            return N1;
        else
            return 0;
    }

    /// @brief Overloaded operator for accessing elements in the matrix.
    /// @param i The index of the element to access.
    /// @return A reference to the element at the specified index.
    constexpr T &operator[](std::size_t i)
    {
        return data[i];
    }

    /// @brief Overloaded operator for accessing elements in the matrix (const version).
    /// @param i The index of the element to access.
    /// @return A const reference to the element at the specified index.
    constexpr const T &operator[](std::size_t i) const
    {
        return data[i];
    }

    /// @brief Accesses elements in the matrix using row and column indices.
    /// @param i The row index of the element.
    /// @param j The column index of the element.
    /// @return A reference to the element at the specified indices.
    constexpr T &operator()(std::size_t i, std::size_t j)
    {
        if constexpr (layout == Layout::RowMajor)
            return data[i * N1 + j];
        else
            return data[j * N0 + i];
    }

    /// @brief Accesses elements in the matrix using row and column indices (const version).
    /// @param i The row index of the element.
    /// @param j The column index of the element.
    /// @return A const reference to the element at the specified indices.
    constexpr const T &operator()(std::size_t i, std::size_t j) const
    {
        if constexpr (layout == Layout::RowMajor)
            return data[i * N1 + j];
        else
            return data[j * N0 + i];
    }

    /// @brief Returns the transpose of the matrix.
    /// @return The transposed matrix.
    auto transpose() const
    {
        Matrix<T, N1, N0, layout> B;
        for (int i = 0; i < N0; ++i)
            for (int j = 0; j < N1; ++j)
                B(j, i) = (*this)(i, j);
        return B;
    }

private:
    std::array<T, N0 * N1> data; ///< The underlying data storage for the matrix.
};

#endif // DATA_STRUCTURES_H