// Include guard
#ifndef TENSOR_H
#define TENSOR_H

// Includes
#include <stddef.h>

// Tensor data type
typedef double tensor_dtype;

// Tensor definition
typedef struct
{
  tensor_dtype* data;
  size_t* shape;
  size_t num_dims;
  size_t num_elements;
} Tensor;

// Create a new tensor
Tensor*
tensor_create(const size_t* shape, size_t num_dims);

// Free a tensor's memory
void
tensor_free(Tensor* tensor);

// Get index of element in tensor
size_t
tensor_get_index(const Tensor* tensor, const size_t* indices);

// Get value of element in tensor
tensor_dtype
tensor_get_value(const Tensor* tensor, const size_t* indices);

// Set value of element in tensor
tensor_dtype
tensor_set_value(Tensor* tensor, const size_t* indices, tensor_dtype value);

// Pretty print tensor
void
tensor_print(const Tensor* tensor);

// Check if two tensors are the same shape
int
tensor_same_shape(const Tensor* tensor1, const Tensor* tensor2);

// Check if two tensors are equal
int
tensor_equal(const Tensor* tensor1, const Tensor* tensor2);

// Compute the dot product of two tensors
Tensor*
tensor_dot(const Tensor* tensor1, const Tensor* tensor2);

// Compute the element-wise sum of two tensors
Tensor*
tensor_add(const Tensor* tensor1, const Tensor* tensor2);

// Compute the element-wise difference of two tensors
Tensor*
tensor_subtract(const Tensor* tensor1, const Tensor* tensor2);

// Compute the element-wise product of two tensors
Tensor*
tensor_multiply(const Tensor* tensor1, const Tensor* tensor2);

// Compute the element-wise division of two tensors
Tensor*
tensor_divide(const Tensor* tensor1, const Tensor* tensor2);

// Compute the element-wise power of a tensor
Tensor*
tensor_power(const Tensor* tensor, tensor_dtype power);

// Compute the element-wise square root of a tensor
Tensor*
tensor_sqrt(const Tensor* tensor);

// Compute the element-wise exponential of a tensor
Tensor*
tensor_exp(const Tensor* tensor);

// Compute the element-wise natural logarithm of a tensor
Tensor*
tensor_log(const Tensor* tensor);

// Compute the element-wise sine of a tensor
Tensor*
tensor_sin(const Tensor* tensor);

// Compute the element-wise cosine of a tensor
Tensor*
tensor_cos(const Tensor* tensor);

// Compute the element-wise tangent of a tensor
Tensor*
tensor_tan(const Tensor* tensor);

// Compute the multiplication of a tensor by a scalar
Tensor*
tensor_scalar_multiply(const Tensor* tensor, tensor_dtype scalar);

// Compute the division of a tensor by a scalar
Tensor*
tensor_scalar_divide(const Tensor* tensor, tensor_dtype scalar);

// Compute the power of a tensor by a scalar
Tensor*
tensor_scalar_power(const Tensor* tensor, tensor_dtype scalar);

// Compute the matrix multiplication of two tensors
Tensor*
tensor_matmul(const Tensor* tensor1, const Tensor* tensor2);

// Compute the transpose of a tensor
Tensor*
tensor_transpose(const Tensor* tensor);

// Compute the sum of a tensor along a given axis
Tensor*
tensor_sum(const Tensor* tensor, size_t axis);

// Compute the mean of a tensor along a given axis
Tensor*
tensor_mean(const Tensor* tensor, size_t axis);

// Compute the maximum of a tensor along a given axis
Tensor*
tensor_max(const Tensor* tensor, size_t axis);

// Compute the minimum of a tensor along a given axis
Tensor*
tensor_min(const Tensor* tensor, size_t axis);

// Compute the mode of a tensor along a given axis
Tensor*
tensor_mode(const Tensor* tensor, size_t axis);

// End of include guard
#endif