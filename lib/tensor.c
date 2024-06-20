// Includes
#include "../includes/tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Creates a new tensor with the given shape and number of dimensions.
 *
 * @param shape The shape of the tensor.
 * @param num_dims The number of dimensions of the tensor.
 * @return A pointer to the newly created tensor, or NULL if memory allocation fails.
 */
Tensor* tensor_create(const size_t* shape, size_t num_dims)
{
  // Allocate memory for tensor
  Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
  if (tensor == NULL) {
    fprintf(stderr, "Error: Unable to allocate memory for tensor\n");
    return NULL;
  }

  // Set tensor shape
  tensor->shape = (size_t*)malloc(num_dims * sizeof(size_t));
  if (tensor->shape == NULL) {
    fprintf(stderr, "Error: Unable to allocate memory for tensor shape\n");
    free(tensor);
    return NULL;
  }
  for (size_t i = 0; i < num_dims; i++) {
    tensor->shape[i] = shape[i];
  }

  // Set tensor number of dimensions
  tensor->num_dims = num_dims;

  // Compute tensor number of elements
  tensor->num_elements = 1;
  for (size_t i = 0; i < num_dims; i++) {
    tensor->num_elements *= shape[i];
  }

  // Allocate memory for tensor data
  tensor->data =
    (tensor_dtype*)malloc(tensor->num_elements * sizeof(tensor_dtype));
  if (tensor->data == NULL) {
    fprintf(stderr, "Error: Unable to allocate memory for tensor data\n");
    free(tensor->shape);
    free(tensor);
    return NULL;
  }

  // Return tensor
  return tensor;
}

// Free a tensor's memory
void
tensor_free(Tensor* tensor)
{
  free(tensor->data);
  free(tensor->shape);
  free(tensor);
}

// Get index of element in tensor
size_t
tensor_get_index(const Tensor* tensor, const size_t* indices)
{
  size_t index = 0;
  size_t stride = 1;
  for (size_t i = 0; i < tensor->num_dims; i++) {
    index += indices[i] * stride;
    stride *= tensor->shape[i];
  }
  return index;
}

// Get value of element in tensor
tensor_dtype
tensor_get_value(const Tensor* tensor, const size_t* indices)
{
  return tensor->data[tensor_get_index(tensor, indices)];
}

// Set value of element in tensor
tensor_dtype
tensor_set_value(Tensor* tensor, const size_t* indices, tensor_dtype value)
{
  size_t index = tensor_get_index(tensor, indices);
  tensor->data[index] = value;
  return value;
}

// Pretty print tensor
void
tensor_print(const Tensor* tensor)
{
  // Print tensor shape
  printf("Tensor shape: (");
  for (size_t i = 0; i < tensor->num_dims; i++) {
    printf("%zu", tensor->shape[i]);
    if (i < tensor->num_dims - 1) {
      printf(", ");
    }
  }
  printf(")\n");

  // Print tensor data
  printf("Tensor data:\n");
  size_t* indices = (size_t*)malloc(tensor->num_dims * sizeof(size_t));
  for (size_t i = 0; i < tensor->num_elements; i++) {
    size_t remainder = i;
    for (size_t j = 0; j < tensor->num_dims; j++) {
      indices[j] = remainder % tensor->shape[j];
      remainder /= tensor->shape[j];
    }
    printf("(");
    for (size_t j = 0; j < tensor->num_dims; j++) {
      printf("%zu", indices[j]);
      if (j < tensor->num_dims - 1) {
        printf(", ");
      }
    }
    printf("): %f\n", tensor_get_value(tensor, indices));
  }
  free(indices);
}

// Check if two tensors are the same shape
int
tensor_same_shape(const Tensor* tensor1, const Tensor* tensor2)
{
  if (tensor1->num_dims != tensor2->num_dims) {
    return 0;
  }
  for (size_t i = 0; i < tensor1->num_dims; i++) {
    if (tensor1->shape[i] != tensor2->shape[i]) {
      return 0;
    }
  }
  return 1;
}

// Check if two tensors are equal
int
tensor_equal(const Tensor* tensor1, const Tensor* tensor2)
{
  if (!tensor_same_shape(tensor1, tensor2)) {
    return 0;
  }
  for (size_t i = 0; i < tensor1->num_elements; i++) {
    if (tensor1->data[i] != tensor2->data[i]) {
      return 0;
    }
  }
  return 1;
}

// Compute the dot product of two tensors
Tensor*
tensor_dot(const Tensor* tensor1, const Tensor* tensor2)
{
  // Check if tensors are compatible for dot product
  if (tensor1->num_dims != 2 || tensor2->num_dims != 2 ||
      tensor1->shape[1] != tensor2->shape[0]) {
    fprintf(stderr, "Error: Tensors are not compatible for dot product\n");
    return NULL;
  }

  // Create new tensor for dot product
  size_t shape[2] = { tensor1->shape[0], tensor2->shape[1] };
  Tensor* tensor = tensor_create(shape, 2);
  if (tensor == NULL) {
    return NULL;
  }

  // Compute dot product
  for (size_t i = 0; i < tensor1->shape[0]; i++) {
    for (size_t j = 0; j < tensor2->shape[1]; j++) {
      tensor_dtype value = 0;
      for (size_t k = 0; k < tensor1->shape[1]; k++) {
        value += tensor_get_value(tensor1, (size_t[]){ i, k }) *
                 tensor_get_value(tensor2, (size_t[]){ k, j });
      }
      tensor_set_value(tensor, (size_t[]){ i, j }, value);
    }
  }

  // Return dot product
  return tensor;
}

// Compute the element-wise sum of two tensors
Tensor*
tensor_add(const Tensor* tensor1, const Tensor* tensor2)
{
  // Check if tensors are compatible for element-wise sum
  if (!tensor_same_shape(tensor1, tensor2)) {
    fprintf(stderr, "Error: Tensors are not compatible for element-wise sum\n");
    return NULL;
  }

  // Create new tensor for element-wise sum
  Tensor* tensor = tensor_create(tensor1->shape, tensor1->num_dims);
  if (tensor == NULL) {
    return NULL;
  }

  // Compute element-wise sum
  for (size_t i = 0; i < tensor1->num_elements; i++) {
    tensor->data[i] = tensor1->data[i] + tensor2->data[i];
  }

  // Return element-wise sum
  return tensor;
}

// Compute the element-wise difference of two tensors
Tensor*
tensor_subtract(const Tensor* tensor1, const Tensor* tensor2)
{
  // Check if tensors are compatible for element-wise difference
  if (!tensor_same_shape(tensor1, tensor2)) {
    fprintf(stderr,
            "Error: Tensors are not compatible for element-wise difference\n");
    return NULL;
  }

  // Create new tensor for element-wise difference
  Tensor* tensor = tensor_create(tensor1->shape, tensor1->num_dims);
  if (tensor == NULL) {
    return NULL;
  }

  // Compute element-wise difference
  for (size_t i = 0; i < tensor1->num_elements; i++) {
    tensor->data[i] = tensor1->data[i] - tensor2->data[i];
  }

  // Return element-wise difference
  return tensor;
}

// Compute the element-wise product of two tensors
Tensor*
tensor_multiply(const Tensor* tensor1, const Tensor* tensor2)
{
  // Check if tensors are compatible for element-wise product
  if (!tensor_same_shape(tensor1, tensor2)) {
    fprintf(stderr,
            "Error: Tensors are not compatible for element-wise product\n");
    return NULL;
  }

  // Create new tensor for element-wise product
  Tensor* tensor = tensor_create(tensor1->shape, tensor1->num_dims);
  if (tensor == NULL) {
    return NULL;
  }

  // Compute element-wise product
  for (size_t i = 0; i < tensor1->num_elements; i++) {
    tensor->data[i] = tensor1->data[i] * tensor2->data[i];
  }

  // Return element-wise product
  return tensor;
}

// Compute the element-wise division of two tensors
Tensor*
tensor_divide(const Tensor* tensor1, const Tensor* tensor2)
{
  // Check if tensors are compatible for element-wise division
  if (!tensor_same_shape(tensor1, tensor2)) {
    fprintf(stderr,
            "Error: Tensors are not compatible for element-wise division\n");
    return NULL;
  }

  // Create new tensor for element-wise division
  Tensor* tensor = tensor_create(tensor1->shape, tensor1->num_dims);
  if (tensor == NULL) {
    return NULL;
  }

  // Compute element-wise division
  for (size_t i = 0; i < tensor1->num_elements; i++) {
    tensor->data[i] = tensor1->data[i] / tensor2->data[i];
  }

  // Return element-wise division
  return tensor;
}

// Compute the element-wise power of a tensor
Tensor*
tensor_power(const Tensor* tensor, tensor_dtype power)
{
  // Create new tensor for element-wise power
  Tensor* result = tensor_create(tensor->shape, tensor->num_dims);
  if (result == NULL) {
    return NULL;
  }

  // Compute element-wise power
  for (size_t i = 0; i < tensor->num_elements; i++) {
    result->data[i] = pow(tensor->data[i], power);
  }

  // Return element-wise power
  return result;
}

// Compute the element-wise square root of a tensor
Tensor*
tensor_sqrt(const Tensor* tensor)
{
  // Create new tensor for element-wise square root
  Tensor* result = tensor_create(tensor->shape, tensor->num_dims);
  if (result == NULL) {
    return NULL;
  }

  // Compute element-wise square root
  for (size_t i = 0; i < tensor->num_elements; i++) {
    result->data[i] = sqrt(tensor->data[i]);
  }

  // Return element-wise square root
  return result;
}

// Compute the element-wise exponential of a tensor
Tensor*
tensor_exp(const Tensor* tensor)
{
  // Create new tensor for element-wise exponential
  Tensor* result = tensor_create(tensor->shape, tensor->num_dims);
  if (result == NULL) {
    return NULL;
  }

  // Compute element-wise exponential
  for (size_t i = 0; i < tensor->num_elements; i++) {
    result->data[i] = exp(tensor->data[i]);
  }

  // Return element-wise exponential
  return result;
}

// Compute the element-wise natural logarithm of a tensor
Tensor*
tensor_log(const Tensor* tensor)
{
  // Create new tensor for element-wise natural logarithm
  Tensor* result = tensor_create(tensor->shape, tensor->num_dims);
  if (result == NULL) {
    return NULL;
  }

  // Compute element-wise natural logarithm
  for (size_t i = 0; i < tensor->num_elements; i++) {
    result->data[i] = log(tensor->data[i]);
  }

  // Return element-wise natural logarithm
  return result;
}

// Compute the element-wise sine of a tensor
Tensor*
tensor_sin(const Tensor* tensor)
{
  // Create new tensor for element-wise sine
  Tensor* result = tensor_create(tensor->shape, tensor->num_dims);
  if (result == NULL) {
    return NULL;
  }

  // Compute element-wise sine
  for (size_t i = 0; i < tensor->num_elements; i++) {
    result->data[i] = sin(tensor->data[i]);
  }

  // Return element-wise sine
  return result;
}

// Compute the element-wise cosine of a tensor
Tensor*
tensor_cos(const Tensor* tensor)
{
  // Create new tensor for element-wise cosine
  Tensor* result = tensor_create(tensor->shape, tensor->num_dims);
  if (result == NULL) {
    return NULL;
  }

  // Compute element-wise cosine
  for (size_t i = 0; i < tensor->num_elements; i++) {
    result->data[i] = cos(tensor->data[i]);
  }

  // Return element-wise cosine
  return result;
}

// Compute the element-wise tangent of a tensor
Tensor*
tensor_tan(const Tensor* tensor)
{
  // Create new tensor for element-wise tangent
  Tensor* result = tensor_create(tensor->shape, tensor->num_dims);
  if (result == NULL) {
    return NULL;
  }

  // Compute element-wise tangent
  for (size_t i = 0; i < tensor->num_elements; i++) {
    result->data[i] = tan(tensor->data[i]);
  }

  // Return element-wise tangent
  return result;
}

// Compute the multiplication of a tensor by a scalar
Tensor*
tensor_scalar_multiply(const Tensor* tensor, tensor_dtype scalar)
{
  // Create new tensor for scalar multiplication
  Tensor* result = tensor_create(tensor->shape, tensor->num_dims);
  if (result == NULL) {
    return NULL;
  }

  // Compute scalar multiplication
  for (size_t i = 0; i < tensor->num_elements; i++) {
    result->data[i] = tensor->data[i] * scalar;
  }

  // Return scalar multiplication
  return result;
}

// Compute the division of a tensor by a scalar
Tensor*
tensor_scalar_divide(const Tensor* tensor, tensor_dtype scalar)
{
  // Create new tensor for scalar division
  Tensor* result = tensor_create(tensor->shape, tensor->num_dims);
  if (result == NULL) {
    return NULL;
  }

  // Compute scalar division
  for (size_t i = 0; i < tensor->num_elements; i++) {
    result->data[i] = tensor->data[i] / scalar;
  }

  // Return scalar division
  return result;
}

// Compute the power of a tensor by a scalar
Tensor*
tensor_scalar_power(const Tensor* tensor, tensor_dtype scalar)
{
  // Create new tensor for scalar power
  Tensor* result = tensor_create(tensor->shape, tensor->num_dims);
  if (result == NULL) {
    return NULL;
  }

  // Compute scalar power
  for (size_t i = 0; i < tensor->num_elements; i++) {
    result->data[i] = pow(tensor->data[i], scalar);
  }

  // Return scalar power
  return result;
}

// Compute the matrix multiplication of two tensors
Tensor*
tensor_matmul(const Tensor* tensor1, const Tensor* tensor2)
{
  // Check if tensors are compatible for matrix multiplication
  if (tensor1->num_dims != 2 || tensor2->num_dims != 2 ||
      tensor1->shape[1] != tensor2->shape[0]) {
    fprintf(stderr,
            "Error: Tensors are not compatible for matrix multiplication\n");
    return NULL;
  }

  // Create new tensor for matrix multiplication
  size_t shape[2] = { tensor1->shape[0], tensor2->shape[1] };
  Tensor* tensor = tensor_create(shape, 2);
  if (tensor == NULL) {
    return NULL;
  }

  // Compute matrix multiplication
  for (size_t i = 0; i < tensor1->shape[0]; i++) {
    for (size_t j = 0; j < tensor2->shape[1]; j++) {
      tensor_dtype value = 0;
      for (size_t k = 0; k < tensor1->shape[1]; k++) {
        value += tensor_get_value(tensor1, (size_t[]){ i, k }) *
                 tensor_get_value(tensor2, (size_t[]){ k, j });
      }
      tensor_set_value(tensor, (size_t[]){ i, j }, value);
    }
  }

  // Return matrix multiplication
  return tensor;
}

// Compute the transpose of a tensor
Tensor*
tensor_transpose(const Tensor* tensor)
{
  // Create new tensor for transpose
  size_t shape[tensor->num_dims];
  for (size_t i = 0; i < tensor->num_dims; i++) {
    shape[i] = tensor->shape[tensor->num_dims - i - 1];
  }
  Tensor* result = tensor_create(shape, tensor->num_dims);
  if (result == NULL) {
    return NULL;
  }

  // Compute transpose
  size_t indices[tensor->num_dims];
  for (size_t i = 0; i < tensor->num_elements; i++) {
    size_t remainder = i;
    for (size_t j = 0; j < tensor->num_dims; j++) {
      indices[j] = remainder % tensor->shape[j];
      remainder /= tensor->shape[j];
    }
    for (size_t j = 0; j < tensor->num_dims; j++) {
      indices[j] = shape[j] - indices[j] - 1;
    }
    tensor_set_value(result, indices, tensor->data[i]);
  }

  // Return transpose
  return result;
}

// Compute the sum of a tensor along a given axis
Tensor*
tensor_sum(const Tensor* tensor, size_t axis)
{
  // Check if axis is valid
  if (axis >= tensor->num_dims) {
    fprintf(stderr, "Error: Axis is out of bounds\n");
    return NULL;
  }

  // Compute sum along axis
  size_t shape[tensor->num_dims - 1];
  size_t num_elements = 1;
  for (size_t i = 0, j = 0; i < tensor->num_dims; i++) {
    if (i != axis) {
      shape[j] = tensor->shape[i];
      num_elements *= tensor->shape[i];
      j++;
    }
  }
  Tensor* result = tensor_create(shape, tensor->num_dims - 1);
  if (result == NULL) {
    return NULL;
  }
  size_t indices[tensor->num_dims];
  for (size_t i = 0; i < num_elements; i++) {
    size_t remainder = i;
    for (size_t j = 0; j < tensor->num_dims - 1; j++) {
      indices[j] = remainder % shape[j];
      remainder /= shape[j];
    }
    indices[axis] = 0;
    tensor_dtype value = 0;
    for (size_t j = 0; j < tensor->shape[axis]; j++) {
      indices[axis] = j;
      value += tensor_get_value(tensor, indices);
    }
    tensor_set_value(result, indices, value);
  }

  // Return sum along axis
  return result;
}

// Compute the mean of a tensor along a given axis
Tensor*
tensor_mean(const Tensor* tensor, size_t axis)
{
  // Check if axis is valid
  if (axis >= tensor->num_dims) {
    fprintf(stderr, "Error: Axis is out of bounds\n");
    return NULL;
  }

  // Compute mean along axis
  size_t shape[tensor->num_dims - 1];
  size_t num_elements = 1;
  for (size_t i = 0, j = 0; i < tensor->num_dims; i++) {
    if (i != axis) {
      shape[j] = tensor->shape[i];
      num_elements *= tensor->shape[i];
      j++;
    }
  }
  Tensor* result = tensor_create(shape, tensor->num_dims - 1);
  if (result == NULL) {
    return NULL;
  }
  size_t indices[tensor->num_dims];
  for (size_t i = 0; i < num_elements; i++) {
    size_t remainder = i;
    for (size_t j = 0; j < tensor->num_dims - 1; j++) {
      indices[j] = remainder % shape[j];
      remainder /= shape[j];
    }
    indices[axis] = 0;
    tensor_dtype value = 0;
    for (size_t j = 0; j < tensor->shape[axis]; j++) {
      indices[axis] = j;
      value += tensor_get_value(tensor, indices);
    }
    value /= tensor->shape[axis];
    tensor_set_value(result, indices, value);
  }

  // Return mean along axis
  return result;
}

// Compute the maximum of a tensor along a given axis
Tensor*
tensor_max(const Tensor* tensor, size_t axis)
{
  // Check if axis is valid
  if (axis >= tensor->num_dims) {
    fprintf(stderr, "Error: Axis is out of bounds\n");
    return NULL;
  }

  // Compute maximum along axis
  size_t shape[tensor->num_dims - 1];
  size_t num_elements = 1;
  for (size_t i = 0, j = 0; i < tensor->num_dims; i++) {
    if (i != axis) {
      shape[j] = tensor->shape[i];
      num_elements *= tensor->shape[i];
      j++;
    }
  }
  Tensor* result = tensor_create(shape, tensor->num_dims - 1);
  if (result == NULL) {
    return NULL;
  }
  size_t indices[tensor->num_dims];
  for (size_t i = 0; i < num_elements; i++) {
    size_t remainder = i;
    for (size_t j = 0; j < tensor->num_dims - 1; j++) {
      indices[j] = remainder % shape[j];
      remainder /= shape[j];
    }
    indices[axis] = 0;
    tensor_dtype value = tensor_get_value(tensor, indices);
    for (size_t j = 0; j < tensor->shape[axis]; j++) {
      indices[axis] = j;
      value = fmax(value, tensor_get_value(tensor, indices));
    }
    tensor_set_value(result, indices, value);
  }

  // Return maximum along axis
  return result;
}

// Compute the minimum of a tensor along a given axis
Tensor*
tensor_min(const Tensor* tensor, size_t axis)
{
  // Check if axis is valid
  if (axis >= tensor->num_dims) {
    fprintf(stderr, "Error: Axis is out of bounds\n");
    return NULL;
  }

  // Compute minimum along axis
  size_t shape[tensor->num_dims - 1];
  size_t num_elements = 1;
  for (size_t i = 0, j = 0; i < tensor->num_dims; i++) {
    if (i != axis) {
      shape[j] = tensor->shape[i];
      num_elements *= tensor->shape[i];
      j++;
    }
  }
  Tensor* result = tensor_create(shape, tensor->num_dims - 1);
  if (result == NULL) {
    return NULL;
  }
  size_t indices[tensor->num_dims];
  for (size_t i = 0; i < num_elements; i++) {
    size_t remainder = i;
    for (size_t j = 0; j < tensor->num_dims - 1; j++) {
      indices[j] = remainder % shape[j];
      remainder /= shape[j];
    }
    indices[axis] = 0;
    tensor_dtype value = tensor_get_value(tensor, indices);
    for (size_t j = 0; j < tensor->shape[axis]; j++) {
      indices[axis] = j;
      value = fmin(value, tensor_get_value(tensor, indices));
    }
    tensor_set_value(result, indices, value);
  }

  // Return minimum along axis
  return result;
}

// Compute the mode of a tensor along a given axis
Tensor*
tensor_mode(const Tensor* tensor, size_t axis)
{
  // Check if axis is valid
  if (axis >= tensor->num_dims) {
    fprintf(stderr, "Error: Axis is out of bounds\n");
    return NULL;
  }

  // Compute mode along axis
  size_t shape[tensor->num_dims - 1];
  size_t num_elements = 1;
  for (size_t i = 0, j = 0; i < tensor->num_dims; i++) {
    if (i != axis) {
      shape[j] = tensor->shape[i];
      num_elements *= tensor->shape[i];
      j++;
    }
  }
  Tensor* result = tensor_create(shape, tensor->num_dims - 1);
  if (result == NULL) {
    return NULL;
  }
  size_t indices[tensor->num_dims];
  for (size_t i = 0; i < num_elements; i++) {
    size_t remainder = i;
    for (size_t j = 0; j < tensor->num_dims - 1; j++) {
      indices[j] = remainder % shape[j];
      remainder /= shape[j];
    }
    indices[axis] = 0;
    tensor_dtype value = tensor_get_value(tensor, indices);
    size_t count = 1;
    size_t max_count = 1;
    tensor_dtype mode = value;
    for (size_t j = 1; j < tensor->shape[axis]; j++) {
      indices[axis] = j;
      tensor_dtype next_value = tensor_get_value(tensor, indices);
      if (next_value == value) {
        count++;
      } else {
        value = next_value;
        count = 1;
      }
      if (count > max_count) {
        max_count = count;
        mode = value;
      }
    }
    tensor_set_value(result, indices, mode);
  }

  // Return mode along axis
  return result;
}