// Includes
#include <assert.h>
#include <stdio.h>

#include "../includes/tensor.h"

// Test tensor_create function
void
test_tensor_create()
{
  size_t shape[2] = { 3, 4 };
  Tensor* tensor = tensor_create(shape, 2);
  assert(tensor != NULL);
  assert(tensor->num_dims == 2);
  assert(tensor->shape[0] == 3);
  assert(tensor->shape[1] == 4);
  assert(tensor->num_elements == 12);
  assert(tensor->data != NULL);
  tensor_free(tensor);
}

// Test suite entry point
int
main()
{
  test_tensor_create();
  printf("All tests passed!\n");
  return 0;
}