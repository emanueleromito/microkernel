#include "microkernel/tensor.h"

// Constructor

Tensor::Tensor(const std::vector<int>& shape) {

    // SHAPES

    shape_ = shape;

    // STRIDES

    // Strides define how many elements to skip in the flat array
    // to move one step along each dimension.
    // Computed right-to-left: strides[last] = 1,
    // strides[i] = strides[i+1] * shape[i+1].
    // Example:
    // Shapes: 2 - - 3 - - 2
    // Strides: 6 - - 2 - - 1

    strides_.resize(shape_.size());

    // shape_.size() - 1 access last element of shape_
    strides_[shape_.size() - 1] = 1;

    // if we have 3 dimensions, shape_.size() returns 3, and we need to calculate 2 strides (last one already done manually)
    // since we access the vector by indices, we need to also remove 1 value more (-2) to access the 0 and 1 slot of the stride_ vector.
    for (int i = shape_.size() - 2; i>=0 ; i--) {
        strides_[i] = strides_[i+1] * shape_[i+1];
    }

    // DATA

    int data_size {1};

    for (int dim: shape_){
        data_size*=dim;
    }

    data_.resize(data_size);

}

// Size

int Tensor::size() const {

    int data_size {1};

    for (int dim: shape_){
        data_size*=dim;
    }

    return data_size;

}

// Fill

void Tensor::fill(float value) {

    for (float& element: data_) {

        element = value;

    }

}

// At

float& Tensor::at(const std::vector<int>& indices) {

    int address{};

    // Retrieve the element address
    for (int i=0; i < strides_.size(); i++) {
        
        address += indices[i] * strides_[i];

    }

    return data_[address];

}