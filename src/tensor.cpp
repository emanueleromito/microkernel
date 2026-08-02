#include "microkernel/tensor.h"

// Constructor

Tensor::Tensor(const std::vector<int>& shape, DType dtype) {

    // 1. Shape

    shape_ = shape;
    dtype_ = dtype;

    // 2. Strides

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

    // 3. Data 

    data_.resize(nbytes());

}

int Tensor::size() const {

    int data_size {1};

    for (int dim: shape_){
        data_size*=dim;
    }

    return data_size;

}

size_t Tensor::nbytes() const {

    switch (dtype_) {

        case DType::Int4 : return (size() + 1) / 2;
        default : return element_size(dtype_) * size();

    };

}

DType Tensor::dtype() const {

    return dtype_;

}

const std::vector<int>& Tensor::shape() const {

return shape_;

}

size_t Tensor::ndim() const {

return shape_.size();

}

int Tensor::get_flat_index(const std::vector<int>& indices) const {

    
    // Flat index = sum over all dimensions i of indices[i] * strides_[i]
    // e.g. for shape (2,3,2) with strides (6,2,1), indices {1,2,0}
    // -> flat = 1*6 + 2*2 + 0*1 = 10


    int flat_index {0};

    for (int i=0; i < (indices.size() ); i++){

        if (indices[i] < 0 || indices[i] >= shape_[i]) {

            throw std::out_of_range("Index " + std::to_string(indices[i]) + 
            " out of range for dimension " + std::to_string(i) + " with shape " +
            std::to_string(shape_[i]));
            
        }

        flat_index += indices[i] * strides_[i];

    }

    return flat_index;

}

int8_t Tensor::get_packed(const std::vector<int>& indices) const {

    if(dtype_ != DType::Int4) {

        throw std::invalid_argument("Dtype " + dtype_to_string(dtype_) + " is not Int4.");

    }

    int flat_index = get_flat_index(indices);

    // Two logical elements share each byte (even index -> low nibble, odd -> high nibble).
    // Which byte: integer-divide the flat index by 2.
    // Which nibble: flat_index % 2 -- 0 means low, nonzero (1) means high.

    int byte_index = flat_index / 2;

    int nibble_indicator = flat_index % 2;

    // Read the existing byte out of data_ at that byte index.

    std::byte data_byte =  data_[byte_index];

    // Call unpack_int4 with that byte and the high/low nibble flag, return its result directly.

    return unpack_int4(data_byte, nibble_indicator);

}

void Tensor::set_packed(const std::vector<int>& indices, int8_t value) {
    
    if(dtype_ != DType::Int4) {

        throw std::invalid_argument("Dtype " + dtype_to_string(dtype_) + " is not Int4.");

    }

    int flat_index = get_flat_index(indices);

    // Two logical elements share each byte (even index -> low nibble, odd -> high nibble).
    // Which byte: integer-divide the flat index by 2.
    // Which nibble: flat_index % 2 -- 0 means low, nonzero (1) means high.

    int byte_index = flat_index / 2;

    int nibble_indicator = flat_index % 2;

    // Read the EXISTING byte at that position first -- pack_int4 needs it to avoid clobbering
    // the other nibble packed into the same byte.

    std::byte existing_byte = data_[byte_index];

    // Call pack_int4 with the existing byte, the nibble flag, and the new value -- it already
    // validates value is in [-8, 7] and throws if not, so no need to duplicate that check here.

    std::byte data_byte = pack_int4(existing_byte, nibble_indicator, value);

    // Write the returned byte back into data_ at byte_index.

    data_[byte_index] = data_byte;

}