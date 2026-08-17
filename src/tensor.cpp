#include "microkernel/tensor.h"

// Constructor for new Tensor

Tensor::Tensor(const std::vector<int>& shape, DType dtype) {

    // 1. Shape

    shape_ = shape;
    dtype_ = dtype;

    // 2. Strides

    strides_ = compute_contiguous_strides(shape_);

    // 3. Data

    data_ = std::make_shared<std::vector<std::byte>>(nbytes());

}


// Strides define how many elements to skip in the flat array to move one step
// along each dimension. Computed right-to-left: strides[last] = 1,
// strides[i] = strides[i+1] * shape[i+1].
// Example:
// Shapes: 2 - - 3 - - 2
// Strides: 6 - - 2 - - 1
std::vector<int> Tensor::compute_contiguous_strides(const std::vector<int>& shape) const {

    std::vector<int> strides(shape.size());

    // shape.size() - 1 accesses the last element of shape
    strides[shape.size() - 1] = 1;

    // if we have 3 dimensions, shape.size() returns 3, and we need to calculate 2 strides
    // (last one already done manually) -- since we access the vector by indices, we need
    // to also remove 1 value more (-2) to access the 0 and 1 slot of the strides vector.
    for (int i = shape.size() - 2; i >= 0; i--) {
        strides[i] = strides[i+1] * shape[i+1];
    }

    return strides;

}


// Constructor for reshaped or transposed tensor

Tensor::Tensor(const std::vector<int>& new_shape, const std::vector<int>& new_stride, 
DType dtype, std::shared_ptr<std::vector<std::byte>> data_pointer) {

    data_ = data_pointer;
    strides_ = new_stride;
    shape_ = new_shape;
    dtype_ = dtype;
}


Tensor Tensor::transpose(int dim0, int dim1) const {

    int n = static_cast<int>(ndim());

    if (dim0 < 0 || dim0 >= n || dim1 < 0 || dim1 >= n) {
        throw std::out_of_range("transpose: dimension index out of range for tensor with " +
            std::to_string(n) + " dimensions.");
    }

    // Swapping the shape entries relabels which dimension is which (dim0 is now
    // dim1's old size and vice versa) on its own this would just describe a
    // different tensor, not a transpose of this one.
    std::vector<int> new_shape = shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);

    // Swapping the *strides* alongside the shape is what actually makes this a
    // transpose: indexing the result at [.., dim0, .., dim1, ..] now walks the
    // buffer using the stride that used to belong to the other dimension, so
    // element [i, j] in the result reads the same byte as [j, i] did before.
    // std::swap(a, b) exchanges two values in place (a becomes b's old value,
    // b becomes a's) -- here it swaps the two vector elements at indices
    // dim0/dim1, same as a manual temp-variable swap would.
    std::vector<int> new_strides = strides_;
    std::swap(new_strides[dim0], new_strides[dim1]);

    return Tensor(new_shape, new_strides, dtype_, data_);

}

// Contiguous -> view (shares data_). Non-contiguous (e.g. after transpose()) -> copy.
Tensor Tensor::reshape(const std::vector<int>& new_shape) const {

    // total elements requested by new_shape
    int new_size = 1;

    for (int dim: new_shape){
        new_size*=dim;
    }

    // must match this tensor's element count
    if (new_size!=size()) {

        throw std::invalid_argument("reshape: new shape " + shape_to_string(new_shape) +
            " has " + std::to_string(new_size) + " elements, but tensor has " +
            std::to_string(size()) + " elements (shape " + shape_to_string(shape_) + ").");

    }

    // what strides_ would be if *this were contiguous
    std::vector<int> contiguous_strides = compute_contiguous_strides(shape_);

    if(strides_ == contiguous_strides) {

        // already contiguous -> view, same data_, no copy
        return Tensor(new_shape, compute_contiguous_strides(new_shape), dtype_, data_);

    } else {

        // fresh, contiguous buffer sized for new_shape
        Tensor new_tensor = Tensor(new_shape, dtype_);

        // k = flat position, same for source's logical order and new_tensor's buffer
        for(int k = 0; k < new_tensor.size(); k++) {

            // scratch copy of k, consumed by the loop below
            int remaining = k;

            // one coordinate per source dimension
            std::vector<int> indices;
            indices.resize(ndim());

            // unravel k into indices against shape_
            for (int dim = ndim() -1 ; dim>=0; dim--) {

                // this dimension's coordinate
                indices[dim] = remaining % shape_[dim];
                // strip it off, leaving the next dimension's remainder
                remaining /= shape_[dim];

            }

            switch (dtype_) {

                case DType::Float32: {
                    // dst points at new_tensor's buffer, read as float*; read the source
                    // element at indices (correct even if *this is non-contiguous), write
                    // it to dst[k] -- new_tensor is contiguous, so slot k is just dst[k].
                    float* dst = reinterpret_cast<float*>(new_tensor.data_->data());
                    dst[k] = at<float>(indices);
                    break;
                }
                case DType::Float16: {
                    Half* dst = reinterpret_cast<Half*>(new_tensor.data_->data());
                    dst[k] = at<Half>(indices);
                    break;
                }
                case DType::Int8: {
                    int8_t* dst = reinterpret_cast<int8_t*>(new_tensor.data_->data());
                    dst[k] = at<int8_t>(indices);
                    break;
                }
                case DType::Int4: {
                    int8_t value = get_packed(indices);
                    int byte_index = k / 2;
                    int nibble = k % 2;
                    std::byte existing_byte = (*new_tensor.data_)[byte_index];
                    (*new_tensor.data_)[byte_index] = pack_int4(existing_byte, nibble, value);
                    break;
                }

            }

        }

        return new_tensor;

    }
    


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

std::string Tensor::shape_to_string(const std::vector<int>& shape) const {

    std::string stringed_shape;

    stringed_shape += '(';

    for(int i=0; i < shape.size(); i++) {

        stringed_shape += std::to_string(shape[i]);

        if (i!= shape.size() - 1) {

            stringed_shape += ", ";

        }

    }

    stringed_shape += ')';

    return stringed_shape;

}

template <typename T, typename Op>
void Tensor::apply_elementwise(const Tensor& other_tensor, Tensor& result, Op op) const {

    // data_ is stored as raw std::byte, with no type info -- reinterpret_cast reads those
    // same bytes as if they were a T*, so the loop below can index them as real T values
    // instead of bytes.
    const T* this_tensor_pointer = reinterpret_cast<const T*>(data_->data());
    const T* other_tensor_pointer = reinterpret_cast<const T*>(other_tensor.data_->data());
    T* result_tensor_pointer = reinterpret_cast<T*>(result.data_->data());

    for (int i=0; i < size(); i++) {

    // op(...) on two Half operands computes in float (Half only converts implicitly to
    // float, not from it), so the result must be explicitly converted back down to T --
    // a no-op for float/int8_t, but required for Half since its float constructor is explicit.
    result_tensor_pointer[i] = T(op(this_tensor_pointer[i], other_tensor_pointer[i]));

    }

}

template <typename Op>
Tensor Tensor::binary_op(const Tensor& other_tensor, Op op) const {


    if (shape_ != other_tensor.shape_) {

        throw std::invalid_argument("Tensor shape: " + Tensor::shape_to_string(shape_) + 
        " does not match operand Tensor shape: " + Tensor::shape_to_string(other_tensor.shape_) + ".");

    }   
    
    if (dtype_ != other_tensor.dtype_ ) {

        throw std::invalid_argument("Tensor dtype: " + dtype_to_string(dtype_) + 
        " does not match operand Tensor dtype: " + dtype_to_string(other_tensor.dtype_) + ".");

    }

    Tensor result(shape_, dtype_);

    switch (dtype_) {

        case DType::Float32:  apply_elementwise<float>(other_tensor, result, op); break;
        case DType::Float16:  apply_elementwise<Half>(other_tensor, result, op); break;
        case DType::Int8:  apply_elementwise<int8_t>(other_tensor, result, op); break;
        // Int4 has no raw T to hand apply_elementwise; two values share a byte, so this
        // branch unpacks both operands to int8_t, applies op, and repacks the result nibble
        // by nibble, same math as get_packed/set_packed.
        case DType::Int4:

            {

            for (int i=0; i < size(); i++) {

                int byte_index = i / 2;
                int nibble = i % 2;

                int8_t temp_result = op(unpack_int4((*data_)[byte_index], nibble),
                                unpack_int4((*other_tensor.data_)[byte_index], nibble));

                // pack_int4 needs the byte as it currently stands in result so it only
                // overwrites this nibble and leaves the other one (already written, or
                // still zero-initialized) untouched.
                std::byte existing_byte = (*result.data_)[byte_index];

                std::byte new_byte = pack_int4(existing_byte, nibble, temp_result);

                (*result.data_)[byte_index] = new_byte;

            }


            break;

        }
    }

    return result;

}

Tensor Tensor::operator+(const Tensor& other_tensor) const {

    return binary_op(other_tensor, std::plus<>{});

}

Tensor Tensor::operator-(const Tensor& other_tensor) const {

    return binary_op(other_tensor, std::minus<>{});

}

Tensor Tensor::operator*(const Tensor& other_tensor) const {

    return binary_op(other_tensor, std::multiplies<>{});

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

    std::byte data_byte =  (*data_)[byte_index];

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

    std::byte existing_byte = (*data_)[byte_index];

    // Call pack_int4 with the existing byte, the nibble flag, and the new value -- it already
    // validates value is in [-8, 7] and throws if not, so no need to duplicate that check here.

    std::byte data_byte = pack_int4(existing_byte, nibble_indicator, value);

    // Write the returned byte back into data_ at byte_index.

    (*data_)[byte_index] = data_byte;

}