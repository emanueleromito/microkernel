#pragma once

#include "microkernel/dtype.h"
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <string>

class Tensor{

    private:
        std::vector<std::byte> data_;
        std::vector<int> shape_;
        std::vector<int> strides_;
        DType dtype_;


    public:
        Tensor(const std::vector<int>& shape, DType dtype = DType::Float32);

        int size() const;
        size_t nbytes() const;

        DType dtype() const;
        const std::vector<int>& shape() const;
        size_t ndim() const;

        template <typename T> T& at(const std::vector<int>& indices) {

            if (dtype_of<T>::dtype != dtype_){
                throw std::invalid_argument("Requested type " + dtype_to_string(dtype_of<T>::dtype) +
                " does not match tensor dtype " + dtype_to_string(dtype_) + " .");
            }

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

            int byte_index = flat_index * sizeof(T);
            // .data() returns a pointer of the first element of the byte vector data_
            std::byte* byte_pointer = data_.data();

            // pointer + index advances the pointer by that many elements of the pointee type
            byte_pointer = byte_pointer + byte_index;

            T* typed_pointer = reinterpret_cast<T*>(byte_pointer);

            return *typed_pointer;

        }

        // overload for const method which uses at
        template <typename T> const T& at(const std::vector<int>& indices) const {

            if (dtype_of<T>::dtype != dtype_){
                throw std::invalid_argument("Requested type " + dtype_to_string(dtype_of<T>::dtype) +
                " does not match tensor dtype " + dtype_to_string(dtype_) + " .");
            }

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

            int byte_index = flat_index * sizeof(T);
            // .data() returns a pointer of the first element of the byte vector data_
            const std::byte* byte_pointer = data_.data();

            // pointer + index advances the pointer by that many elements of the pointee type
            byte_pointer = byte_pointer + byte_index;

            const T* typed_pointer = reinterpret_cast<const T*>(byte_pointer);

            return *typed_pointer;
    
        }

        template <typename T> void fill(T value) {

            if (dtype_of<T>::dtype != dtype_){
                throw std::invalid_argument("Requested fill type " + dtype_to_string(dtype_of<T>::dtype) +
                " does not match tensor dtype " + dtype_to_string(dtype_) + " .");
            }

            T* typed_pointer = reinterpret_cast<T*>(data_.data());

            for (int i=0; i < size() ; i++){

                typed_pointer[i] = value;

            }

        }

};