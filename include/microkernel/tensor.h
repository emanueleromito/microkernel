#pragma once

#include "microkernel/dtype.h"
#include <vector>
#include <memory>
#include <cstddef>
#include <stdexcept>
#include <string>

class Tensor{

    private:
        std::shared_ptr<std::vector<std::byte>> data_;
        std::vector<int> shape_;
        std::vector<int> strides_;
        DType dtype_;

        int get_flat_index(const std::vector<int>& indices) const;

        std::string shape_to_string(const std::vector<int>& shape) const;

        template <typename T, typename Op>
        void apply_elementwise(const Tensor& other_tensor, Tensor& result, Op op) const;
        template <typename Op>
        Tensor binary_op(const Tensor& other_tensor, Op op) const;

    public:
        Tensor(const std::vector<int>& shape, DType dtype = DType::Float32);

        Tensor operator+(const Tensor& other_tensor) const;
        Tensor operator-(const Tensor& other_tensor) const;
        Tensor operator*(const Tensor& other_tensor) const;

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

            int flat_index = get_flat_index(indices);

            int byte_index = flat_index * sizeof(T);
            // .data() returns a pointer of the first element of the byte vector data_
            std::byte* byte_pointer = data_->data();

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

            int flat_index = get_flat_index(indices);

            int byte_index = flat_index * sizeof(T);
            // .data() returns a pointer of the first element of the byte vector data_
            const std::byte* byte_pointer = data_->data();

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

            T* typed_pointer = reinterpret_cast<T*>(data_->data());

            for (int i=0; i < size() ; i++){

                typed_pointer[i] = value;

            }

        }


        int8_t get_packed(const std::vector<int>& indices) const;
        void set_packed(const std::vector<int>& indices, int8_t value);
};