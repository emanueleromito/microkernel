#pragma once

#include <cstddef>
#include <string>
#include <cstdint>

enum class DType { Float32, Float16, Int8, Int4 };

template <typename T> struct dtype_of;

template <> struct dtype_of<float> {

    static constexpr DType dtype = DType::Float32;

};

template <> struct dtype_of<int8_t> {

    static constexpr DType dtype = DType::Int8;

};

// TODO: int4/8 and float16

size_t element_size(DType dtype);
std::string dtype_to_string(DType dtype);