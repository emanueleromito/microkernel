#pragma once

#include <cstddef>
#include <string>
#include <cstdint>

enum class DType { Float32, Float16, Int8, Int4 };

struct Half {

    uint16_t bits;

    // Default constructor: Half h; -- ask the compiler to generate the normal one, since
    // declaring the constructor below would otherwise suppress it.
    Half() = default;

    // Converting constructor: Half(3.14f). `explicit` blocks *implicit* float -> Half
    // conversions (e.g. passing a float where a Half is expected without an explicit cast) --
    // this conversion loses precision, so it should always be visible at the call site.
    explicit Half(float f);

    // Conversion operator: lets a Half be used anywhere a float is expected, e.g.
    // `float f = some_half;`. No name/return type of its own -- the target type (float) is both.
    // Left non-explicit (unlike the constructor above) since this direction doesn't lose
    // precision and this type exists mainly for storage/interop convenience.
    operator float() const;

};

static_assert(sizeof(Half) == 2);

template <typename T> struct dtype_of;

template <> struct dtype_of<float> {

    static constexpr DType dtype = DType::Float32;

};

template <> struct dtype_of<int8_t> {

    static constexpr DType dtype = DType::Int8;

};

template <> struct dtype_of<Half> {

    static constexpr DType dtype = DType::Float16;

};

uint16_t float_to_half_bits(const uint32_t value) noexcept;
uint32_t half_to_float_bits(const uint16_t value) noexcept;

int8_t unpack_int4(std::byte b, bool high_nibble);
std::byte pack_int4(std::byte b, bool high_nibble, int8_t value);

size_t element_size(DType dtype);
std::string dtype_to_string(DType dtype);