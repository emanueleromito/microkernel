#include <gtest/gtest.h>
#include "microkernel/dtype.h"
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// Test helpers -- raw bit access, independent of Half's own implementation.
// ---------------------------------------------------------------------------

namespace {

uint32_t bits_of(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

float float_of(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

} // namespace

// ---------------------------------------------------------------------------
// element_size()
// ---------------------------------------------------------------------------

TEST(ElementSizeTest, Float32Is4Bytes) {
    EXPECT_EQ(element_size(DType::Float32), 4u);
}

TEST(ElementSizeTest, Float16Is2Bytes) {
    EXPECT_EQ(element_size(DType::Float16), 2u);
}

TEST(ElementSizeTest, Int8Is1Byte) {
    EXPECT_EQ(element_size(DType::Int8), 1u);
}

TEST(ElementSizeTest, Int4Is1Byte) {
    // element_size() reports the per-call unit; Int4's 2-values-per-byte
    // packing is handled separately by Tensor::nbytes(), not here.
    EXPECT_EQ(element_size(DType::Int4), 1u);
}

TEST(ElementSizeTest, UnknownDTypeThrows) {
    EXPECT_THROW(element_size(static_cast<DType>(99)), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// dtype_to_string()
// ---------------------------------------------------------------------------

TEST(DTypeToStringTest, Float32) {
    EXPECT_EQ(dtype_to_string(DType::Float32), "Float32");
}

TEST(DTypeToStringTest, Float16) {
    EXPECT_EQ(dtype_to_string(DType::Float16), "Float16");
}

TEST(DTypeToStringTest, Int8) {
    EXPECT_EQ(dtype_to_string(DType::Int8), "Int8");
}

TEST(DTypeToStringTest, Int4) {
    EXPECT_EQ(dtype_to_string(DType::Int4), "Int4");
}

TEST(DTypeToStringTest, UnknownDTypeThrows) {
    EXPECT_THROW(dtype_to_string(static_cast<DType>(99)), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// float_to_half_bits() -- well-known IEEE-754 binary16 patterns
// ---------------------------------------------------------------------------

TEST(FloatToHalfBitsTest, PositiveZero) {
    EXPECT_EQ(float_to_half_bits(bits_of(0.0f)), 0x0000);
}

TEST(FloatToHalfBitsTest, NegativeZero) {
    EXPECT_EQ(float_to_half_bits(bits_of(-0.0f)), 0x8000);
}

TEST(FloatToHalfBitsTest, PositiveOne) {
    EXPECT_EQ(float_to_half_bits(bits_of(1.0f)), 0x3C00);
}

TEST(FloatToHalfBitsTest, NegativeOne) {
    EXPECT_EQ(float_to_half_bits(bits_of(-1.0f)), 0xBC00);
}

TEST(FloatToHalfBitsTest, Two) {
    EXPECT_EQ(float_to_half_bits(bits_of(2.0f)), 0x4000);
}

TEST(FloatToHalfBitsTest, OneHalf) {
    EXPECT_EQ(float_to_half_bits(bits_of(0.5f)), 0x3800);
}

TEST(FloatToHalfBitsTest, MaxNormalIsExact) {
    // 65504 is the largest finite value binary16 can represent.
    EXPECT_EQ(float_to_half_bits(bits_of(65504.0f)), 0x7BFF);
}

TEST(FloatToHalfBitsTest, FiniteOverflowRoundsToInfinity) {
    // 70000 is finite in float32 but exceeds binary16's range.
    EXPECT_EQ(float_to_half_bits(bits_of(70000.0f)), 0x7C00);
}

TEST(FloatToHalfBitsTest, PositiveInfinity) {
    EXPECT_EQ(float_to_half_bits(bits_of(std::numeric_limits<float>::infinity())), 0x7C00);
}

TEST(FloatToHalfBitsTest, NegativeInfinity) {
    EXPECT_EQ(float_to_half_bits(bits_of(-std::numeric_limits<float>::infinity())), 0xFC00);
}

TEST(FloatToHalfBitsTest, QuietNaNPreservesNaNPattern) {
    uint16_t half_bits = float_to_half_bits(0x7FC00000); // canonical quiet NaN
    uint16_t exponent = (half_bits >> 10) & 0x1F;
    uint16_t mantissa = half_bits & 0x3FF;
    EXPECT_EQ(exponent, 0x1Fu) << "NaN must keep the all-ones exponent";
    EXPECT_NE(mantissa, 0u) << "A zero mantissa here would silently become Infinity";
}

TEST(FloatToHalfBitsTest, SmallestSubnormal) {
    // 2^-24 is the smallest positive value binary16 can represent at all.
    EXPECT_EQ(float_to_half_bits(bits_of(0x1p-24f)), 0x0001);
}

TEST(FloatToHalfBitsTest, MidRangeSubnormal) {
    // 2^-20 lands squarely inside the subnormal branch with a nonzero,
    // non-edge mantissa (0x0010).
    EXPECT_EQ(float_to_half_bits(bits_of(0x1p-20f)), 0x0010);
}

TEST(FloatToHalfBitsTest, PositiveUnderflowFlushesToPositiveZero) {
    EXPECT_EQ(float_to_half_bits(bits_of(1e-12f)), 0x0000);
}

TEST(FloatToHalfBitsTest, NegativeUnderflowFlushesToNegativeZero) {
    EXPECT_EQ(float_to_half_bits(bits_of(-1e-12f)), 0x8000);
}

// ---------------------------------------------------------------------------
// half_to_float_bits() -- mirror of the above, decoding the same patterns
// ---------------------------------------------------------------------------

TEST(HalfToFloatBitsTest, PositiveZero) {
    EXPECT_EQ(float_of(half_to_float_bits(0x0000)), 0.0f);
}

TEST(HalfToFloatBitsTest, NegativeZero) {
    float f = float_of(half_to_float_bits(0x8000));
    EXPECT_EQ(f, 0.0f);
    EXPECT_TRUE(std::signbit(f));
}

TEST(HalfToFloatBitsTest, PositiveOne) {
    EXPECT_FLOAT_EQ(float_of(half_to_float_bits(0x3C00)), 1.0f);
}

TEST(HalfToFloatBitsTest, NegativeOne) {
    EXPECT_FLOAT_EQ(float_of(half_to_float_bits(0xBC00)), -1.0f);
}

TEST(HalfToFloatBitsTest, MaxNormal) {
    EXPECT_FLOAT_EQ(float_of(half_to_float_bits(0x7BFF)), 65504.0f);
}

TEST(HalfToFloatBitsTest, PositiveInfinity) {
    EXPECT_TRUE(std::isinf(float_of(half_to_float_bits(0x7C00))));
}

TEST(HalfToFloatBitsTest, NegativeInfinity) {
    float f = float_of(half_to_float_bits(0xFC00));
    EXPECT_TRUE(std::isinf(f));
    EXPECT_TRUE(std::signbit(f));
}

TEST(HalfToFloatBitsTest, NaN) {
    EXPECT_TRUE(std::isnan(float_of(half_to_float_bits(0x7E00))));
}

TEST(HalfToFloatBitsTest, SmallestSubnormal) {
    EXPECT_FLOAT_EQ(float_of(half_to_float_bits(0x0001)), 0x1p-24f);
}

TEST(HalfToFloatBitsTest, MinNormal) {
    EXPECT_FLOAT_EQ(float_of(half_to_float_bits(0x0400)), 0x1p-14f);
}

// ---------------------------------------------------------------------------
// Half -- round trip through the constructor / conversion operator
// ---------------------------------------------------------------------------

TEST(HalfTest, DefaultConstructedHalfIsTwoBytes) {
    Half h;
    (void)h; // default construction just needs to compile and hold 2 bytes.
    EXPECT_EQ(sizeof(h), 2u);
}

TEST(HalfTest, ExactValuesRoundTripExactly) {
    for (float f : {0.0f, -0.0f, 1.0f, -1.0f, 2.5f, -2.5f, 100.0f, 65504.0f}) {
        EXPECT_FLOAT_EQ(static_cast<float>(Half(f)), f) << "for input " << f;
    }
}

TEST(HalfTest, InexactValueRoundTripsApproximately) {
    // 0.1 is not exactly representable in binary16; the round trip should
    // land close, not exact.
    float f = static_cast<float>(Half(0.1f));
    EXPECT_NEAR(f, 0.1f, 1e-3);
}

// ---------------------------------------------------------------------------
// pack_int4() / unpack_int4() -- standalone, independent of Tensor
// ---------------------------------------------------------------------------

TEST(PackUnpackInt4Test, RoundTripFullRangeLowNibble) {
    for (int8_t v = -8; v <= 7; ++v) {
        std::byte b = pack_int4(std::byte{0}, /*high_nibble=*/false, v);
        EXPECT_EQ(unpack_int4(b, false), v) << "for value " << static_cast<int>(v);
    }
}

TEST(PackUnpackInt4Test, RoundTripFullRangeHighNibble) {
    for (int8_t v = -8; v <= 7; ++v) {
        std::byte b = pack_int4(std::byte{0}, /*high_nibble=*/true, v);
        EXPECT_EQ(unpack_int4(b, true), v) << "for value " << static_cast<int>(v);
    }
}

TEST(PackUnpackInt4Test, NibblesAreIndependent) {
    std::byte b{0};
    b = pack_int4(b, /*high_nibble=*/true, 5);
    b = pack_int4(b, /*high_nibble=*/false, -3);

    EXPECT_EQ(unpack_int4(b, true), 5);
    EXPECT_EQ(unpack_int4(b, false), -3);
}

TEST(PackUnpackInt4Test, OverwritingOneNibbleLeavesTheOtherAlone) {
    std::byte b{0};
    b = pack_int4(b, true, 1);
    b = pack_int4(b, false, 2);
    b = pack_int4(b, true, -1); // overwrite high nibble only

    EXPECT_EQ(unpack_int4(b, true), -1);
    EXPECT_EQ(unpack_int4(b, false), 2);
}

TEST(PackInt4Test, ValueAboveRangeThrows) {
    EXPECT_THROW(pack_int4(std::byte{0}, false, 8), std::out_of_range);
}

TEST(PackInt4Test, ValueBelowRangeThrows) {
    EXPECT_THROW(pack_int4(std::byte{0}, false, -9), std::out_of_range);
}

TEST(PackInt4Test, RangeBoundariesDoNotThrow) {
    EXPECT_NO_THROW(pack_int4(std::byte{0}, false, -8));
    EXPECT_NO_THROW(pack_int4(std::byte{0}, false, 7));
}

TEST(UnpackInt4Test, KnownBitPattern) {
    // 0xAB = 1010 1011 -> low nibble 1011 (-5), high nibble 1010 (-6)
    std::byte b{0xAB};
    EXPECT_EQ(unpack_int4(b, false), -5);
    EXPECT_EQ(unpack_int4(b, true), -6);
}
