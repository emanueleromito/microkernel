#include <gtest/gtest.h>
#include "microkernel/tensor.h"

// ---------------------------------------------------------------------------
// TensorConstructorTest — shape, strides, and initial state
// ---------------------------------------------------------------------------

TEST(TensorConstructorTest, Size1DIsProductOfShape) {
    Tensor t({4});
    EXPECT_EQ(t.size(), 4);
}

TEST(TensorConstructorTest, Size2DIsProductOfShape) {
    Tensor t({2, 3});
    EXPECT_EQ(t.size(), 6);
}

TEST(TensorConstructorTest, Size3DIsProductOfShape) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.size(), 24);
}

TEST(TensorConstructorTest, SingleElementTensorHasSizeOne) {
    Tensor t({1, 1, 1});
    EXPECT_EQ(t.size(), 1);
}

// ---------------------------------------------------------------------------
// TensorSizeTest — size() correctness
// ---------------------------------------------------------------------------

TEST(TensorSizeTest, SizeMatchesExplicitProduct) {
    Tensor t({5, 6});
    EXPECT_EQ(t.size(), 5 * 6);
}

// ---------------------------------------------------------------------------
// TensorFillTest — fill() sets every element
// ---------------------------------------------------------------------------

TEST(TensorFillTest, FillSetsAllElementsToValue) {
    Tensor t({2, 3});
    t.fill(7.0f);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t.at<float>({i, j}), 7.0f)
                << "Mismatch at [" << i << ", " << j << "]";
        }
    }
}

TEST(TensorFillTest, FillWithZero) {
    Tensor t({3, 3});
    t.fill(0.0f);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t.at<float>({i, j}), 0.0f);
        }
    }
}

TEST(TensorFillTest, FillWithNegativeValue) {
    Tensor t({2, 2});
    t.fill(-1.5f);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(t.at<float>({i, j}), -1.5f);
        }
    }
}

// ---------------------------------------------------------------------------
// TensorAtTest — at() read/write and stride correctness
// ---------------------------------------------------------------------------

TEST(TensorAtTest, WriteAndReadBack1D) {
    Tensor t({5});
    t.at<float>({2}) = 42.0f;
    EXPECT_FLOAT_EQ(t.at<float>({2}), 42.0f);
}

TEST(TensorAtTest, WriteAndReadBack2D) {
    Tensor t({3, 4});
    t.at<float>({1, 2}) = 9.0f;
    EXPECT_FLOAT_EQ(t.at<float>({1, 2}), 9.0f);
}

TEST(TensorAtTest, WriteAndReadBack3D) {
    Tensor t({2, 3, 4});
    t.at<float>({1, 2, 3}) = 3.14f;
    EXPECT_FLOAT_EQ(t.at<float>({1, 2, 3}), 3.14f);
}

TEST(TensorAtTest, IndependentElementsDoNotAlias) {
    Tensor t({3, 3});
    t.fill(0.0f);
    t.at<float>({0, 0}) = 1.0f;
    t.at<float>({1, 1}) = 2.0f;
    t.at<float>({2, 2}) = 3.0f;

    EXPECT_FLOAT_EQ(t.at<float>({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at<float>({1, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t.at<float>({2, 2}), 3.0f);
    // Off-diagonal elements must remain zero.
    EXPECT_FLOAT_EQ(t.at<float>({0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(t.at<float>({1, 0}), 0.0f);
}

// ---------------------------------------------------------------------------
// TensorFixtureTest — shared setup via fixture
// ---------------------------------------------------------------------------

class TensorFixtureTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 2×3×2 tensor, total 12 elements.
        t = std::make_unique<Tensor>(std::vector<int>{2, 3, 2});
        t->fill(0.0f);
    }

    std::unique_ptr<Tensor> t;
};

TEST_F(TensorFixtureTest, SizeIs12) {
    EXPECT_EQ(t->size(), 12);
}

TEST_F(TensorFixtureTest, FillInitialisesToZero) {
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 2; ++k)
                EXPECT_FLOAT_EQ(t->at<float>({i, j, k}), 0.0f);
}

TEST_F(TensorFixtureTest, WriteToCornerElements) {
    t->at<float>({0, 0, 0}) = 1.0f;
    t->at<float>({1, 2, 1}) = 99.0f;

    EXPECT_FLOAT_EQ(t->at<float>({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t->at<float>({1, 2, 1}), 99.0f);
    // All other elements should still be zero.
    EXPECT_FLOAT_EQ(t->at<float>({0, 1, 0}), 0.0f);
}

TEST_F(TensorFixtureTest, FillOverwritesPreviousValues) {
    t->at<float>({0, 0, 0}) = 5.0f;
    t->fill(2.0f);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 2; ++k)
                EXPECT_FLOAT_EQ(t->at<float>({i, j, k}), 2.0f);
}

// ---------------------------------------------------------------------------
// TensorInt8Test — at<int8_t>()/fill<int8_t>() round-trip on DType::Int8
// ---------------------------------------------------------------------------

TEST(TensorInt8Test, FillSetsAllElementsToValue) {
    Tensor t({2, 3}, DType::Int8);
    t.fill<int8_t>(7);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_EQ(t.at<int8_t>({i, j}), 7)
                << "Mismatch at [" << i << ", " << j << "]";
        }
    }
}

TEST(TensorInt8Test, FillWithNegativeValue) {
    Tensor t({2, 2}, DType::Int8);
    t.fill<int8_t>(-42);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_EQ(t.at<int8_t>({i, j}), -42);
        }
    }
}

TEST(TensorInt8Test, WriteAndReadBack2D) {
    Tensor t({3, 4}, DType::Int8);
    t.at<int8_t>({1, 2}) = 9;
    EXPECT_EQ(t.at<int8_t>({1, 2}), 9);
}

TEST(TensorInt8Test, IndependentElementsDoNotAlias) {
    Tensor t({3, 3}, DType::Int8);
    t.fill<int8_t>(0);
    t.at<int8_t>({0, 0}) = 1;
    t.at<int8_t>({1, 1}) = 2;
    t.at<int8_t>({2, 2}) = 3;

    EXPECT_EQ(t.at<int8_t>({0, 0}), 1);
    EXPECT_EQ(t.at<int8_t>({1, 1}), 2);
    EXPECT_EQ(t.at<int8_t>({2, 2}), 3);
    // Off-diagonal elements must remain zero.
    EXPECT_EQ(t.at<int8_t>({0, 1}), 0);
    EXPECT_EQ(t.at<int8_t>({1, 0}), 0);
}

TEST(TensorInt8Test, NbytesMatchesOneBytePerElement) {
    Tensor t({2, 3, 4}, DType::Int8);
    EXPECT_EQ(t.nbytes(), static_cast<size_t>(t.size()));
}

TEST(TensorInt8Test, AtWithMismatchedTypeThrows) {
    Tensor t({2, 2}, DType::Int8);
    EXPECT_THROW(t.at<float>({0, 0}), std::invalid_argument);
}

TEST(TensorInt8Test, FillWithMismatchedTypeThrows) {
    Tensor t({2, 2}, DType::Int8);
    EXPECT_THROW(t.fill<float>(1.0f), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// TensorFloat16Test — at<Half>()/fill<Half>() round-trip on DType::Float16
// ---------------------------------------------------------------------------

TEST(TensorFloat16Test, FillSetsAllElementsToValue) {
    Tensor t({2, 3}, DType::Float16);
    t.fill<Half>(Half(3.5f));

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(static_cast<float>(t.at<Half>({i, j})), 3.5f)
                << "Mismatch at [" << i << ", " << j << "]";
        }
    }
}

TEST(TensorFloat16Test, FillWithNegativeValue) {
    Tensor t({2, 2}, DType::Float16);
    t.fill<Half>(Half(-1.5f));

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(static_cast<float>(t.at<Half>({i, j})), -1.5f);
        }
    }
}

TEST(TensorFloat16Test, WriteAndReadBack2D) {
    Tensor t({3, 4}, DType::Float16);
    t.at<Half>({1, 2}) = Half(9.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t.at<Half>({1, 2})), 9.0f);
}

TEST(TensorFloat16Test, IndependentElementsDoNotAlias) {
    Tensor t({3, 3}, DType::Float16);
    t.fill<Half>(Half(0.0f));
    t.at<Half>({0, 0}) = Half(1.0f);
    t.at<Half>({1, 1}) = Half(2.0f);
    t.at<Half>({2, 2}) = Half(3.0f);

    EXPECT_FLOAT_EQ(static_cast<float>(t.at<Half>({0, 0})), 1.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t.at<Half>({1, 1})), 2.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t.at<Half>({2, 2})), 3.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t.at<Half>({0, 1})), 0.0f);
    EXPECT_FLOAT_EQ(static_cast<float>(t.at<Half>({1, 0})), 0.0f);
}

TEST(TensorFloat16Test, StoredValueLosesPrecisionRelativeToFloat32) {
    // 0.1 is not exactly representable in binary16, unlike in the Float32
    // path -- this distinguishes Float16 storage from a same-shaped
    // Float32 tensor rather than just re-testing Half's own conversion.
    Tensor t({1}, DType::Float16);
    t.fill<Half>(Half(0.1f));
    EXPECT_NE(static_cast<float>(t.at<Half>({0})), 0.1f);
    EXPECT_NEAR(static_cast<float>(t.at<Half>({0})), 0.1f, 1e-3);
}

TEST(TensorFloat16Test, NbytesIsTwoBytesPerElement) {
    Tensor t({2, 3, 4}, DType::Float16);
    EXPECT_EQ(t.nbytes(), static_cast<size_t>(t.size()) * 2);
}

TEST(TensorFloat16Test, AtWithMismatchedTypeThrows) {
    Tensor t({2, 2}, DType::Float16);
    EXPECT_THROW(t.at<float>({0, 0}), std::invalid_argument);
}

TEST(TensorFloat16Test, FillWithMismatchedTypeThrows) {
    Tensor t({2, 2}, DType::Float16);
    EXPECT_THROW(t.fill<float>(1.0f), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// TensorInt4Test — get_packed()/set_packed() round-trip on DType::Int4
// ---------------------------------------------------------------------------

TEST(TensorInt4Test, DefaultInitializedTensorReadsAsZero) {
    Tensor t({4}, DType::Int4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(t.get_packed({i}), 0);
    }
}

TEST(TensorInt4Test, RoundTripFullRange) {
    Tensor t({16}, DType::Int4);
    int idx = 0;
    for (int8_t v = -8; v <= 7; ++v, ++idx) {
        t.set_packed({idx}, v);
    }
    idx = 0;
    for (int8_t v = -8; v <= 7; ++v, ++idx) {
        EXPECT_EQ(t.get_packed({idx}), v) << "Mismatch at index " << idx;
    }
}

TEST(TensorInt4Test, AdjacentElementsSharingAByteDoNotAlias) {
    // Flat indices 0 and 1 pack into the same byte (low/high nibble).
    Tensor t({2}, DType::Int4);
    t.set_packed({0}, -8);
    t.set_packed({1}, 7);

    EXPECT_EQ(t.get_packed({0}), -8);
    EXPECT_EQ(t.get_packed({1}), 7);

    t.set_packed({0}, 3);
    EXPECT_EQ(t.get_packed({0}), 3);
    EXPECT_EQ(t.get_packed({1}), 7) << "Overwriting index 0 must not disturb index 1's nibble";
}

TEST(TensorInt4Test, OddSizeRoundsUpToWholeByte) {
    // 5 packed int4 elements need ceil(5/2) = 3 bytes.
    Tensor t({5}, DType::Int4);
    EXPECT_EQ(t.nbytes(), 3u);

    for (int i = 0; i < 5; ++i) {
        t.set_packed({i}, static_cast<int8_t>(i - 2)); // -2..2
    }
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(t.get_packed({i}), static_cast<int8_t>(i - 2));
    }
}

TEST(TensorInt4Test, GetPackedOnNonInt4TensorThrows) {
    Tensor t({2, 2}, DType::Float32);
    EXPECT_THROW(t.get_packed({0, 0}), std::invalid_argument);
}

TEST(TensorInt4Test, SetPackedOnNonInt4TensorThrows) {
    Tensor t({2, 2}, DType::Int8);
    EXPECT_THROW(t.set_packed({0, 0}, 1), std::invalid_argument);
}

TEST(TensorInt4Test, SetPackedOutOfRangeValueThrows) {
    Tensor t({2}, DType::Int4);
    EXPECT_THROW(t.set_packed({0}, 8), std::out_of_range);
    EXPECT_THROW(t.set_packed({0}, -9), std::out_of_range);
}

// ---------------------------------------------------------------------------
// TensorAccessorTest — dtype() / shape() / ndim()
// ---------------------------------------------------------------------------

TEST(TensorAccessorTest, DtypeDefaultsToFloat32) {
    Tensor t({2, 2});
    EXPECT_EQ(t.dtype(), DType::Float32);
}

TEST(TensorAccessorTest, DtypeMatchesExplicitConstructorArgument) {
    Tensor t({2, 2}, DType::Int8);
    EXPECT_EQ(t.dtype(), DType::Int8);
}

TEST(TensorAccessorTest, ShapeMatchesConstructorArgument) {
    std::vector<int> shape{2, 3, 4};
    Tensor t(shape);
    EXPECT_EQ(t.shape(), shape);
}

TEST(TensorAccessorTest, NdimMatchesShapeSize1D) {
    Tensor t({7});
    EXPECT_EQ(t.ndim(), 1u);
}

TEST(TensorAccessorTest, NdimMatchesShapeSize3D) {
    Tensor t({2, 3, 4});
    EXPECT_EQ(t.ndim(), 3u);
}

// ---------------------------------------------------------------------------
// TensorBoundsTest — out-of-range indices
// ---------------------------------------------------------------------------

TEST(TensorBoundsTest, AtThrowsWhenIndexTooHigh) {
    Tensor t({3, 3});
    EXPECT_THROW(t.at<float>({3, 0}), std::out_of_range);
}

TEST(TensorBoundsTest, AtThrowsWhenIndexNegative) {
    Tensor t({3, 3});
    EXPECT_THROW(t.at<float>({-1, 0}), std::out_of_range);
}

// ---------------------------------------------------------------------------
// TensorConstAtTest — const at() overload
// ---------------------------------------------------------------------------

namespace {
float read_const(const Tensor& t, const std::vector<int>& indices) {
    return t.at<float>(indices);
}
} // namespace

TEST(TensorConstAtTest, ConstOverloadReadsWrittenValue) {
    Tensor t({2, 2});
    t.at<float>({1, 1}) = 42.0f;
    EXPECT_FLOAT_EQ(read_const(t, {1, 1}), 42.0f);
}

// ---------------------------------------------------------------------------
// TensorArithmeticTest — operator+, operator-, operator* (element-wise)
// ---------------------------------------------------------------------------

TEST(TensorArithmeticTest, AddFloat32) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(2.0f);
    b.fill(3.0f);

    Tensor c = a + b;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_FLOAT_EQ(c.at<float>({i, j}), 5.0f);
}

TEST(TensorArithmeticTest, SubtractFloat32) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(5.0f);
    b.fill(3.0f);

    Tensor c = a - b;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_FLOAT_EQ(c.at<float>({i, j}), 2.0f);
}

TEST(TensorArithmeticTest, MultiplyFloat32) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(4.0f);
    b.fill(2.5f);

    Tensor c = a * b;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_FLOAT_EQ(c.at<float>({i, j}), 10.0f);
}

TEST(TensorArithmeticTest, AddFloat32PerElementValues) {
    // Distinct per-element values, not a uniform fill -- catches index-mapping
    // bugs that a uniform fill() would hide.
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.at<float>({0, 0}) = 1.0f; a.at<float>({0, 1}) = 2.0f;
    a.at<float>({1, 0}) = 3.0f; a.at<float>({1, 1}) = 4.0f;
    b.at<float>({0, 0}) = 10.0f; b.at<float>({0, 1}) = 20.0f;
    b.at<float>({1, 0}) = 30.0f; b.at<float>({1, 1}) = 40.0f;

    Tensor c = a + b;

    EXPECT_FLOAT_EQ(c.at<float>({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(c.at<float>({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(c.at<float>({1, 0}), 33.0f);
    EXPECT_FLOAT_EQ(c.at<float>({1, 1}), 44.0f);
}

TEST(TensorArithmeticTest, AddDoesNotMutateOperands) {
    Tensor a({2});
    Tensor b({2});
    a.fill(1.0f);
    b.fill(2.0f);

    Tensor c = a + b;

    EXPECT_FLOAT_EQ(a.at<float>({0}), 1.0f);
    EXPECT_FLOAT_EQ(b.at<float>({0}), 2.0f);
}

TEST(TensorArithmeticTest, AddNegativeValues) {
    Tensor a({2});
    Tensor b({2});
    a.fill(-3.0f);
    b.fill(5.0f);

    Tensor c = a + b;

    EXPECT_FLOAT_EQ(c.at<float>({0}), 2.0f);
}

TEST(TensorArithmeticTest, ResultShapeAndDtypeMatchOperands) {
    Tensor a({2, 3, 4});
    Tensor b({2, 3, 4});
    a.fill(1.0f);
    b.fill(1.0f);

    Tensor c = a + b;

    EXPECT_EQ(c.shape(), a.shape());
    EXPECT_EQ(c.dtype(), a.dtype());
}

TEST(TensorArithmeticTest, ThreeDimensionalAdd) {
    Tensor a({2, 3, 2});
    Tensor b({2, 3, 2});
    a.fill(1.0f);
    b.fill(1.0f);

    Tensor c = a + b;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 2; ++k)
                EXPECT_FLOAT_EQ(c.at<float>({i, j, k}), 2.0f);
}

TEST(TensorArithmeticTest, SingleElementTensorAdd) {
    Tensor a({1, 1, 1});
    Tensor b({1, 1, 1});
    a.fill(9.0f);
    b.fill(1.0f);

    Tensor c = a + b;

    EXPECT_FLOAT_EQ(c.at<float>({0, 0, 0}), 10.0f);
}

TEST(TensorArithmeticTest, ChainedOperationsFloat32) {
    Tensor a({2}); a.fill(2.0f);
    Tensor b({2}); b.fill(3.0f);
    Tensor c({2}); c.fill(4.0f);

    Tensor result = (a + b) * c; // (2+3)*4 = 20

    EXPECT_FLOAT_EQ(result.at<float>({0}), 20.0f);
    EXPECT_FLOAT_EQ(result.at<float>({1}), 20.0f);
}

TEST(TensorArithmeticTest, AddInt8) {
    Tensor a({2, 2}, DType::Int8);
    Tensor b({2, 2}, DType::Int8);
    a.fill<int8_t>(10);
    b.fill<int8_t>(-3);

    Tensor c = a + b;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_EQ(c.at<int8_t>({i, j}), 7);
}

TEST(TensorArithmeticTest, SubtractInt8) {
    Tensor a({2}, DType::Int8);
    Tensor b({2}, DType::Int8);
    a.fill<int8_t>(5);
    b.fill<int8_t>(20);

    Tensor c = a - b;

    EXPECT_EQ(c.at<int8_t>({0}), -15);
}

TEST(TensorArithmeticTest, MultiplyInt8) {
    Tensor a({2}, DType::Int8);
    Tensor b({2}, DType::Int8);
    a.fill<int8_t>(6);
    b.fill<int8_t>(7);

    Tensor c = a * b;

    EXPECT_EQ(c.at<int8_t>({0}), 42);
}

TEST(TensorArithmeticTest, AddFloat16) {
    Tensor a({2}, DType::Float16);
    Tensor b({2}, DType::Float16);
    a.fill<Half>(Half(1.5f));
    b.fill<Half>(Half(2.25f));

    Tensor c = a + b;

    EXPECT_FLOAT_EQ(static_cast<float>(c.at<Half>({0})), 3.75f);
    EXPECT_EQ(c.dtype(), DType::Float16);
}

TEST(TensorArithmeticTest, SubtractFloat16) {
    Tensor a({2}, DType::Float16);
    Tensor b({2}, DType::Float16);
    a.fill<Half>(Half(5.0f));
    b.fill<Half>(Half(1.5f));

    Tensor c = a - b;

    EXPECT_FLOAT_EQ(static_cast<float>(c.at<Half>({0})), 3.5f);
}

TEST(TensorArithmeticTest, MultiplyFloat16) {
    Tensor a({2}, DType::Float16);
    Tensor b({2}, DType::Float16);
    a.fill<Half>(Half(2.0f));
    b.fill<Half>(Half(4.0f));

    Tensor c = a * b;

    EXPECT_FLOAT_EQ(static_cast<float>(c.at<Half>({0})), 8.0f);
}

TEST(TensorArithmeticTest, AddInt4) {
    Tensor a({2}, DType::Int4);
    Tensor b({2}, DType::Int4);
    a.set_packed({0}, 3);
    a.set_packed({1}, -2);
    b.set_packed({0}, 2);
    b.set_packed({1}, 1);

    Tensor c = a + b;

    EXPECT_EQ(c.get_packed({0}), 5);
    EXPECT_EQ(c.get_packed({1}), -1);
}

TEST(TensorArithmeticTest, SubtractInt4) {
    Tensor a({2}, DType::Int4);
    Tensor b({2}, DType::Int4);
    a.set_packed({0}, 7);
    a.set_packed({1}, -8);
    b.set_packed({0}, 3);
    b.set_packed({1}, -3);

    Tensor c = a - b;

    EXPECT_EQ(c.get_packed({0}), 4);
    EXPECT_EQ(c.get_packed({1}), -5);
}

TEST(TensorArithmeticTest, MultiplyInt4WithinRange) {
    Tensor a({2}, DType::Int4);
    Tensor b({2}, DType::Int4);
    a.set_packed({0}, 2);
    a.set_packed({1}, -2);
    b.set_packed({0}, 3);
    b.set_packed({1}, 3);

    Tensor c = a * b;

    EXPECT_EQ(c.get_packed({0}), 6);
    EXPECT_EQ(c.get_packed({1}), -6);
}

TEST(TensorArithmeticTest, AddInt4OddSizeDoesNotAliasNeighborNibble) {
    // 3 elements pack into 2 bytes -- verifies odd-length packing survives
    // binary_op's byte/nibble loop intact, same concern as the existing
    // TensorInt4Test.OddSizeRoundsUpToWholeByte case.
    Tensor a({3}, DType::Int4);
    Tensor b({3}, DType::Int4);
    a.set_packed({0}, 1); a.set_packed({1}, 2); a.set_packed({2}, 3);
    b.set_packed({0}, 1); b.set_packed({1}, 1); b.set_packed({2}, 1);

    Tensor c = a + b;

    EXPECT_EQ(c.get_packed({0}), 2);
    EXPECT_EQ(c.get_packed({1}), 3);
    EXPECT_EQ(c.get_packed({2}), 4);
}

TEST(TensorArithmeticTest, AddInt4OverflowThrows) {
    // 7 + 7 = 14, outside Int4's representable [-8, 7] range -- pack_int4
    // must reject it rather than silently wrapping.
    Tensor a({1}, DType::Int4);
    Tensor b({1}, DType::Int4);
    a.set_packed({0}, 7);
    b.set_packed({0}, 7);

    EXPECT_THROW(a + b, std::out_of_range);
}

TEST(TensorArithmeticTest, AddInt4UnderflowThrows) {
    // -8 + -8 = -16, also outside [-8, 7].
    Tensor a({1}, DType::Int4);
    Tensor b({1}, DType::Int4);
    a.set_packed({0}, -8);
    b.set_packed({0}, -8);

    EXPECT_THROW(a + b, std::out_of_range);
}

TEST(TensorArithmeticTest, AddShapeMismatchThrows) {
    Tensor a({2, 2});
    Tensor b({3, 3});
    a.fill(1.0f);
    b.fill(1.0f);

    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(TensorArithmeticTest, SubtractShapeMismatchThrows) {
    Tensor a({2, 2});
    Tensor b({2, 3});
    a.fill(1.0f);
    b.fill(1.0f);

    EXPECT_THROW(a - b, std::invalid_argument);
}

TEST(TensorArithmeticTest, MultiplyShapeMismatchThrows) {
    Tensor a({4});
    Tensor b({5});
    a.fill(1.0f);
    b.fill(1.0f);

    EXPECT_THROW(a * b, std::invalid_argument);
}

TEST(TensorArithmeticTest, AddDtypeMismatchThrows) {
    Tensor a({2, 2}, DType::Float32);
    Tensor b({2, 2}, DType::Int8);
    a.fill(1.0f);
    b.fill<int8_t>(1);

    EXPECT_THROW(a + b, std::invalid_argument);
}

TEST(TensorArithmeticTest, SubtractDtypeMismatchThrows) {
    Tensor a({2}, DType::Float32);
    Tensor b({2}, DType::Float16);
    a.fill(1.0f);
    b.fill<Half>(Half(1.0f));

    EXPECT_THROW(a - b, std::invalid_argument);
}

TEST(TensorArithmeticTest, MultiplyDtypeMismatchThrows) {
    Tensor a({2}, DType::Int8);
    Tensor b({2}, DType::Int4);
    a.fill<int8_t>(1);
    b.set_packed({0}, 1);
    b.set_packed({1}, 1);

    EXPECT_THROW(a * b, std::invalid_argument);
}

// ---------------------------------------------------------------------------
// TensorTransposeTest — transpose() is a view (shares data_), swaps shape/strides
// ---------------------------------------------------------------------------

TEST(TensorTransposeTest, SwapsShape) {
    Tensor t({2, 3});
    Tensor tt = t.transpose(0, 1);

    std::vector<int> expected{3, 2};
    EXPECT_EQ(tt.shape(), expected);
}

TEST(TensorTransposeTest, ReadsCorrectValues) {
    // t[a,b] = a*3 + b, distinct per element.
    Tensor t({2, 3});
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 3; ++b)
            t.at<float>({a, b}) = static_cast<float>(a * 3 + b);

    Tensor tt = t.transpose(0, 1);

    // tt[i,j] must equal t[j,i].
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_FLOAT_EQ(tt.at<float>({i, j}), t.at<float>({j, i}))
                << "Mismatch at tt[" << i << "," << j << "]";
}

TEST(TensorTransposeTest, SharesStorageWithOriginal) {
    Tensor t({2, 3});
    t.fill(0.0f);

    Tensor tt = t.transpose(0, 1);
    tt.at<float>({1, 0}) = 99.0f;

    // tt[1,0] and t[0,1] land on the same flat byte position (flat = 1 either way).
    EXPECT_FLOAT_EQ(t.at<float>({0, 1}), 99.0f);
}

TEST(TensorTransposeTest, MutatingOriginalIsVisibleThroughView) {
    Tensor t({2, 3});
    t.fill(0.0f);

    Tensor tt = t.transpose(0, 1);
    t.at<float>({0, 1}) = 42.0f;

    EXPECT_FLOAT_EQ(tt.at<float>({1, 0}), 42.0f);
}

TEST(TensorTransposeTest, ThreeDimensionalSwapsCorrectDims) {
    // t[a,b,c] = a*12 + b*4 + c, distinct per element.
    Tensor t({2, 3, 4});
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 4; ++c)
                t.at<float>({a, b, c}) = static_cast<float>(a * 12 + b * 4 + c);

    Tensor tt = t.transpose(0, 2); // shape (4,3,2), dim 1 untouched

    std::vector<int> expected{4, 3, 2};
    EXPECT_EQ(tt.shape(), expected);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 2; ++k)
                EXPECT_FLOAT_EQ(tt.at<float>({i, j, k}), t.at<float>({k, j, i}))
                    << "Mismatch at tt[" << i << "," << j << "," << k << "]";
}

TEST(TensorTransposeTest, SameDimIsIdentity) {
    Tensor t({2, 3});
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 3; ++b)
            t.at<float>({a, b}) = static_cast<float>(a * 3 + b);

    Tensor tt = t.transpose(0, 0);

    EXPECT_EQ(tt.shape(), t.shape());
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 3; ++b)
            EXPECT_FLOAT_EQ(tt.at<float>({a, b}), t.at<float>({a, b}));
}

TEST(TensorTransposeTest, Dim0OutOfRangeThrows) {
    Tensor t({2, 3});
    EXPECT_THROW(t.transpose(2, 0), std::out_of_range);
}

TEST(TensorTransposeTest, Dim1OutOfRangeThrows) {
    Tensor t({2, 3});
    EXPECT_THROW(t.transpose(0, 2), std::out_of_range);
}

TEST(TensorTransposeTest, NegativeDimThrows) {
    Tensor t({2, 3});
    EXPECT_THROW(t.transpose(-1, 0), std::out_of_range);
}

// ---------------------------------------------------------------------------
// TensorReshapeTest — view when contiguous, copy otherwise (e.g. after transpose)
// ---------------------------------------------------------------------------

TEST(TensorReshapeTest, ContiguousProducesCorrectShape) {
    Tensor t({2, 3});
    t.fill(1.0f);

    Tensor r = t.reshape({6});

    std::vector<int> expected{6};
    EXPECT_EQ(r.shape(), expected);
    EXPECT_EQ(r.dtype(), t.dtype());
}

TEST(TensorReshapeTest, ContiguousPreservesRowMajorOrder) {
    // t[a,b] = a*3 + b -- flattening row-major should give 0,1,2,3,4,5.
    Tensor t({2, 3});
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 3; ++b)
            t.at<float>({a, b}) = static_cast<float>(a * 3 + b);

    Tensor r = t.reshape({6});

    for (int k = 0; k < 6; ++k)
        EXPECT_FLOAT_EQ(r.at<float>({k}), static_cast<float>(k));
}

TEST(TensorReshapeTest, ToDifferentMultiDimShapePreservesOrder) {
    // (2,3) -> (3,2): same flat sequence 0..5, relabeled into a different shape.
    Tensor t({2, 3});
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 3; ++b)
            t.at<float>({a, b}) = static_cast<float>(a * 3 + b);

    Tensor r = t.reshape({3, 2});

    float expected[3][2] = {{0, 1}, {2, 3}, {4, 5}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_FLOAT_EQ(r.at<float>({i, j}), expected[i][j]);
}

TEST(TensorReshapeTest, ContiguousReshapeIsView) {
    Tensor t({2, 3});
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 3; ++b)
            t.at<float>({a, b}) = static_cast<float>(a * 3 + b);

    Tensor r = t.reshape({6});
    r.at<float>({3}) = 99.0f; // flat position 3 == t[1,0]

    EXPECT_FLOAT_EQ(t.at<float>({1, 0}), 99.0f);
}

TEST(TensorReshapeTest, IdentityReshapeKeepsValues) {
    Tensor t({2, 3});
    t.fill(5.0f);

    Tensor r = t.reshape({2, 3});

    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 3; ++b)
            EXPECT_FLOAT_EQ(r.at<float>({a, b}), 5.0f);
}

TEST(TensorReshapeTest, ElementCountMismatchThrows) {
    Tensor t({2, 3});
    t.fill(1.0f);

    EXPECT_THROW(t.reshape({5}), std::invalid_argument);
}

TEST(TensorReshapeTest, AfterTransposeProducesLogicalOrderNotBufferOrder) {
    // t = [[0,1,2],[3,4,5]]; transpose -> logically [[0,3],[1,4],[2,5]];
    // flattening that (not the raw pre-transpose buffer) should give 0,3,1,4,2,5.
    Tensor t({2, 3});
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 3; ++b)
            t.at<float>({a, b}) = static_cast<float>(a * 3 + b);

    Tensor tt = t.transpose(0, 1);
    Tensor r = tt.reshape({6});

    float expected[6] = {0, 3, 1, 4, 2, 5};
    for (int k = 0; k < 6; ++k)
        EXPECT_FLOAT_EQ(r.at<float>({k}), expected[k]) << "Mismatch at k=" << k;
}

TEST(TensorReshapeTest, AfterTransposeIsCopyNotView) {
    Tensor t({2, 3});
    t.fill(0.0f);

    Tensor tt = t.transpose(0, 1);
    Tensor r = tt.reshape({6});

    r.at<float>({0}) = 123.0f;

    // r is a fresh buffer -- mutating it must not disturb t or tt.
    EXPECT_FLOAT_EQ(t.at<float>({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(tt.at<float>({0, 0}), 0.0f);
}

TEST(TensorReshapeTest, Int8ContiguousView) {
    Tensor t({2, 2}, DType::Int8);
    t.at<int8_t>({0, 0}) = 1;
    t.at<int8_t>({0, 1}) = 2;
    t.at<int8_t>({1, 0}) = 3;
    t.at<int8_t>({1, 1}) = 4;

    Tensor r = t.reshape({4});

    int8_t expected[4] = {1, 2, 3, 4};
    for (int k = 0; k < 4; ++k)
        EXPECT_EQ(r.at<int8_t>({k}), expected[k]);
}

TEST(TensorReshapeTest, Int8AfterTransposeCopy) {
    Tensor t({2, 2}, DType::Int8);
    t.at<int8_t>({0, 0}) = 1;
    t.at<int8_t>({0, 1}) = 2;
    t.at<int8_t>({1, 0}) = 3;
    t.at<int8_t>({1, 1}) = 4;

    Tensor tt = t.transpose(0, 1); // logically [[1,3],[2,4]]
    Tensor r = tt.reshape({4});

    int8_t expected[4] = {1, 3, 2, 4};
    for (int k = 0; k < 4; ++k)
        EXPECT_EQ(r.at<int8_t>({k}), expected[k]);
}

TEST(TensorReshapeTest, Float16ContiguousView) {
    Tensor t({2, 2}, DType::Float16);
    t.at<Half>({0, 0}) = Half(1.5f);
    t.at<Half>({0, 1}) = Half(2.5f);
    t.at<Half>({1, 0}) = Half(3.5f);
    t.at<Half>({1, 1}) = Half(4.5f);

    Tensor r = t.reshape({4});

    float expected[4] = {1.5f, 2.5f, 3.5f, 4.5f};
    for (int k = 0; k < 4; ++k)
        EXPECT_FLOAT_EQ(static_cast<float>(r.at<Half>({k})), expected[k]);
}

TEST(TensorReshapeTest, Float16AfterTransposeCopy) {
    Tensor t({2, 2}, DType::Float16);
    t.at<Half>({0, 0}) = Half(1.5f);
    t.at<Half>({0, 1}) = Half(2.5f);
    t.at<Half>({1, 0}) = Half(3.5f);
    t.at<Half>({1, 1}) = Half(4.5f);

    Tensor tt = t.transpose(0, 1); // logically [[1.5,3.5],[2.5,4.5]]
    Tensor r = tt.reshape({4});

    float expected[4] = {1.5f, 3.5f, 2.5f, 4.5f};
    for (int k = 0; k < 4; ++k)
        EXPECT_FLOAT_EQ(static_cast<float>(r.at<Half>({k})), expected[k]);
}

TEST(TensorReshapeTest, Int4ContiguousView) {
    Tensor t({2, 2}, DType::Int4);
    t.set_packed({0, 0}, -8);
    t.set_packed({0, 1}, 7);
    t.set_packed({1, 0}, -1);
    t.set_packed({1, 1}, 3);

    Tensor r = t.reshape({4});

    int8_t expected[4] = {-8, 7, -1, 3};
    for (int k = 0; k < 4; ++k)
        EXPECT_EQ(r.get_packed({k}), expected[k]);
}

TEST(TensorReshapeTest, Int4AfterTransposeCopy) {
    // Exercises the byte/nibble packing path in reshape's copy branch, including
    // negative values, since Int4 has no raw T and packs two values per byte.
    Tensor t({2, 2}, DType::Int4);
    t.set_packed({0, 0}, -8);
    t.set_packed({0, 1}, 7);
    t.set_packed({1, 0}, -1);
    t.set_packed({1, 1}, 3);

    Tensor tt = t.transpose(0, 1); // logically [[-8,-1],[7,3]]
    Tensor r = tt.reshape({4});

    int8_t expected[4] = {-8, -1, 7, 3};
    for (int k = 0; k < 4; ++k)
        EXPECT_EQ(r.get_packed({k}), expected[k]) << "Mismatch at k=" << k;
}

TEST(TensorReshapeTest, DtypePreservedAfterTransposeCopy) {
    Tensor t({2, 2}, DType::Int8);
    t.fill<int8_t>(1);

    Tensor tt = t.transpose(0, 1);
    Tensor r = tt.reshape({4});

    EXPECT_EQ(r.dtype(), DType::Int8);
}
