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
