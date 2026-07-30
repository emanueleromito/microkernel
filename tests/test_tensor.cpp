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
            EXPECT_FLOAT_EQ(t.at({i, j}), 7.0f)
                << "Mismatch at [" << i << ", " << j << "]";
        }
    }
}

TEST(TensorFillTest, FillWithZero) {
    Tensor t({3, 3});
    t.fill(0.0f);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t.at({i, j}), 0.0f);
        }
    }
}

TEST(TensorFillTest, FillWithNegativeValue) {
    Tensor t({2, 2});
    t.fill(-1.5f);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(t.at({i, j}), -1.5f);
        }
    }
}

// ---------------------------------------------------------------------------
// TensorAtTest — at() read/write and stride correctness
// ---------------------------------------------------------------------------

TEST(TensorAtTest, WriteAndReadBack1D) {
    Tensor t({5});
    t.at({2}) = 42.0f;
    EXPECT_FLOAT_EQ(t.at({2}), 42.0f);
}

TEST(TensorAtTest, WriteAndReadBack2D) {
    Tensor t({3, 4});
    t.at({1, 2}) = 9.0f;
    EXPECT_FLOAT_EQ(t.at({1, 2}), 9.0f);
}

TEST(TensorAtTest, WriteAndReadBack3D) {
    Tensor t({2, 3, 4});
    t.at({1, 2, 3}) = 3.14f;
    EXPECT_FLOAT_EQ(t.at({1, 2, 3}), 3.14f);
}

TEST(TensorAtTest, IndependentElementsDoNotAlias) {
    Tensor t({3, 3});
    t.fill(0.0f);
    t.at({0, 0}) = 1.0f;
    t.at({1, 1}) = 2.0f;
    t.at({2, 2}) = 3.0f;

    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t.at({2, 2}), 3.0f);
    // Off-diagonal elements must remain zero.
    EXPECT_FLOAT_EQ(t.at({0, 1}), 0.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 0.0f);
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
                EXPECT_FLOAT_EQ(t->at({i, j, k}), 0.0f);
}

TEST_F(TensorFixtureTest, WriteToCornerElements) {
    t->at({0, 0, 0}) = 1.0f;
    t->at({1, 2, 1}) = 99.0f;

    EXPECT_FLOAT_EQ(t->at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t->at({1, 2, 1}), 99.0f);
    // All other elements should still be zero.
    EXPECT_FLOAT_EQ(t->at({0, 1, 0}), 0.0f);
}

TEST_F(TensorFixtureTest, FillOverwritesPreviousValues) {
    t->at({0, 0, 0}) = 5.0f;
    t->fill(2.0f);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 2; ++k)
                EXPECT_FLOAT_EQ(t->at({i, j, k}), 2.0f);
}
