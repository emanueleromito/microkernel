#pragma once

#include <vector>

class Tensor{

    private:
        // Attributes
        std::vector<float> data_;
        std::vector<int> shape_;
        std::vector<int> strides_;

    public:
        // Constructor
        Tensor(const std::vector<int>& shape);

        int size() const;

        // Fill all tensor dimensions with one value
        void fill(float value);

        // Write and Read at a position defined by indices
        float& at(const std::vector<int>& indices);

};