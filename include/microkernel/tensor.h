#pragma once

#include <vector>

class Tensor{

    private:
        std::vector<float> data_;
        std::vector<int> shape_;
        std::vector<int> strides_;


    public:
        Tensor(const std::vector<int>& shape);

        int size() const;
        int nbytes() const;

        void fill(float value);

        float& at(const std::vector<int>& indices);

};