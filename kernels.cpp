#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#pragma once

class Kernels {
public:
    static std::vector<float> generateGaussianKernel(int size, float sigma) {
        if (size % 2 == 0) throw std::invalid_argument("Kernel size must be odd");
        std::vector<float> kernel(size * size);
        int center = size / 2;
        float sum = 0.0f;

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                int x = i - center;
                int y = j - center;
                float value = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
                kernel[i * size + j] = value;
                sum += value;
            }
        }

        for (auto& v : kernel)
            v /= sum;

        return kernel;
    }

    static std::vector<float> generateSobelXKernel(int size) {
        if (size % 2 == 0) throw std::invalid_argument("Kernel size must be odd");
        std::vector<float> kernel(size * size);
        int center = size / 2;

        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                kernel[i * size + j] = j - center;

        return kernel;
    }

    static std::vector<float> generateLaplacianKernel(int size) {
        if (size % 2 == 0) throw std::invalid_argument("Kernel size must be odd");
        std::vector<float> kernel(size * size, -1.0f);
        int center = size / 2;
        kernel[center * size + center] = size * size - 1;
        return kernel;
    }
};

