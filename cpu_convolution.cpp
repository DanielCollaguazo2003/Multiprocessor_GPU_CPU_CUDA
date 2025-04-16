#include "filter.h"
#include "filter.h"
#include <thread>
#include <vector>
#include <algorithm>

template <typename T>
T clamp(const T& val, const T& low, const T& high) {
    return std::max(low, std::min(val, high));
}

// --- Escala de grises (paralelo) ---
void applyFilterGrayscaleWorker(const cv::Mat& input, cv::Mat& output,
    const std::vector<float>& kernel, int kernelSize,
    int startY, int endY) {
    int k = kernelSize / 2;

    for (int y = startY; y < endY; ++y) {
        for (int x = k; x < input.cols - k; ++x) {
            float sum = 0.0f;
            for (int dy = -k; dy <= k; ++dy) {
                for (int dx = -k; dx <= k; ++dx) {
                    int ky = dy + k;
                    int kx = dx + k;
                    float kernelVal = kernel[ky * kernelSize + kx];
                    sum += kernelVal * static_cast<float>(input.at<uchar>(y + dy, x + dx));
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(clamp(sum, 0.0f, 255.0f));
        }
    }
}

void applyFilterCPUParallel(const cv::Mat& input, cv::Mat& output, const std::vector<float>& kernel, int kernelSize) {
    int k = kernelSize / 2;
    output = cv::Mat::zeros(input.size(), input.type());

    int numThreads = std::thread::hardware_concurrency();
    int rowsPerThread = (input.rows - 2 * k) / numThreads;

    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        int startY = k + i * rowsPerThread;
        int endY = (i == numThreads - 1) ? input.rows - k : startY + rowsPerThread;

        threads.emplace_back(applyFilterGrayscaleWorker,
            std::cref(input), std::ref(output),
            std::cref(kernel), kernelSize, startY, endY);
    }

    for (auto& t : threads) t.join();
}

// --- Color (paralelo) ---
void applyFilterColorWorker(const cv::Mat& input, cv::Mat& output,
    const std::vector<float>& kernel, int kernelSize,
    int startY, int endY) {
    int k = kernelSize / 2;

    for (int y = startY; y < endY; ++y) {
        for (int x = k; x < input.cols - k; ++x) {
            for (int c = 0; c < 3; ++c) {
                float sum = 0.0f;
                for (int dy = -k; dy <= k; ++dy) {
                    for (int dx = -k; dx <= k; ++dx) {
                        int ky = dy + k;
                        int kx = dx + k;
                        float kernelVal = kernel[ky * kernelSize + kx];
                        sum += kernelVal * static_cast<float>(input.at<cv::Vec3b>(y + dy, x + dx)[c]);
                    }
                }
                output.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(clamp(sum, 0.0f, 255.0f));
            }
        }
    }
}

void applyFilterColorCPUParallel(const cv::Mat& input, cv::Mat& output, const std::vector<float>& kernel, int kernelSize) {
    int k = kernelSize / 2;
    output = cv::Mat::zeros(input.size(), input.type());

    int numThreads = std::thread::hardware_concurrency();
    int rowsPerThread = (input.rows - 2 * k) / numThreads;

    std::vector<std::thread> threads;

    for (int i = 0; i < numThreads; ++i) {
        int startY = k + i * rowsPerThread;
        int endY = (i == numThreads - 1) ? input.rows - k : startY + rowsPerThread;

        threads.emplace_back(applyFilterColorWorker,
            std::cref(input), std::ref(output),
            std::cref(kernel), kernelSize, startY, endY);
    }

    for (auto& t : threads) t.join();
}