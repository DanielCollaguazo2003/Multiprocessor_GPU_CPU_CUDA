#include "filter.h"

template <typename T>
T clamp(const T& val, const T& low, const T& high) {
    return std::max(low, std::min(val, high));
}

void applyFilterCPU(const cv::Mat& input, cv::Mat& output, const std::vector<float>& kernel, int kernelSize) {
    int k = kernelSize / 2;

    for (int y = k; y < input.rows - k; ++y) {
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

void applyFilterColorCPU(const cv::Mat& input, cv::Mat& output, const std::vector<float>& kernel, int kernelSize) {
    int k = kernelSize / 2;

    output = cv::Mat::zeros(input.size(), input.type());

    for (int y = k; y < input.rows - k; ++y) {
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