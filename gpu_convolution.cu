#include "filter.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void meanFilterKernel(const unsigned char* input, unsigned char* output, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int k = kernelSize / 2;

    if (x >= k && x < (width - k) && y >= k && y < (height - k)) {
        int sum = 0;
        for (int dy = -k; dy <= k; ++dy) {
            for (int dx = -k; dx <= k; ++dx) {
                sum += input[(y + dy) * width + (x + dx)];
            }
        }
        output[y * width + x] = sum / (kernelSize * kernelSize);
    }
}

__global__ void genericFilterKernel(const unsigned char* input, unsigned char* output, const float* kernel, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = kernelSize / 2;

    if (x >= k && x < (width - k) && y >= k && y < (height - k)) {
        float sum = 0.0f;
        for (int dy = -k; dy <= k; ++dy) {
            for (int dx = -k; dx <= k; ++dx) {
                float val = input[(y + dy) * width + (x + dx)];
                float coeff = kernel[(dy + k) * kernelSize + (dx + k)];
                sum += val * coeff;
            }
        }
        output[y * width + x] = static_cast<unsigned char>(fminf(fmaxf(sum, 0.0f), 255.0f));
    }
}

__global__ void colorFilterKernel(const unsigned char* input, unsigned char* output, const float* kernel, int width, int height, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = kernelSize / 2;

    if (x >= k && x < (width - k) && y >= k && y < (height - k)) {
        for (int c = 0; c < 3; ++c) {  
            float sum = 0.0f;
            for (int dy = -k; dy <= k; ++dy) {
                for (int dx = -k; dx <= k; ++dx) {
                    int pixelIndex = ((y + dy) * width + (x + dx)) * 3 + c;
                    int kernelIndex = (dy + k) * kernelSize + (dx + k);
                    sum += input[pixelIndex] * kernel[kernelIndex];
                }
            }
            output[(y * width + x) * 3 + c] = static_cast<unsigned char>(fminf(fmaxf(sum, 0.0f), 255.0f));
        }
    }
}

void applyFilterGPU(const cv::Mat& input, cv::Mat& output, const std::vector<float>& kernel, int kernelSize) {
    int width = input.cols;
    int height = input.rows;
    size_t imageSize = width * height * sizeof(unsigned char);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);

    unsigned char* d_input, * d_output;
    float* d_kernel;

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);

    cudaMemcpy(d_input, input.ptr(), imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), kernelSizeBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    genericFilterKernel <<<gridSize,blockSize>>> (d_input, d_output, d_kernel, width, height, kernelSize);
    cudaDeviceSynchronize();

    cudaMemcpy(output.ptr(), d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

void applyFilterColorGPU(const cv::Mat& input, cv::Mat& output, const std::vector<float>& kernel, int kernelSize) {
    int width = input.cols;
    int height = input.rows;
    size_t imageSize = width * height * 3 * sizeof(unsigned char); 
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);

    unsigned char* d_input;
    unsigned char* d_output;
    float* d_kernel;

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);

    cudaMemcpy(d_input, input.ptr(), imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), kernelSizeBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + 15) / 16, (height + 15) / 16);

    colorFilterKernel <<<gridSize, blockSize >>>(d_input, d_output, d_kernel, width, height, kernelSize);
    cudaDeviceSynchronize();

    output = cv::Mat(input.size(), input.type());
    cudaMemcpy(output.ptr(), d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}