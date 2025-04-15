#pragma once
#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

// CPU
void applyFilterCPU(const cv::Mat& input, cv::Mat& output, const std::vector<float>& kernel, int kernelSize);
void applyFilterColorCPU(const cv::Mat& input, cv::Mat& output, const std::vector<float>& kernel, int kernelSize);

// GPU 
void applyFilterGPU(const cv::Mat& input, cv::Mat& output, const std::vector<float>& kernel, int kernelSize);
void applyFilterColorGPU(const cv::Mat& input, cv::Mat& output, const std::vector<float>& kernel, int kernelSize);
	
#endif