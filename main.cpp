#include <iostream>
#include <opencv2/opencv.hpp>
#include "filter.h"
#include "kernels.cpp"

void calcularErrorPorcentual(const cv::Mat& cpu, const cv::Mat& gpu) {
    if (cpu.size() != gpu.size() || cpu.type() != gpu.type()) {
        std::cerr << "Las imágenes tienen diferente tamaño o tipo, no se pueden comparar." << std::endl;
        return;
    }

    int canales = cpu.channels();
    int total_valores = cpu.rows * cpu.cols * canales;
    int valores_diferentes = 0;

    for (int y = 0; y < cpu.rows; ++y) {
        for (int x = 0; x < cpu.cols; ++x) {
            if (canales == 1) {
                uchar v1 = cpu.at<uchar>(y, x);
                uchar v2 = gpu.at<uchar>(y, x);
                if (v1 != v2) {
                    valores_diferentes++;
                }
            }
            else if (canales == 3) {
                cv::Vec3b v1 = cpu.at<cv::Vec3b>(y, x);
                cv::Vec3b v2 = gpu.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; ++c) {
                    if (v1[c] != v2[c]) {
                        valores_diferentes++;
                    }
                }
            }
        }
    }

    float porcentaje_error = (valores_diferentes * 100.0f) / total_valores;

    std::cout << "Valores diferentes (por canal, error exacto): "
        << valores_diferentes << " de " << total_valores << std::endl;
    std::cout << "Porcentaje de error CPU vs GPU: " << porcentaje_error << "%" << std::endl;
}


int main() {
    std::string image_path = "C:/Users/colla/OneDrive/Escritorio/Universidad/8vo Ciclo/Computacion_Paralela/cuda_practicas/Multiprocessor_Convolution_Cuda/x64/Debug/ej1.jpg";
    cv::Mat input = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat inputColor = cv::imread(image_path, cv::IMREAD_COLOR_BGR);

    if (input.empty()) {
        std::cerr << "No se pudo cargar la imagen." << std::endl;
        return -1;
    }

    std::cout << "Imagen cargada: " << input.cols << "x" << input.rows << std::endl;

    int kernelSize = 3;
    float sigma = 1.5f;
    auto kernel = Kernels::generateSobelXKernel(kernelSize);
    auto kernel2 = Kernels::generateGaussianKernel(kernelSize, sigma);
    auto kernel3 = Kernels::generateLaplacianKernel(kernelSize);

    std::cout << "EJECUCION CPU " << std::endl;
    cv::Mat output_cpu(input.size(), input.type());

    double t1 = static_cast<double>(cv::getTickCount());
    //applyFilterCPU(input, output_cpu, kernel, kernelSize);
    applyFilterColorCPU(inputColor, output_cpu, kernel2, kernelSize);
    double t2 = static_cast<double>(cv::getTickCount());
    double time_cpu = (t2 - t1) / cv::getTickFrequency();

    std::cout << "Tiempo CPU (filtro 9x9): " << time_cpu << " segundos" << std::endl;

    
    std::cout << "EJECUCION GPU " << std::endl;
    cv::Mat output_gpu(input.size(), input.type());

    double t3 = static_cast<double>(cv::getTickCount());
    //applyFilterGPU(input, output_gpu, kernel, kernelSize);
    applyFilterColorGPU(inputColor, output_gpu, kernel2, kernelSize);

    double t4 = static_cast<double>(cv::getTickCount());
    double time_gpu = (t4 - t3) / cv::getTickFrequency();

    std::cout << "Tiempo GPU (filtro 9x9): " << time_gpu << " segundos" << std::endl;


    cv::Mat cpu_fixed, gpu_fixed;
    output_cpu.convertTo(cpu_fixed, CV_8UC3);
    output_gpu.convertTo(gpu_fixed, CV_8UC3);

    //// Modificación temporal para test: cambiar manualmente 10 píxeles en un solo canal
    //for (int i = 0; i < 10; ++i) {
    //    int x = rand() % output_gpu.cols;
    //    int y = rand() % output_gpu.rows;
    //    if (gpu_fixed.channels() == 3) {
    //        cv::Vec3b& pixel = gpu_fixed.at<cv::Vec3b>(y, x);
    //        pixel[2] = (pixel[2] + 50) % 256;
    //    }
    //}

    calcularErrorPorcentual(cpu_fixed, gpu_fixed);


    // Mostrar resultado
    cv::Mat input_resized, output_cpu_resized, output_gpu_resized;
    cv::resize(input, input_resized, cv::Size(), 0.2, 0.2);
    cv::resize(output_cpu, output_cpu_resized, cv::Size(), 0.2, 0.2);
    cv::resize(output_gpu, output_gpu_resized, cv::Size(), 0.2, 0.2);

    // Mostrar resultado
    cv::imshow("Original (resized)", input_resized);
    cv::imshow("Filtrado CPU (resized)", output_cpu_resized);
    cv::imshow("Filtrado GPU (resized)", output_gpu_resized);
    cv::waitKey(0);
    return 0;
}
