#include <iostream>
#include <opencv2/opencv.hpp>
#include "filter.h"
#include "kernels.cpp"
#include <filesystem>

namespace fs = std::filesystem;

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
    std::string output_dir = "salidas";

    // Crear carpeta si no existe
    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
    }

    cv::Mat input = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    cv::Mat inputColor = cv::imread(image_path, cv::IMREAD_COLOR_BGR);

    if (input.empty()) {
        std::cerr << "No se pudo cargar la imagen." << std::endl;
        return -1;
    }

    std::cout << "Imagen cargada: " << input.cols << "x" << input.rows << std::endl;

    std::cout << "------------------------------\n------ Filto Gaussiano -------\n------------------------------\n" << std::endl;
    std::string base_name = fs::path(image_path).stem().string();
    // Primer kernel: 9x9
    std::cout << "\n------------------------------\n--------- Kernel 9x9 ---------\n------------------------------" << std::endl;
    int kernelSize9 = 9;
    float sigma = 1.5f;
    auto kernel9 = Kernels::generateGaussianKernel(kernelSize9, sigma);

    std::cout << "EJECUCION EN LA CPU " << std::endl;
    cv::Mat output_cpu9(input.size(), input.type());
    double t1_9 = static_cast<double>(cv::getTickCount());
    applyFilterColorCPUParallel(inputColor, output_cpu9, kernel9, kernelSize9);
    double t2_9 = static_cast<double>(cv::getTickCount());
    double time_cpu9 = (t2_9 - t1_9) / cv::getTickFrequency();
    std::cout << "Tiempo CPU (filtro 9x9): " << time_cpu9 << " segundos" << std::endl;

    std::cout << "EJECUCION EN LA GPU " << std::endl;
    cv::Mat output_gpu9(input.size(), input.type());
    double t3_9 = static_cast<double>(cv::getTickCount());
    applyFilterColorGPU(inputColor, output_gpu9, kernel9, kernelSize9);
    double t4_9 = static_cast<double>(cv::getTickCount());
    double time_gpu9 = (t4_9 - t3_9) / cv::getTickFrequency();
    std::cout << "Tiempo GPU (filtro 9x9): " << time_gpu9 << " segundos" << std::endl;

    cv::Mat cpu_fixed9, gpu_fixed9;
    output_cpu9.convertTo(cpu_fixed9, CV_8UC3);
    output_gpu9.convertTo(gpu_fixed9, CV_8UC3);
    calcularErrorPorcentual(cpu_fixed9, gpu_fixed9);

    cv::imwrite(output_dir + "/" + base_name + "gausseano9x9_cpu.jpg", output_cpu9);
    cv::imwrite(output_dir + "/" + base_name + "gausseano9x9_gpu.jpg", output_gpu9);

    // Segundo kernel: 13x13
    std::cout << "\n------------------------------\n-------- Kernel 13x13 --------\n------------------------------" << std::endl;
    int kernelSize13 = 13;
    auto kernel13 = Kernels::generateGaussianKernel(kernelSize13, sigma);

    std::cout << "EJECUCION EN LA CPU " << std::endl;
    cv::Mat output_cpu13(input.size(), input.type());
    double t1_13 = static_cast<double>(cv::getTickCount());
    applyFilterColorCPUParallel(inputColor, output_cpu13, kernel13, kernelSize13);
    double t2_13 = static_cast<double>(cv::getTickCount());
    double time_cpu13 = (t2_13 - t1_13) / cv::getTickFrequency();
    std::cout << "Tiempo CPU (filtro 13x13): " << time_cpu13 << " segundos" << std::endl;

    std::cout << "EJECUCION EN LA GPU " << std::endl;
    cv::Mat output_gpu13(input.size(), input.type());
    double t3_13 = static_cast<double>(cv::getTickCount());
    applyFilterColorGPU(inputColor, output_gpu13, kernel13, kernelSize13);
    double t4_13 = static_cast<double>(cv::getTickCount());
    double time_gpu13 = (t4_13 - t3_13) / cv::getTickFrequency();
    std::cout << "Tiempo GPU (filtro 13x13): " << time_gpu13 << " segundos" << std::endl;

    cv::Mat cpu_fixed13, gpu_fixed13;
    output_cpu13.convertTo(cpu_fixed13, CV_8UC3);
    output_gpu13.convertTo(gpu_fixed13, CV_8UC3);
    calcularErrorPorcentual(cpu_fixed13, gpu_fixed13);

    cv::imwrite(output_dir + "/" + base_name + "gausseano13x13_cpu.jpg", output_cpu13);
    cv::imwrite(output_dir + "/" + base_name + "gausseano13x13_gpu.jpg", output_gpu13);

    // Tercer kernel: 21x21
    std::cout << "\n------------------------------\n-------- Kernel 21x21 --------\n------------------------------" << std::endl;
    int kernelSize21 = 21;
    auto kernel21 = Kernels::generateGaussianKernel(kernelSize21, sigma);

    std::cout << "EJECUCION EN LA CPU " << std::endl;
    cv::Mat output_cpu21(input.size(), input.type());
    double t1_21 = static_cast<double>(cv::getTickCount());
    applyFilterColorCPUParallel(inputColor, output_cpu21, kernel21, kernelSize21);
    double t2_21 = static_cast<double>(cv::getTickCount());
    double time_cpu21 = (t2_21 - t1_21) / cv::getTickFrequency();
    std::cout << "Tiempo CPU (filtro 21x21): " << time_cpu21 << " segundos" << std::endl;

    std::cout << "EJECUCION EN LA GPU " << std::endl;
    cv::Mat output_gpu21(input.size(), input.type());
    double t3_21 = static_cast<double>(cv::getTickCount());
    applyFilterColorGPU(inputColor, output_gpu21, kernel21, kernelSize21);
    double t4_21 = static_cast<double>(cv::getTickCount());
    double time_gpu21 = (t4_21 - t3_21) / cv::getTickFrequency();
    std::cout << "Tiempo GPU (filtro 21x21): " << time_gpu21 << " segundos" << std::endl;

    cv::Mat cpu_fixed21, gpu_fixed21;
    output_cpu21.convertTo(cpu_fixed21, CV_8UC3);
    output_gpu21.convertTo(gpu_fixed21, CV_8UC3);
    calcularErrorPorcentual(cpu_fixed21, gpu_fixed21);

    cv::imwrite(output_dir + "/" + base_name + "gausseano21x21_cpu.jpg", output_cpu21);
    cv::imwrite(output_dir + "/" + base_name + "gausseano21x21_gpu.jpg", output_gpu21);

    return 0;
}