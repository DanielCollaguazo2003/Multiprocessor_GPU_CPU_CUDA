//// kernel.cu
//
//#include <iostream>
//#include <cuda_runtime.h>
//#include "device_launch_parameters.h"
//
//__global__ void helloFromGPU() {
//    printf("Hola desde GPU (hilo %d)\n", threadIdx.x);
//}
//
//int main() {
//    std::cout << "Llamando al kernel CUDA...\n";
//    helloFromGPU <<<1,5>>> ();
//    cudaDeviceSynchronize();
//
//    return 0;
//}
