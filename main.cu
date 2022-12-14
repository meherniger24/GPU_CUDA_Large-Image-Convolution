#include<iostream>
#include "cuda_runtime.h"
#include<string>
#include<vector>
#include<fstream>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>


# define PI  3.14159265358979323846 
# define __syncthreads()
using namespace std;

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << "in" << file << "at line" << line;
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ )) 





// convolution along x on device  (using shared memory)
__global__ void Convolution_x_shared_memory(char* out_img, char* in_img, float* kernel_x, int img_w, int out_h, int out_w, int K) {
    extern __shared__ unsigned char sharedPtr[];

    size_t i = blockDim.y * blockIdx.y + threadIdx.y;   // calculate  i (row) index, pointing to the outarray
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;   // calculate  j (column) index, pointing to the outarray
    size_t xi = threadIdx.y;							// calculate  i (row) index, pointing to the shared memory
    size_t xj = threadIdx.x;							// calculate  j (column) index, pointing to the shared memory

    int idx = blockDim.x + K -1;							// calculating length of data that are required in one block
    // storing the  data to shared memory
    for (int m = xj; m < idx; m += blockDim.x) {
        sharedPtr[3 * (xi * idx + m) + 0] = in_img[3 * (i * img_w + blockDim.x * blockIdx.x + m) + 0];
        sharedPtr[3 * (xi * idx + m) + 1] = in_img[3 * (i * img_w + blockDim.x * blockIdx.x + m) + 1];
        sharedPtr[3 * (xi * idx + m) + 2] = in_img[3 * (i * img_w + blockDim.x * blockIdx.x + m) + 2];
    }
    __syncthreads();

    // initializing the register c to store the results
    float c[3];
    for (int x = 0; x < 3; x++) {
        c[x] = 0;
    }
    // applying the convolution with Gaussian kernel along x axis
    for (int k = 0; k < K; k++) {
        c[0] += (unsigned char)sharedPtr[3 * (xi * idx + xj + k) + 0] * kernel_x[k];
        c[1] += (unsigned char)sharedPtr[3 * (xi * idx + xj + k) + 1] * kernel_x[k];
        c[2] += (unsigned char)sharedPtr[3 * (xi * idx + xj + k) + 2] * kernel_x[k];
    }
    // storing the  results from register to outarray
    out_img[3 * (i * out_w + j) + 0] = (unsigned char)c[0];
    out_img[3 * (i * out_w + j) + 1] = (unsigned char)c[1];
    out_img[3 * (i * out_w + j) + 2] = (unsigned char)c[2];
}





// convolution along y on device  (using shared memory)
__global__ void Convolution_y_shared_memory(char* out_img, char* in_img, float* kernel_y, int img_w, int out_h, int out_w, int K) {
    extern __shared__ unsigned char sharedPtr[];

    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;
    size_t xi = threadIdx.y;
    size_t xj = threadIdx.x;

    int idx = blockDim.y + K -1;
    for (int m = xi; m < idx; m += blockDim.y) {
        sharedPtr[3 * (m * blockDim.x + xj) + 0] = in_img[3 * ((blockDim.y * blockIdx.y + m) * img_w + j) + 0];
        sharedPtr[3 * (m * blockDim.x + xj) + 1] = in_img[3 * ((blockDim.y * blockIdx.y + m) * img_w + j) + 1];
        sharedPtr[3 * (m * blockDim.x + xj) + 2] = in_img[3 * ((blockDim.y * blockIdx.y + m) * img_w + j) + 2];
    }
    __syncthreads();

    float c[3];
    for (int x = 0; x < 3; x++) {
        c[x] = 0;
    }

    // applying the convolution with Gaussian kernel along y axis
    for (int k = 0; k < K; k++) {
        c[0] += (unsigned char)sharedPtr[3 * ((xi + k) * blockDim.x + xj) + 0] * kernel_y[k];
        c[1] += (unsigned char)sharedPtr[3 * ((xi + k) * blockDim.x + xj) + 1] * kernel_y[k];
        c[2] += (unsigned char)sharedPtr[3 * ((xi + k) * blockDim.x + xj) + 2] * kernel_y[k];
    }

    // storing the  results from register to outarray
    out_img[3 * (i * out_w + j) + 0] = (unsigned char)c[0];
    out_img[3 * (i * out_w + j) + 1] = (unsigned char)c[1];
    out_img[3 * (i * out_w + j) + 2] = (unsigned char)c[2];
}





// convolution along x on device(without using shared memory)
__global__ void Convolution_x_on_device(char* out_img, char* in_img, float* kernel_x, int img_w, int out_h, int out_w, int C) {
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;	// calculate row index, point to the output
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;	// calculate column index, point to the output


    // initialize the register c
    float c[3];
    for (int x = 0; x < 3; x++)
        c[x] = 0.0f;

    // apply the convolution with Gaussian kernel along x axis
    for (int k = 0; k < C; k++) {
        c[0] += (unsigned char)in_img[3 * (i * img_w + j + k) + 0] * kernel_x[k];
        c[1] += (unsigned char)in_img[3 * (i * img_w + j + k) + 1] * kernel_x[k];
        c[2] += (unsigned char)in_img[3 * (i * img_w + j + k) + 2] * kernel_x[k];
    }

    // storing the  results from register to outarray
    out_img[3 * (i * out_w + j) + 0] = (unsigned char)c[0];
    out_img[3 * (i * out_w + j) + 1] = (unsigned char)c[1];
    out_img[3 * (i * out_w + j) + 2] = (unsigned char)c[2];
}






//  convolution along y on device(without using shared memory)
__global__ void Convolution_y_on_device(char* out_img, char* in_img, float* kernel_y, int img_w, int out_h, int out_w, int C) {
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;
    // initialize the register c
    float c[3];
    for (int x = 0; x < 3; x++)
        c[x] = 0.0f;


    // apply the convolution with Gaussian kernel along y axis
    for (int k = 0; k < C; k++) {
        c[0] += (unsigned char)in_img[3 * ((i + k) * img_w + j) + 0] * kernel_y[k];
        c[1] += (unsigned char)in_img[3 * ((i + k) * img_w + j) + 1] * kernel_y[k];
        c[2] += (unsigned char)in_img[3 * ((i + k) * img_w + j) + 2] * kernel_y[k];
    }

    // storing the  results from register to outarray
    out_img[3 * (i * out_w + j) + 0] = (unsigned char)c[0];
    out_img[3 * (i * out_w + j) + 1] = (unsigned char)c[1];
    out_img[3 * (i * out_w + j) + 2] = (unsigned char)c[2];
}





// convolution kernel along x running on host
void Convolution_x_host(char* out_img, char* in_img, float* kernel_x, int img_w, int out_h, int out_w, int C) {
    float c[3];
    for (int i = 0; i < out_h; i++) {										// calculate the i (row) index, point to the outarray
        for (int j = 0; j < out_w; j++) {									// calculate the j (column) index, point to the outarray
            // initialize the register c to store the results
            for (int x = 0; x < 3; x++)
                c[x] = 0.0f;
            // convolving with Gaussian kernel along x axis
            for (int k = 0; k < C; k++) {
                c[0] += (unsigned char)in_img[3 * (i * img_w + j + k) + 0] * kernel_x[k];
                c[1] += (unsigned char)in_img[3 * (i * img_w + j + k) + 1] * kernel_x[k];
                c[2] += (unsigned char)in_img[3 * (i * img_w + j + k) + 2] * kernel_x[k];
            }
            out_img[3 * (i * out_w + j) + 0] = (unsigned char)c[0];
            out_img[3 * (i * out_w + j) + 1] = (unsigned char)c[1];
            out_img[3 * (i * out_w + j) + 2] = (unsigned char)c[2];
        }
    }
}



// convolution kernel along y running on host, same as convolution_x
void Convolution_y_host(char* out_img, char* in_img, float* kernel_y, int inp_w, int out_h, int out_w, int C) {
    float c[3];
    for (int j = 0; j < out_w; j++) {
        for (int i = 0; i < out_h; i++) {
            for (int x = 0; x < 3; x++)
                c[x] = 0.0f;

            // convolving with Gaussian kernel along y axis
            for (int k = 0; k < C; k++) {
                c[0] += (unsigned char)in_img[3 * ((i + k) * inp_w + j) + 0] * kernel_y[k];
                c[1] += (unsigned char)in_img[3 * ((i + k) * inp_w + j) + 1] * kernel_y[k];
                c[2] += (unsigned char)in_img[3 * ((i + k) * inp_w + j) + 2] * kernel_y[k];
            }
            out_img[3 * (i * out_w + j) + 0] = (unsigned char)c[0];
            out_img[3 * (i * out_w + j) + 1] = (unsigned char)c[1];
            out_img[3 * (i * out_w + j) + 2] = (unsigned char)c[2];
        }
    }
}




// writing targa image array
void write_tga(std::string filename, char* bytes, int width, int height) {
    std::ofstream outfile;
    outfile.open(filename, std::ios::binary | std::ios::out);	// open a binary file
    outfile.put(0);						// id length (field 1)
    outfile.put(0);						// color map type (field 2)
    outfile.put(2);						// image_type (field 3)
    outfile.put(0); outfile.put(0);		// color map field entry index (field 4)
    outfile.put(0); outfile.put(0);		// color map length (field 4)
    outfile.put(0);				// color map entry size (field 4)
    outfile.put(0); outfile.put(0);		// x origin (field 5)
    outfile.put(0); outfile.put(0);		// y origin (field 5)
    outfile.write((char*)&width, 2);		// image width (field 5)
    outfile.write((char*)&height, 2);		// image height (field 5)
    outfile.put(24);				// pixel depth (field 5)
    outfile.put(0);				// image descriptor (field 5)
    outfile.write(bytes, width * height * 3);		// write the image data
    outfile.close();				// close the file
}


//reading targa image array
std::vector<char> read_tga(std::string filename, int& width, int& height) {
    std::ifstream infile;
    infile.open(filename, std::ios::binary | std::ios::out);        // open the file for binary writing
    if (!infile.is_open()) {
        std::cout << "ERROR: Unable to open file " << filename << std::endl;
        return std::vector<char>();
    }
    char id_length;                                infile.get(id_length);                            // id length (field 1)
    char cmap_type;                                infile.get(cmap_type);                            // color map type (field 2)
    char image_type;                            infile.get(image_type);                        // image_type (field 3)
    char field_entry_a, field_entry_b;
    infile.get(field_entry_a);                infile.get(field_entry_b);                        // color map field entry index (field 4)
    char map_length_a, map_length_b;
    infile.get(map_length_a);                infile.get(map_length_b);                        // color map field entry index (field 4)
    char map_size;                                infile.get(map_size);                            // color map entry size (field 4)
    char origin_x_a, origin_x_b;
    infile.get(origin_x_a);                infile.get(origin_x_b);                        // x origin (field 5)
    char origin_y_a, origin_y_b;
    infile.get(origin_y_a);                infile.get(origin_y_b);                        // x origin (field 5)

    infile.read((char*)&width, 2);
    infile.read((char*)&height, 2);
    char pixel_depth;                            infile.get(pixel_depth);
    char descriptor;                            infile.get(descriptor);

    std::vector<char> bytes(width * height * 3);
    infile.read(&bytes[0], width * height * 3);

    infile.close();                    // close the file

    return bytes;
}




int main(int argc, char* argv[]) {
    
    if (argc != 3) {
        
        return 1;
    }



    std::string File_name(argv[1]);
    int sigma = atoi(argv[2]);

    std::cout << "Targa extension Filename is : " << File_name << std::endl;
    std::cout << "Value of Sigma is : " << sigma << std::endl;


    /*std::string File_name = "whirlpool.tga";
    int sigma = 20;*/

    int width = 0;		  //width of the image
    int height = 0;		  //height of the image
    int size = 0;		  // size of the image

    //Calculating kernel size
    int k = 6 * sigma;                         //Standard deviation sigma = 40 pixels,

    if (k % 2 == 0) k++;
    float miu = k / 2;




    //evaluating the Gaussian kernel by defining a float pointer
    float* gkernel = (float*)malloc(k * sizeof(float));

    for (int xi = 0; xi < k; xi++) {
        int u = 2 * sigma * sigma;
        gkernel[xi] = 1 / sqrt(u * (float)PI) * exp(-(xi - miu) * (xi - miu) / u);
    }




    // Get image array from targa into a char* array

    std::vector<char> targa_image = read_tga(File_name, width, height);
    std::cout << "image width: " << width << "image height: " << height << std::endl;
    size = width * height * 3;
    char* given_image = &targa_image[0];



    // read the targa file and store its values inside an array and return the pointer to that array to imagearray
    //Allocating space for pixels
    // allocating the output array for pixels after convolution along x axis
    int x_height = height;
    int x_width = width - k + 1;
    int x_size = x_height * x_width * 3;
    char* x_output = (char*)malloc(x_size * sizeof(char));


    // convolution along y axis
    // allocating the output array for pixels after convolution along y axis
    int y_height = x_height - k + 1;
    int y_width = x_width;
    int y_size = y_height * y_width * 3;
    char* y_output = (char*)malloc(y_size * sizeof(char));




    std::cout << "------------------------- CPU version -------------------------" << std::endl;
    std::cout << "Convolution on CPU is going to start ..........." << std::endl;

    clock_t start, finish;   //creating timer
    double time_difference;
    start = clock();         //starting CPU time

    // doing kernel convolvolution along x axis
    Convolution_x_host(x_output, given_image, gkernel, width, x_height, x_width, k);
    // doing kernel convolvolution along y axis
    Convolution_y_host(y_output, x_output, gkernel, x_width, y_height, y_width, k);

    finish = clock();  // time ends
    time_difference = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "It takes " << time_difference << " s to do the convolution on CPU" << std::endl;
    write_tga("CPU_X.tga", x_output, x_width, x_height);
    write_tga("CPU_Y.tga", y_output, y_width, y_height);
    std::cout << "......Convolution on CPU has completed ........." << std::endl;
    // --------------------- CPU ----------------- //






    // ------------------- GPU ------------------------ //




    std::cout << "------------------------- GPU version -------------------------" << std::endl;
    std::cout << "Convolution on GPU is going to start ........... " << std::endl;

    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));

    float* gkernel_gpu;
    char* image_gpu;
    char* gpu_output_x;
    char* gpu_output_y;
    char* image_gpu_x;
    

    // allocate memory for image, kernel, and two convoled outputs
    HANDLE_ERROR(cudaMalloc(&gkernel_gpu, k * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&image_gpu, size * sizeof(char)));
    HANDLE_ERROR(cudaMalloc(&gpu_output_x, x_size * sizeof(char)));
    HANDLE_ERROR(cudaMalloc(&gpu_output_y, y_size * sizeof(char)));
    HANDLE_ERROR(cudaMalloc(&image_gpu_x, size * sizeof(char)));
    // copy image and kernel from main memory to Device
    HANDLE_ERROR(cudaMemcpy(image_gpu, given_image, size * sizeof(char), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(gkernel_gpu, gkernel, k * sizeof(float), cudaMemcpyHostToDevice));

    size_t blockDim = sqrt(prop.maxThreadsPerBlock);
    dim3 threads(blockDim, blockDim);
    dim3 blocks(width / threads.x + 1, height / threads.y + 1);
   

    // creating timer for GPU
    cudaEvent_t c_start;
    cudaEvent_t c_stop;
    cudaEventCreate(&c_start);
    cudaEventCreate(&c_stop);
    cudaEventRecord(c_start, NULL);

    // convolving along x
    Convolution_x_on_device << < blocks, threads >> > (gpu_output_x, image_gpu, gkernel_gpu, width, x_height, x_width, k);
    // convolving along y
    Convolution_y_on_device << < blocks, threads >> > (gpu_output_y, gpu_output_x, gkernel_gpu, x_width, y_height, y_width, k);


    // copy convolved outputs from Device to main memory
    HANDLE_ERROR(cudaMemcpy(x_output, gpu_output_x, x_size * sizeof(char), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(y_output, gpu_output_y, y_size * sizeof(char), cudaMemcpyDeviceToHost));
   


    // timer ends
    cudaEventRecord(c_stop, NULL);
    cudaEventSynchronize(c_stop);
    float time_difference_gpu;
    cudaEventElapsedTime(&time_difference_gpu, c_start, c_stop);
    std::cout << "It takes " << time_difference_gpu << " ms to do the convolution on GPU" << std::endl;

    // output file
    write_tga("GPU_X.tga", x_output, x_width, x_height);
    write_tga("GPU_Y.tga", y_output, y_width, y_height);




    std::cout << "......Convolution on GPU has completed ........." << std::endl;




    // ------------------- GPU using shared memory------------------------ //


    std::cout << "------------------------- GPU version -------------------------" << std::endl;
    std::cout << "Convolution on GPU using shared memory is going to start ........... " << std::endl;


    // calculate the size of shared memory
    size_t sharedmemory = 3 * blockDim * (blockDim + k - 1) * sizeof(char);
    cout << "sharedmemory is: " << sharedmemory << endl;
    if (prop.sharedMemPerBlock < sharedmemory) {
        std::cout << "ERROR:  shared memory is insufficient " << std::endl;
        exit(1);
    }



    Convolution_x_shared_memory << <blocks, threads, sharedmemory >> > (gpu_output_x, image_gpu, gkernel_gpu, width, x_height, x_width, k);
    Convolution_y_shared_memory << <blocks, threads, sharedmemory >> > (gpu_output_y, gpu_output_x, gkernel_gpu, x_width, y_height, y_width, k);

    HANDLE_ERROR(cudaMemcpy(x_output, gpu_output_x, x_size * sizeof(char), cudaMemcpyDeviceToHost));  //copy the array  from device to main memory
    HANDLE_ERROR(cudaMemcpy(y_output, gpu_output_y, y_size * sizeof(char), cudaMemcpyDeviceToHost));  //copy the array from device to main memory


    //	 calculate the time and show
    cudaEventRecord(c_stop, NULL);
    cudaEventSynchronize(c_stop);
    cudaEventElapsedTime(&time_difference_gpu, c_start, c_stop);
    cout << "It takes " << time_difference_gpu << " ms to do the GPU based convolution using shared memory" << endl;


    write_tga("GPU_x_shared.tga", x_output, x_width, x_height);
    write_tga("GPU_y_shared.tga", y_output, y_width, y_height);



    std::cout << "Convolution on GPU using shared memory has completed  ........... " << std::endl;

    //-------------- GPU using shared memory------------ //








}
