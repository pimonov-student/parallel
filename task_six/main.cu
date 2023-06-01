#include <iostream>
#include <memory>
#include <cstdio>
#include <cmath>

#include <cublas_v2.h>

// sigmoid function
__global__ void sigmoid(float* data, size_t size)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    // out of array
    if (id >= size)
    {
        return;
    }

    data[id] = 1.0 / (1.0 + expf(-data[id]));
}

// fully connected layer class
class Linear
{
public:
    // constructor
    Linear(size_t input_size, size_t output_size, char* path, cublasHandle_t handle)
    {
        FILE* fin;
        float* buffer_for_weights;

        this->input_size = input_size;
        this->output_size = output_size;
        this->handle = handle;

        cudaMallocHost(&buffer_for_weights, sizeof(float) * input_size * output_size);
        cudaMalloc(&this->weights, sizeof(float) * input_size * output_size);
        cudaMalloc(&this->buffer_for_output, sizeof(float) * output_size);

        // read data from file in C way
        fin = std::fopen(path, "rb");
        std::fread(buffer_for_weights, sizeof(float), input_size * output_size, fin);
        std::fclose(fin);

        cudaMemcpy((void*)this->weights, (void*)buffer_for_weights, sizeof(float) * input_size * output_size, cudaMemcpyHostToDevice);

        cudaFreeHost(buffer_for_weights);
    }
    // destructor
    ~Linear()
    {
        cudaFree(this->weights);
        cudaFree(this->buffer_for_output);
    }

    // forward pass
    void forward(float* input, float** input_ptr)
    {
        // necessary variables for cublas function
        const float alpha = 1.0;
        const float beta = 0.0;

        cublasSgemv_v2(this->handle, CUBLAS_OP_T,
                       this->input_size, this->output_size,
                       &alpha, this->weights, this->input_size,
                       input, 1,
                       &beta, this->buffer_for_output, 1);
        
        *input_ptr = this->buffer_for_output;
    }

    size_t get_output_size()
    {
        return this->output_size;
    }
private:
    size_t input_size;
    size_t output_size;
    float* weights;
    float* buffer_for_output;
    cublasHandle_t handle;
};

class Net
{
public:
    // constructor
    Net(size_t* sizes, char** paths)
    {
        cublasCreate(&handle);

        cudaMallocHost(&input, sizeof(float) * sizes[0]);
        cudaMalloc(&dev_input, sizeof(float) * sizes[0]);

        this->fc1 = std::make_unique<Linear>(sizes[0], sizes[1], paths[0], handle);
        this->fc2 = std::make_unique<Linear>(sizes[1], sizes[2], paths[1], handle);
        this->fc3 = std::make_unique<Linear>(sizes[2], sizes[3], paths[2], handle);
    }
    // destructor
    ~Net()
    {
        cublasDestroy(handle);
        cudaFreeHost(input);
        cudaFree(dev_input);
    }

    void read_input(char* path)
    {
        // reading inputs
        FILE* fin;
        fin = std::fopen(path, "rb");
        std::fread(input, sizeof(float), 32 * 32, fin);
        std::fclose(fin);

        // copy them to device
        cudaMemcpy(dev_input, input, sizeof(float) * 32 * 32, cudaMemcpyHostToDevice);
    }

    // forward pass
    void forward(float* output)
    {
        float* result = nullptr;
        size_t threads = 32;
        size_t blocks;

        this->fc1->forward(dev_input, &result);
        blocks = std::ceil((float)this->fc1->get_output_size() / threads);
        sigmoid<<<blocks, threads>>>(result, this->fc1->get_output_size());

        this->fc2->forward(result, &result);
        blocks = std::ceil((float)this->fc2->get_output_size() / threads);
        sigmoid<<<blocks, threads>>>(result, this->fc2->get_output_size());

        this->fc3->forward(result, &result);
        blocks = std::ceil((float)this->fc3->get_output_size() / threads);
        sigmoid<<<blocks, threads>>>(result, this->fc3->get_output_size());

        cudaMemcpy(output, result, sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << *output << std::endl;
    }
private:
    std::unique_ptr<Linear> fc1;
    std::unique_ptr<Linear> fc2;
    std::unique_ptr<Linear> fc3;
    float* input;
    float* dev_input;
    cublasHandle_t handle;
};

int main()
{
    size_t sizes[4] = { 32 * 32, 16 * 16, 4 * 4, 1 };
    char* paths[3] = { "./weights/weights_fc1.bin",
                       "./weights/weights_fc2.bin",
                       "./weights/weights_fc3.bin" };

    // output result
    float output;

    // network object
    Net net(sizes, paths);
    // reading input from .bin file
    net.read_input("./weights/weights_input.bin");

    // forward pass
    net.forward(&output);

    return 0;
}