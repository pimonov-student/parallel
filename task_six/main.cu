#include <iostream>
#include <cstdio>
#include <cmath>

#include <cublas_v2.h>

// sigmoid function
__global__ void actual_sigmoid(float* data, size_t size)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    // out of array
    if (id >= size)
    {
        return;
    }

    data[id] = 1.0 / (1.0 + expf(-data[id]));
}

__host__ void sigmoid(float* data, size_t size)
{
    // template sizes
    size_t threads = 32;
    size_t blocks = std::ceil((float)size / threads);

    actual_sigmoid<<<blocks, threads>>>(data, size);
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
    Net(size_t* sizes, char** paths, cublasHandle_t handle)
    {
        this->fc1 = &Linear(sizes[0], sizes[1], paths[0], handle);
        this->fc2 = &Linear(sizes[1], sizes[2], paths[1], handle);
        this->fc3 = &Linear(sizes[2], sizes[3], paths[2], handle);
    }
    // destructor
    ~Net() = default;

    // forward pass
    void forward(float* input, float* output)
    {
        float* result = nullptr;

        this->fc1->forward(input, &result);
        sigmoid(result, this->fc1->get_output_size());
        this->fc2->forward(result, &result);
        sigmoid(result, this->fc2->get_output_size());
        this->fc3->forward(result, &result);
        sigmoid(result, this->fc3->get_output_size());

        cudaMemcpy(output, result, sizeof(float), cudaMemcpyDeviceToHost);
    }
private:
    Linear* fc1;
    Linear* fc2;
    Linear* fc3;
};

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    size_t sizes[4] = { 32 * 32, 16 * 16, 4 * 4, 1 };
    char* paths[3] = { "./weights/weights_fc1.bin",
                       "./weights/weights_fc2.bin",
                       "./weights/weights_fc3.bin" };

    // input data
    float* input;
    float* dev_input;
    // output result
    float output;

    cudaMallocHost(&input, sizeof(float) * sizes[0]);
    cudaMalloc(&dev_input, sizeof(float) * sizes[0]);

    // reading inputs
    FILE* fin;
    fin = std::fopen("./weights/weights_input.bin", "rb");
    std::fread(input, sizeof(float), sizes[0], fin);
    std::fclose(fin);

    // copy them to device
    cudaMemcpy(dev_input, input, sizeof(float) * sizes[0], cudaMemcpyHostToDevice);

    // network object
    Net* net = new Net(sizes, paths, handle);

    // forward pass
    net->forward(dev_input, &output);

    std::cout << output << std::endl;

    delete net;
    cudaFreeHost(input);
    cudaFree(dev_input);

    return 0;
}
