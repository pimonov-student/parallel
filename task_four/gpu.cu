#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cub/cub.cuh>

// calculate "a_new" matrix
__global__ void calculate_matrix(double* a, double* a_new, int size)
{
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    // out of array
    if (i * size + j >= size * size)
    {
        return;
    }
    // first and last rows, first and last columns
    if (i == 0 || i == size - 1 || j == 0 || j == size - 1)
    {
        return;
    }
    
    a_new[i * size + j] = 0.25 * (a[i * size + j - 1] +
                                  a[i * size + j + 1] +
                                  a[(i - 1) * size + j] +
                                  a[(i + 1) * size + j]);
}

// calculate substraction of matrices (errors, matrix of errors)
__global__ void calculate_error_matrix(double* a, double* a_new, double* err_matrix, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= size * size) return;

    err_matrix[i] = fabs(a_new[i] - a[i]);
}

// <program name> size tol iter_max by launching
int main(int argc, char** argv)
{
    // variables from cmd
    int size = atoi(argv[1]);
    double tol = atof(argv[2]);
    double iter_max = atof(argv[3]);

    int iter = 0;
    double step = 10.0 / size;
    double* err;

    // "corner" indices
    int up_left = 0;
    int up_right = size - 1;
    int down_left = size * size - size;
    int down_right = size * size - 1;

    // our "matrices"
    double* a;
    double* a_new;
    // for pinned memory using
    cudaMallocHost(&err, sizeof(double));
    cudaMallocHost(&a, size * size * sizeof(double));
    cudaMallocHost(&a_new, size * size * sizeof(double));

    // init error value
    *err = 1.0;

    // fill "corner" values
    a[up_left] = 10;
    a[up_right] = 20;
    a[down_right] = 30;
    a[down_left] = 20;
    a_new[up_left] = 10;
    a_new[up_right] = 20;
    a_new[down_right] = 30;
    a_new[down_left] = 20;   

    // transpose up_left and up_right
    for (int i = 1; i < up_right; ++i)
    {
        a[i] = a[i - 1] + step;
	    a_new[i] = a[i];
    }
    // transpose down_left and down_right
    for (int i = down_left + 1; i < down_right; ++i)
    {
        a[i] = a[i - 1] + step;
	    a_new[i] = a[i];
    }
    // transpose up_left and down_left, up_right and down_right
    for (int i = size; i < down_left; i += size)
    {
        a[i] = a[i - size] + step;
        a[i + size - 1] = a[i - 1] + step;
	    a_new[i] = a[i];
	    a_new[i + size - 1] = a[i + size - 1];
    }


    // cuda part

    // choose device
    cudaSetDevice(1);

    // device copy variables
    double* dev_a = NULL;
    double* dev_a_new = NULL;
    double* dev_temp = NULL;
    double* dev_err = NULL;
    // substraction of matrices
    double* err_matrix = NULL;
    // temporary storage for reduce (and its size)
    double* temp_storage = NULL;
    size_t temp_storage_size = 0;
    // template variables
    size_t thread_count = size > 1024 ? 1024 : size;
    size_t block_count = size / thread_count;
    dim3 blockDim(thread_count / 32, thread_count / 32);
    dim3 gridDim(block_count * 32, block_count * 32);

    // allocate memory on device
    cudaMalloc((void**)&dev_a, sizeof(double) * size * size);
    cudaMalloc((void**)&dev_a_new, sizeof(double) * size * size);
    cudaMalloc((void**)&dev_temp, sizeof(double*));
    cudaMalloc((void**)&err_matrix, sizeof(double) * size * size);
    cudaMalloc((void**)&dev_err, sizeof(double));

    // determine "temp_storage_size" with cub function and allocate memory
    cub::DeviceReduce::Max(temp_storage, temp_storage_size, err_matrix, dev_err, size * size);
    cudaMalloc((void**)&temp_storage, temp_storage_size);

    // copy data into device
    cudaMemcpy(dev_a, a, sizeof(double) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_a_new, a_new, sizeof(double) * size * size, cudaMemcpyHostToDevice);

    // graph etc variables
    char graph_created = 0;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    clock_t begin = clock();

    // main cycle
    while (*err > tol && iter < iter_max)
    {
        if (!graph_created)
        {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
            for (int i = 0; i < 100; ++i)
            {
                // calculate "a_new" matrix
                calculate_matrix<<<gridDim, blockDim, 0, stream>>>(dev_a, dev_a_new, size);
                // swap matrices
                dev_temp = dev_a;
                dev_a = dev_a_new;
                dev_a_new = dev_temp;
            }

            // calculate matrix of errors, then find max error and copy its value to "err" stored on CPU
            calculate_error_matrix<<<thread_count * block_count * block_count, thread_count, 0, stream>>>(dev_a, dev_a_new, err_matrix, size);
            cub::DeviceReduce::Max(temp_storage, temp_storage_size, err_matrix, dev_err, size * size, stream);

            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graph_created = 1;
        }
        else
        {
            iter += 100;
            cudaGraphLaunch(instance, stream);
            cudaMemcpyAsync(err, dev_err, sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    }

    clock_t end = clock();

    printf("%d:\t%-32.25lf\n", iter, *err);
    printf("Time:\t %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);

    cudaFree(dev_a);
    cudaFree(dev_a_new);
    cudaFree(err_matrix);
    cudaFree(dev_err);
    cudaFree(temp_storage);

    cudaFreeHost(a);
    cudaFreeHost(a_new);
    cudaFreeHost(err);

    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);

    return 0;
}
