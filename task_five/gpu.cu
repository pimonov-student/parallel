#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cub/cub.cuh>
#include <mpi.h>

// calculate first and/or last row(s)
__global__ void calculate_borders(double* a, double* a_new, int size, int rows_count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // out of array
    if (i >= size)
    {
        return;
    }
    // first and last elements
    if (i == 0 || i == size - 1)
    {
        return;
    }

    // upper border
    a_new[size + i] = 0.25 * (a[size + i - 1] +
                              a[size + i + 1] +
                              a[i] +
                              a[2 * size + i]);
    // lower border
    a_new[(rows_count - 2) * size + i] = 0.25 * (a[(rows_count - 2) * size + i - 1] +
                                                 a[(rows_count - 2) * size + i + 1] +
                                                 a[(rows_count - 3) * size + i] +
                                                 a[(rows_count - 1) * size + i]);
}

// calculate "a_new" matrix
__global__ void calculate_matrix(double* a, double* a_new, int size, int rows_count)
{
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    // out of array
    if (i * size + j >= size * size)
    {
        return;
    }
    // first and last rows, first and last columns
    if (i <= 1 || i >= rows_count - 2 || j == 0 || j == size - 1)
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


    // mpi part

    // mpi variables (rank is unique for each device)
    int rank;
    int devices_count;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &devices_count);

    int actual_devices_count;
    cudaGetDeviceCount(&actual_devices_count);

    if (devices_count > actual_devices_count)
    {
        printf("Error: wrong devices count\n");
        return -1;
    }

    // choose device (according to rank)
    cudaSetDevice(rank);

    // we need to split our grid on several parts
    // define those parts
    size_t rows_count = size / devices_count;
    size_t first_row = rows_count * rank;

    // all parts need additional rows, one for the first and the last parts, two for all other
    if (devices_count != 1)
    {
    	if (rank == 0 || rank == devices_count - 1)
    	{
        	rows_count += 1;
    	}
    	else
    	{
        	rows_count += 2;
    	}
    }


    // cuda part

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

    // now we have no need to allocate data for a whole array
    size_t size_to_allocate = size * rows_count;

    // allocate memory on device
    cudaMalloc((void**)&dev_a, sizeof(double) * size_to_allocate);
    cudaMalloc((void**)&dev_a_new, sizeof(double) * size_to_allocate);
    cudaMalloc((void**)&err_matrix, sizeof(double) * size_to_allocate);
    cudaMalloc((void**)&dev_temp, sizeof(double*));
    cudaMalloc((void**)&dev_err, sizeof(double));

    // determine "temp_storage_size" with cub function and allocate memory
    cub::DeviceReduce::Max(temp_storage, temp_storage_size, err_matrix, dev_err, size_to_allocate);
    cudaMalloc((void**)&temp_storage, temp_storage_size);

    // no need to copy whole array, only part of it
    size_t offset = rank == 0 ? 0 : size;
    size_t start_point_to_copy = first_row * size - offset;

    // copy data into device
    cudaMemcpy(dev_a, a + start_point_to_copy, sizeof(double) * size_to_allocate, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_a_new, a_new + start_point_to_copy, sizeof(double) * size_to_allocate, cudaMemcpyHostToDevice);


    // streams
    cudaStream_t stream;
    cudaStream_t calculation_stream;
    cudaStreamCreate(&stream);
    cudaStreamCreate(&calculation_stream);


    // grid and block sizes
    size_t thread_count = size > 1024 ? 1024 : size;
    size_t block_x = size / thread_count;
    size_t block_y = rows_count;

    dim3 blockDim(thread_count, 1);
    dim3 gridDim(block_x, block_y);


    clock_t begin = clock();

    // main cycle
    while (*err > tol && iter < iter_max)
    {
        iter += 100;

        for (int i = 0; i < 100; ++i)
        {
            // calculate borders
            calculate_borders<<<size, 1, 0, stream>>>(dev_a, dev_a_new, size, rows_count);
            cudaStreamSynchronize(stream);

            // calculate "a_new" matrix (no borders)
            calculate_matrix<<<gridDim, blockDim, 0, calculation_stream>>>(dev_a, dev_a_new, size, rows_count);

            // swap calculated borders

            // upper border
            if (rank != 0)
            {
                MPI_Sendrecv(dev_a_new + size, size, MPI_DOUBLE,
                             rank - 1, 0,
                             dev_a_new, size, MPI_DOUBLE,
                             rank - 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // lower border
            if (rank != devices_count - 1 && devices_count != 1)
            {
                MPI_Sendrecv(dev_a_new + (rows_count - 2) * size, size, MPI_DOUBLE,
                             rank + 1, 0,
                             dev_a_new + (rows_count - 1) * size, size, MPI_DOUBLE,
                             rank + 1, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            cudaStreamSynchronize(calculation_stream);

            // swap matrices
            dev_temp = dev_a;
            dev_a = dev_a_new;
            dev_a_new = dev_temp;
        }

        // calculate matrix of errors, then find max error and copy its value to "err" stored on CPU
        calculate_error_matrix<<<block_x * block_y, thread_count, 0, calculation_stream>>>(dev_a, dev_a_new, err_matrix, size);
        cub::DeviceReduce::Max(temp_storage, temp_storage_size, err_matrix, dev_err, size_to_allocate, calculation_stream);
        cudaStreamSynchronize(calculation_stream);
        MPI_Allreduce((void*)dev_err, (void*)dev_err, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        cudaMemcpyAsync(err, dev_err, sizeof(double), cudaMemcpyDeviceToHost, calculation_stream);
	cudaStreamSynchronize(stream);
    }

    clock_t end = clock();

    if (rank == 0)
    {
        printf("%d:\t%-32.25lf\n", iter, *err);
        printf("Time:\t %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
    }

    MPI_Finalize();

    cudaFree(dev_a);
    cudaFree(dev_a_new);
    cudaFree(err_matrix);
    cudaFree(dev_err);
    cudaFree(temp_storage);

    cudaFreeHost(a);
    cudaFreeHost(a_new);
    cudaFreeHost(err);

    cudaStreamDestroy(stream);
    cudaStreamDestroy(calculation_stream);

    return 0;
}
