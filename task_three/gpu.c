#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cublas_v2.h>

// <program name> size tol iter_max by launching
int main(int argc, char** argv)
{
    // variables from cmd
    int size = atoi(argv[1]);
    double tol = atof(argv[2]);
    double iter_max = atof(argv[3]);

    int iter = 0;
    double step = 10.0 / size;
    double err = 1.0;

    int err_index = 0;
    const double alpha_pos = 1.0;
    const double alpha_neg = -1.0;

    // "corner" indices
    int up_left = 0;
    int up_right = size - 1;
    int down_left = size * size - size;
    int down_right = size * size - 1;

    // our "matrices" and temp variable for swap
    double* temp;
    double* a = (double*)calloc(size * size, sizeof(double));
    double* a_new = (double*)calloc(size * size, sizeof(double));

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

    cublasHandle_t handle;
    cublasCreate(&handle);

    clock_t begin = clock();

#pragma acc data copyin(a[:size*size], a_new[:size*size], alpha_pos, alpha_neg, err, err_index)
    {
    // main cycle
    while (err > tol && iter < iter_max)
    {
        iter++;

#pragma acc data present(a, a_new)
#pragma acc parallel loop collapse(2)
        for (int i = size; i < size * (size - 1); i += size)
        {
            for (int j = 1; j < size - 1; ++j)
            {
                a_new[i + j] = 0.25 * (a[i + j - 1] +
                                       a[i + j + 1] +
                                       a[i - size + j] +
                                       a[i + size + j]);
            }
        }

        if (iter % 100 == 0)
    	        {
#pragma acc host_data use_device(a, a_new)
	        {
	            cublasDaxpy(handle, size * size, &alpha_neg, a, 1, a_new, 1);
	            cublasIdamax(handle, size * size, a_new, 1, &err_index);
	            cublasDaxpy(handle, size * size, &alpha_pos, a, 1, a_new, 1);

#pragma acc update device(err_index)
	        }
	

#pragma acc kernels
	        {
	            err = 0;
	            err = fmax(err, fabs(a_new[err_index - 1] - a[err_index - 1]));
	        }

#pragma acc update host(err)
	        }

        temp = a;
	a = a_new;
	a_new = temp;
    }
    }

    clock_t end = clock();

    printf("%d:\t%-32.25lf\n", iter, err);
    printf("Time:\t %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);

    cublasDestroy(handle);
    free(a);
    free(a_new);

    return 0;
}
