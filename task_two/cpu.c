#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <time.h>

// <program name> size tol iter_max by launching
int main(int argc, char** argv)
{
    // variables from cmd
    int size = atoi(argv[1]);
    double tol = atof(argv[2]);
    double iter_max = atof(argv[3]);

    clock_t begin = clock();
    size_t num_of_bytes = sizeof(double) * size * size;
    int iter = 0;
    double step = 10.0 / size;
    double err = 1.0;

    // "corner" indices
    int up_left = 0;
    int up_right = size - 1;
    int down_left = size * size - size;
    int down_right = size * size - 1;

    // our "matrices"
    double* a = (double*)calloc(size * size, sizeof(double));
    double* a_new = (double*)calloc(size * size, sizeof(double*));

    // fill "corner" values
    a[up_left] = 10;
    a[up_right] = 20;
    a[down_right] = 30;
    a[down_left] = 20;

    // transpose up_left and up_right
    for (int i = 1; i < up_right; ++i)
    {
        a[i] = a[i - 1] + step;
    }
    // transpose down_left and down_right
    for (int i = down_left + 1; i < down_right; ++i)
    {
        a[i] = a[i - 1] + step;
    }
    // transpose up_left and down_left, up_right and down_right
    for (int i = size; i < down_left; i += size)
    {
        a[i] = a[i - size] + step;
        a[i + size - 1] = a[i - 1] + step;
    }

    // copy to a_new
    memcpy(a_new, a, num_of_bytes);

    // main cycle
    while (err > tol && iter < iter_max)
    {
        iter++;
        err = 0;

        for (int i = size; i < size * (size - 1); i += size)
        {
            for (int j = 1; j < size - 1; ++j)
            {
                a_new[i + j] = 0.25 * (a[i + j - 1] +
                                       a[i + j + 1] +
                                       a[i - size + j] +
                                       a[i + size + j]);
                err = fmax(err, a_new[i + j] - a[i + j]);
            }
        }

        memcpy(a, a_new, num_of_bytes);
    }

    clock_t end = clock();

    printf("%d:\t%-32.25lf\n", iter, err);
    printf("Time:\t %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);

    free(a);
    free(a_new);

    return 0;
}
