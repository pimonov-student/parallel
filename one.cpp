#include <iostream>
#include <cmath>
#include <time.h>

double two_pi = 2 * 3.14159265358979323846;
double step = two_pi / 9999999;

void fill_cpu(double* arr)
{
	for (int i = 0; i < 10000000; ++i)
	{
		arr[i] = sin(i * step);
	}
}

void fill_gpu(double* arr)
{
#pragma acc parallel
{
	for (int i = 0; i < 10000000; ++i)
	{
		arr[i] = sin(i * step);
	}
}
}

void sum_cpu(double* arr, double* sum)
{
	*sum = 0;
	for (int i = 0; i < 10000000; ++i)
	{
		*sum += arr[i];
	}
}

void sum_gpu(double* arr, double* sum)
{
	double tmp = 0;
	*sum = 0;
#pragma acc parallel
{
	for (int i = 0; i < 10000000; ++i)
	{
		tmp += arr[i];
	}
}
	*sum = tmp;
}

int main()
{
	double sum;
	double* arr = new double[10000000];
	clock_t begin;

#pragma acc data copy(arr[0:10000000])
#pragma acc data copy(sum)

	begin = clock();
	fill_cpu(arr);
	printf("Time fill_cpu:\t %lf\n", (double)(clock() - begin) / CLOCKS_PER_SEC);

	begin = clock();
	sum_cpu(arr, &sum);
	printf("Time sum_cpu:\t %lf\n\n", (double)(clock() - begin) / CLOCKS_PER_SEC);

	begin = clock();
	fill_gpu(arr);
	printf("Time fill_gpu:\t %lf\n", (double)(clock() - begin) / CLOCKS_PER_SEC);

	begin = clock();
	sum_gpu(arr, &sum);
	printf("Time sum_gpu:\t %lf\n\n", (double)(clock() - begin) / CLOCKS_PER_SEC);

	delete[] arr;

	return 0;
}
