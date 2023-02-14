#include <iostream>
#include <cmath>
#include <time.h>

double two_pi = 2 * 3.14159265358979323846;
double step = two_pi / 9999999;

int main()
{
	double sum = 0;
	double* arr = new double[10000000];
	clock_t begin;

	begin = clock();
//#pragma acc data copy(arr[0:10000000])
//#pragma acc kernels
	for (int i = 0; i < 10000000; ++i)
	{
		arr[i] = sin(i * step);
	}
	printf("Time to fill:\t %lf\n", (double)(clock() - begin) / CLOCKS_PER_SEC);

	begin = clock();
//#pragma acc kernels
	for (int i = 0; i < 10000000; ++i)
	{
		sum += arr[i];
	}
	printf("Time to sum:\t %lf\n", (double)(clock() - begin) / CLOCKS_PER_SEC);

	printf("Sum:\t %-32.25lf\n", sum);

	delete[] arr;

	return 0;
}
