#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main()
{
	double sum = 0;
	double two_pi = 2 * 3.1415926535;
	double step = two_pi / 9999999;
	double* arr = (double*)malloc(sizeof(double) * 10000000);

#pragma acc data create(arr[:10000000]) copy(sum) copyin(step)
{
#pragma acc parallel loop vector vector_length(160) gang
	for (int i = 0; i < 10000000; ++i)
	{
		arr[i] = sin(i * step);
	}

#pragma acc parallel loop vector vector_length(160) gang
	for (int i = 0; i < 10000000; ++i)
	{
		sum += arr[i];
	}
}
	printf("Sum:\t %-32.25lf\n", sum);

	free(arr);
	return 0;
}
