#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main()
{
	float sum = 0.0;
	float step = 2.0 * 3.1415926535 / 9999999.0;
	float* arr = (float*)malloc(sizeof(float) * 10000000);

#pragma acc data create(arr[:10000000])
{
	clock_t begin = clock();

#pragma acc parallel loop vector vector_length(160) gang present(arr)
	for (int i = 0; i < 10000000; ++i)
	{
		arr[i] = sinf(i * step);
	}

#pragma acc parallel loop reduction(+:sum) present(arr)
	for (int i = 0; i < 10000000; ++i)
	{
		sum += arr[i];
	}

	free(arr);

	clock_t end = clock();

	printf("Sum:\t %-32.25f\n", sum);
	printf("Time:\t %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
}

	return 0;
}
