#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef TYPE_FLOAT
#define TYPE float
#define FORMAT "%-32.25f"
#else
#define TYPE double
#define FORMAT "%-32.25lf"
#endif

int main()
{
	TYPE sum = 0.0;
	TYPE step = 2.0 * 3.1415926535 / 9999999.0;
	TYPE* arr = (TYPE*)malloc(sizeof(TYPE) * 10000000);

//#pragma acc data create(arr[:10000000])
{
	clock_t begin = clock();


#ifdef TYPE_FLOAT
//#pragma acc parallel loop vector vector_length(160) gang present(arr)
		for (int i = 0; i < 10000000; ++i)
		{
			arr[i] = sinf(i * step);
		}
#else
//#pragma acc parallel loop vector vector_length(160) gang present(arr)
		for (int i = 0; i < 10000000; ++i)
		{
			arr[i] = sin(i * step);
		}
#endif


//#pragma acc parallel loop reduction(+:sum) present(arr)
	for (int i = 0; i < 10000000; ++i)
	{
		sum += arr[i];
	}

	free(arr);

	clock_t end = clock();

	printf("Sum:\t ");
	printf(FORMAT, sum);
	printf("\nTime:\t %lf\n", (double)(end - begin) / CLOCKS_PER_SEC);
}

	return 0;
}
