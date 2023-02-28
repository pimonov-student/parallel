#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

int     size;
double   tol;
double   iter_max;

// При запуске: size tol iter_max по порядку в командной строке
int main(int argc, char** argv)
{
    // Из командной строки
    size        = atoi(argv[1]);
    tol         = atod(argv[2]);
    iter_max    = atod(argv[3]);

    int iter = 0;
    double step_10_20 = (10 + 20) / size;
    double step_20_30 = (20 + 30) / size;
    double err;

    // Массивы, создаем и выделяем память
    double** a = (double**)malloc(size * sizeof(double*));
    double** a_new = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; ++i)
    {
        a[i] = (double*)calloc(size, sizeof(double));
        a_new[i] = (double*)calloc(size, sizeof(double));
    }

    // Заполняем "угловые" пограничные значения
    a[0][0]                 = a_new[0][0]               = 10;
    a[0][size - 1]          = a_new[0][size - 1]        = 20;
    a[size - 1][size - 1]   = a_new[size - 1][size - 1] = 30;
    a[size - 1][0]          = a_new[size - 1][0]        = 20;

    // Интерполируем
    for (int i = 1; i < size - 1; ++i)
    {
        a[0][i] = a[0][i - 1] + step_10_20;
        a[i][0] = a[i - 1][0] + step_10_20;
        a[size - 1][i] = 
    }

    // Основной рабочий цикл
    while (err > tol && iter < iter_max)
    {
        iter++;
        err = 1.0;

        for (int j = 0; j < size; ++j)
        {
            for (int i = 0; i < size; ++i)
            {
                a_new[i][j] = 0.25 * (a[i + 1][j] + 
                                      a[i - 1][j] + 
                                      a[i][j - 1] + 
                                      a[i][j + 1]);
                err = max(err, a_new[i][j] - a[i][j]);
            }
        }


    }

    // Очищаем память из под массивов
    for (int i = 0; i < size; ++i)
    {
        free(a[i]);
        free(a_new[i]);
    }
    free(a);
    free(a_new);

    return 0;
}