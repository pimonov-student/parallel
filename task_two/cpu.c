#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>

// При запуске: size tol iter_max по порядку в командной строке
int main(int argc, char** argv)
{
    // Из командной строки
    int size = atoi(argv[1]);
    double tol = atof(argv[2]);
    double iter_max = atof(argv[3]);

    size_t num_of_bytes = sizeof(double) * size * size;
    int iter = 0;
    double step = 10.0 / size;
    double err = 1.0;

    // Вспомогательные "угловые" индексы
    int up_left = 0;
    int up_right = size - 1;
    int down_left = size * size - size;
    int down_right = size * size - 1;

    // Массивы, создаем и выделяем память
    double* a = (double*)calloc(size * size, sizeof(double));
    double* a_new = (double*)calloc(size * size, sizeof(double*));

    // Заполняем "угловые" пограничные значения
    a[up_left] = 10;
    a[up_right] = 20;
    a[down_right] = 30;
    a[down_left] = 20;

    // Интерполируем up_left и up_right
    for (int i = 1; i < up_right; ++i)
    {
        a[i] = a[i - 1] + step;
    }
    // Интерполируем down_left и down_right
    for (int i = down_left + 1; i < down_right; ++i)
    {
        a[i] = a[i - 1] + step;
    }
    // Интерполируем up_left и down_left, up_right и up_down
    for (int i = size; i < down_left; i += size)
    {
        a[i] = a[i - size] + step;
        a[i + size - 1] = a[i - 1] + step;
    }

    // Дублируем во вторую матрицу
    memcpy(a_new, a, num_of_bytes);

    // Основной рабочий цикл
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
                err = max(err, a_new[i + j] - a[i + j]);
            }
        }

        memcpy(a, a_new, num_of_bytes);

        if (iter % 100 == 0 || iter == 1)
        {
            printf("%d\t%lf\n", iter, err);
        }
    }

    free(a);
    free(a_new);

    return 0;
}