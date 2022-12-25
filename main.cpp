#include <time.h>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include "matrix.h"
#include "threadPool.h"

#define ARRAY_SIZE 10000000ull
#define REPEAT 10

static std::mt19937 rng;

uint64_t rdtsc()
{
    int reg[4];
    __cpuid(reg, 0); // Insert barrier
    return __rdtsc();
}

Matrix randomMatrix()
{
    Matrix m;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            float f = rng() / (float)rng.max();
            m.m[i][j] = f;
        }
    return m;
}

void generateMatrices(std::vector<Matrix>& a, volatile uint64_t arraySize)
{
    a.resize(arraySize);
    for (volatile uint64_t i = 0ull; i < arraySize; ++i)
        a[i] = randomMatrix();
}

void multiplyMatrices(std::vector<Matrix>& a, std::vector<Matrix>& b,
    volatile uint64_t begin, volatile uint64_t end) noexcept
{
    for (volatile uint64_t i = begin; i < end; ++i)
        b[i] = multiply(a[i], b[i]);
}

float sumAll(const std::vector<Matrix>& v) noexcept
{
    float sum = 0.f;
    for (const Matrix& m: v)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                sum += m.m[i][j];
    return sum;
}

void computeSingleThreaded(volatile uint64_t arraySize, volatile uint64_t repeat)
{
    std::vector<Matrix> a, b;
    generateMatrices(a, arraySize);
    generateMatrices(b, arraySize);
    for (volatile uint64_t j = 0; j < repeat; ++j)
    {
        multiplyMatrices(a, b, 0ull, arraySize);
    }
    std::cout << "sum: " << sumAll(b) << std::endl;
}

void computeMultiThreaded(volatile uint64_t arraySize, volatile uint64_t repeat)
{
    std::unique_ptr<ThreadPool> threadPool = std::make_unique<ThreadPool>();
    std::vector<Matrix> a, b;
    generateMatrices(a, arraySize);
    generateMatrices(b, arraySize);
    for (volatile uint64_t j = 0; j < repeat; ++j)
    {
        threadPool->parallelFor(0ull, arraySize,
            [&](uint64_t begin, uint64_t end)
            {
                multiplyMatrices(a, b, begin, end);
            });
        threadPool->waitAllTasks();
    }
    std::cout << "sum: " << sumAll(b) << std::endl;
}

int main()
{
    volatile uint64_t arraySize = ARRAY_SIZE;
    volatile uint64_t repeat = REPEAT;
    rng.seed(static_cast<std::mt19937::result_type>(clock()));
    std::cout << "Run computations single threaded..." << std::endl;
    uint64_t begin = rdtsc();
    {
        computeSingleThreaded(arraySize, repeat);
    }
    uint64_t end = rdtsc();
    const uint64_t clocksSt = end - begin;
    std::cout << "clocks elasped: " << clocksSt << std::endl;
    std::cout << "Run computations multi threaded..." << std::endl;
    begin = rdtsc();
    {
        computeMultiThreaded(arraySize, repeat);
    }
    end = rdtsc();
    const uint64_t clocksMt = end - begin;
    std::cout << "clocks elasped: " << clocksMt << std::endl;
    std::cout << "Boost factor: " << clocksSt / (double)clocksMt << std::endl;
    return 0;
}
