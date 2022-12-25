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

float sumAll(const std::vector<Matrix>& v) noexcept
{
    float sum = 0.f;
    for (const Matrix& m: v)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                sum += m.m[i][j];
    return sum;
}

uint64_t computeSingleThreaded(uint64_t count)
{
    std::vector<Matrix> a(ARRAY_SIZE);
    std::vector<Matrix> b(ARRAY_SIZE);
    std::vector<Matrix> c(ARRAY_SIZE);
    for (uint64_t i = 0ull; i < ARRAY_SIZE; ++i)
    {   // Initialize matrices
        a[i] = randomMatrix();
        b[i] = randomMatrix();
    }
    const uint64_t begin = rdtsc();
    {
        for (volatile uint64_t j = 0; j < count; ++j)
        {
            for (volatile uint64_t i = 0ull; i < ARRAY_SIZE; ++i)
                c[i] = multiply(a[i], b[i]);
        }
    }
    const uint64_t end = rdtsc();
    volatile float sum = sumAll(c);
    std::cout << "sum: " << sum << std::endl;
    return end - begin;
}

uint64_t computeMultiThreaded(uint64_t count)
{
    std::unique_ptr<ThreadPool> threadPool = std::make_unique<ThreadPool>();
    std::vector<Matrix> a(ARRAY_SIZE);
    std::vector<Matrix> b(ARRAY_SIZE);
    std::vector<Matrix> c(ARRAY_SIZE);
    for (uint64_t i = 0ull; i < ARRAY_SIZE; ++i)
    {   // Initialize matrices
        a[i] = randomMatrix();
        b[i] = randomMatrix();
    }
    const uint64_t begin = rdtsc();
    {
        for (volatile uint64_t j = 0; j < count; ++j)
        {
            threadPool->parallelFor(0ull, ARRAY_SIZE,
                [&](const uint64_t begin, const uint64_t end)
                {
                    for (uint64_t i = begin; i < end; ++i)
                        c[i] = multiply(a[i], b[i]);
                });
            threadPool->waitAllTasks();
        }
    }
    const uint64_t end = rdtsc();
    volatile float sum = sumAll(c);
    std::cout << "sum: " << sum << std::endl;
    return end - begin;
}

int main()
{
    rng.seed(static_cast<std::mt19937::result_type>(clock()));
    std::cout << "Run computations single threaded..." << std::endl;
    const uint64_t clocksSt = computeSingleThreaded(REPEAT);
    std::cout << "clocks elasped: " << clocksSt << std::endl;
    std::cout << "Run computations multi threaded..." << std::endl;
    const uint64_t clocksMt = computeMultiThreaded(REPEAT);
    std::cout << "clocks elasped: " << clocksMt << std::endl;
    std::cout << "Boost factor: " << clocksSt / (double)clocksMt << std::endl;
    return 0;
}
