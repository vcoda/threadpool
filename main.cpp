#include <time.h>
#include <iostream>
#include <memory>
#include <random>
#include <numeric>
#include <vector>
#include "matrix.h"
#include "threadPool.h"

// Note: too large size of matrix array is cache unfriendly
// which results in matrix multiplication to be memory bound.
#define ARRAY_SIZE 100000ull
#define REPEAT 10000ull

static std::mt19937 rng;

uint64_t rdtsc() noexcept
{
    int reg[4];
    __cpuid(reg, 0); // Insert barrier
    return __rdtsc();
}

inline Matrix randomMatrix()
{
    constexpr float rcpmax = 1.f/(rng.max() - rng.min());
    Matrix m;
    for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
        m.m[i][j] = rng() * rcpmax;
    return m;
}

void generateMatrices(std::vector<Matrix>& a, volatile uint64_t arraySize)
{
    a.resize(arraySize);
    for (volatile uint64_t i = 0ull; i < arraySize; ++i)
        a[i] = randomMatrix();
}

#pragma optimize("", off)
void multiplyMatrices(Matrix *a, Matrix *b,
    volatile uint64_t begin, volatile uint64_t end) noexcept
{
    for (volatile uint64_t i = begin; i < end; ++i)
        b[i] = multiply(a[i], b[i]);
}
#pragma optimize("", on)

double sum(const Matrix *a, volatile uint64_t begin, volatile uint64_t end) noexcept
{
    double total = 0.0;
    for (uint64_t k = begin; k < end; ++k)
    {
        const Matrix& m = a[k];
        float sum = 0.f;
        for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            sum += m.m[i][j];
        total += sum;
    }
    return total;
}

uint64_t computeSingleThreaded(volatile uint64_t arraySize, volatile uint64_t repeat)
{
    std::vector<Matrix> a, b;
    generateMatrices(a, arraySize);
    generateMatrices(b, arraySize);
    std::vector<double> sums(repeat);
    const uint64_t begin = rdtsc();
    for (volatile uint64_t i = 0; i < repeat; ++i)
    {
        multiplyMatrices(a.data(), b.data(), 0ull, arraySize);
        sums[i] = sum(b.data(), 0ull, arraySize);
        if (i && (i % 20) == 0) // Regenerate periodically to avoid inf
            generateMatrices(b, arraySize);
    }
    const uint64_t end = rdtsc();
    std::cout << "sum: " << std::accumulate(sums.begin(), sums.end(), 0.) << std::endl;
    return end - begin;
}

uint64_t computeMultiThreaded(volatile uint64_t arraySize, volatile uint64_t repeat)
{
    std::unique_ptr<ThreadPool> threadPool = std::make_unique<ThreadPool>();
    std::vector<Matrix> a, b;
    generateMatrices(a, arraySize);
    generateMatrices(b, arraySize);
    std::vector<double> sums(repeat);
    const uint64_t begin = rdtsc();
    for (volatile uint64_t i = 0; i < repeat; ++i)
    {
        double rangeSum[128];
        std::atomic<uint32_t> sumId = 0;
        threadPool->parallelFor(0ull, arraySize,
            [&](uint64_t begin, uint64_t end)
            {
                multiplyMatrices(a.data(), b.data(), begin, end);
                rangeSum[sumId++] = sum(b.data(), begin, end);
            });
        threadPool->waitAllTasks(0);
        sums[i] = std::accumulate(rangeSum, rangeSum + sumId, 0.);
        if (i && (i % 20) == 0) // Regenerate periodically to avoid inf
            generateMatrices(b, arraySize);
    }
    const uint64_t end = rdtsc();
    std::cout << "sum: " << std::accumulate(sums.begin(), sums.end(), 0.) << std::endl;
    return end - begin;
}

int main()
{
    volatile uint64_t arraySize = ARRAY_SIZE;
    volatile uint64_t repeat = REPEAT;
    rng.seed(static_cast<std::mt19937::result_type>(clock()));
    std::cout << "Run computations single threaded..." << std::endl;
    const uint64_t clocksSt = computeSingleThreaded(arraySize, repeat);
    std::cout << "clocks elasped: " << clocksSt << std::endl;
    std::cout << "Run computations multi threaded..." << std::endl;
    const uint64_t clocksMt = computeMultiThreaded(arraySize, repeat);
    std::cout << "clocks elasped: " << clocksMt << std::endl;
    std::cout << "Boost factor: " << clocksSt / (double)clocksMt << std::endl;
    return 0;
}
