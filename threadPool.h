#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <queue>
#include <functional>
#include <future>

#include <cassert>

/* Thread-safe tasks queue. Tasks are enqueued by thread pool
   and dequeued by worker threads. When queue is drained,
   no more new task will be executed. */

class TaskQueue
{
public:
    TaskQueue() noexcept;
    void enqueue(std::function<void()> task);
    std::function<void()> dequeue();
    void drain() noexcept;
    bool isDrained() const noexcept;
    bool isDraining() const noexcept { return draining; }

private:
    std::queue<std::function<void()>> queue;
    mutable std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> draining;
};

/* Worker thread runs continuously on dedicated processor thread.
   When a new task is submitted to the task queue, the thread is awakened
   and is considered active until the task is finished.
   Any exception that is thrown by the task is caught internally to avoid
   thread termination. */

class WorkerThread
{
    friend class ThreadPool;
    explicit WorkerThread(TaskQueue& tasks, std::atomic<int>& taskCount);

public:
    void operator()();

private:
    TaskQueue& tasks;
    std::atomic<int>& taskCount;
};

/* A thread pool implements a dispatching of asynchronous tasks as functions objects.
   Pool maintains multiple threads waiting for tasks to be allocated
   for concurrent execution by the supervising program. By maintaining a pool of threads,
   the model increases performance and avoids latency in execution due to frequent
   creation and destruction of threads for short-lived tasks. */

class ThreadPool
{
public:
    explicit ThreadPool();
    ~ThreadPool();
    template<class Fn, class... Args>
    [[nodiscard]] auto addTask(const char *description, Fn&& func, Args&&... args);
    template<class Fn, class... Args>
    [[nodiscard]] auto addTask(Fn&& func, Args&&... args);
    template<class Fn, class Int>
    void parallelFor(Int loopBegin, Int loopEnd, Fn&& range);
    void waitAllTasks(uint32_t sleepNanoseconds = 100) noexcept;

private:
    std::vector<std::thread> threads;
    TaskQueue tasks;
    std::atomic<int> asyncTaskCount;
    std::mutex waitMtx;
};

#include "threadPool.inl"
