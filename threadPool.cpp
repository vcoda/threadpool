#include "threadPool.h"
#include <cassert>
#include <iostream>

TaskQueue::TaskQueue() noexcept:
    draining(false)
{}

void TaskQueue::enqueue(std::function<void()> task)
{
    if (!draining)
    {
        std::unique_lock<std::mutex> lock(mtx);
        queue.push(task);
        cv.notify_one(); // Wake up one of the threads
    }
}

std::function<void()> TaskQueue::dequeue()
{
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock,
        [this]
        {   // Wait for a new task or queue drain
            return !queue.empty() || draining;
        });
    std::function<void()> task;
    if (!queue.empty())
    {
        task = queue.front();
        queue.pop();
    }
    return task;
}

void TaskQueue::drain() noexcept
{
    draining = true;
    cv.notify_all(); // Wake up all idle threads
}

bool TaskQueue::isDrained() const noexcept
{
    std::unique_lock<std::mutex> lock(mtx);
    return draining && queue.empty();
}

WorkerThread::WorkerThread(TaskQueue& tasks, std::atomic<int>& taskCount):
    tasks(tasks),
    taskCount(taskCount)
{}

void WorkerThread::operator()()
{   // Run continuously
    while (true)
    {   // Remain idle until got a new task
        std::function<void()> task = tasks.dequeue();
        if (task)
        {
            ++taskCount;
            try {
                task();
            } catch (const std::exception& error) {
                std::cout << error.what() << std::endl;
            }
            --taskCount;
        } else
        {   // If no more tasks and queue is draining
            if (tasks.isDraining())
            {   // Leave the thread
                break;
            }
        }
    }
}

ThreadPool::ThreadPool():
    asyncTaskCount(0)
{   // Get the number of concurrent threads supported by the implementation.
    const uint32_t hardwareThreadContexts = std::thread::hardware_concurrency();
    threads.reserve(hardwareThreadContexts);
    for (uint32_t i = 0; i < hardwareThreadContexts; ++i)
    {   // Spawn a new worker thread
        threads.emplace_back(std::thread(WorkerThread(tasks, asyncTaskCount)));
    }
}

ThreadPool::~ThreadPool()
{
    tasks.drain();
    for (auto& thread: threads)
        thread.join();
    assert(tasks.isDrained());
}

void ThreadPool::waitAllTasks(uint32_t sleepNanoseconds /* 100 */) noexcept
{
    const auto duration = std::chrono::nanoseconds(sleepNanoseconds);
    std::unique_lock<std::mutex> lock(waitMtx);
    while (asyncTaskCount > 0)
    {   // Blocks the execution of the current thread for at least
        // the specified <duration>. This function may block for longer
        // than <duration> due to scheduling or resource contention delays.
        if (sleepNanoseconds > 0)
            std::this_thread::sleep_for(duration);
    }
}
