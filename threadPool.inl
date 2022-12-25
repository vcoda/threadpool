template<class Fn, class... Args>
[[nodiscard]] auto ThreadPool::addTask(const char *description, Fn&& func, Args&&... args)
{
    assert(!tasks.isDraining());
    using InvokeResult = std::invoke_result_t<Fn, Args...>;
    auto asyncTask = std::make_shared<std::packaged_task<InvokeResult()>>(
        std::bind(std::forward<Fn>(func), std::forward<Args>(args)...));
    std::unique_lock<std::mutex> lock(waitMtx);
    tasks.enqueue(
        [asyncTask, description]
        {   // Wrap into std::function()
            (*asyncTask)();
        });
    return asyncTask.get_future();
}

template<class Fn, class... Args>
[[nodiscard]] auto ThreadPool::addTask(Fn&& func, Args&&... args)
{
    return addTask("", std::move(func), std::move(args)...);
}

template<class Fn, class Int>
void ThreadPool::parallelFor(Int loopBegin, Int loopEnd, Fn&& range)
{
    const Int threadCound = static_cast<Int>(threads.size());
    const Int loopLength = loopEnd - loopBegin;
    const Int rangeLength = loopLength / threadCound;
    const Int remainder = loopLength % threadCound;
    for (Int begin = loopBegin, end = rangeLength + remainder;
        end <= loopEnd;
        begin = end, end += rangeLength)
    {
        tasks.enqueue(
            [range, begin, end]
            {   // Wrap into std::function()
                range(begin, end);
            });
    }
}
