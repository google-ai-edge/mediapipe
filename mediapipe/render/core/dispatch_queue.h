#pragma once
#include <functional>
#include <memory>
#include <thread>

//#define TimerEnabled

class dispatch_queue
{
public:
    dispatch_queue(std::string name);
    ~dispatch_queue();

    void dispatch_async(std::function< void() > func);
    void dispatch_sync(std::function< void() > func);
#ifdef TimerEnabled
    void dispatch_after(int msec, std::function< void() > func);
#endif
    void dispatch_flush();
    bool isCurrent();
    
    dispatch_queue(dispatch_queue const&) = delete;
    dispatch_queue& operator =(dispatch_queue const&) = delete;

private:
    struct impl;
    std::unique_ptr< impl > m;
    std::thread::id thread_id;
};

