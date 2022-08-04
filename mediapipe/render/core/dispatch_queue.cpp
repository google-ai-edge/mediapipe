#include "dispatch_queue.h"

#include <atomic>
#include <thread>
#include <queue>
#include <deque>
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <string>
#if defined(__APPLE__)
#include "GPUImageMacros.h"
#else
#include "GPUImageMacros.h"
#endif
#if defined(__APPLE__)
#include <pthread.h>
#else
#include <sys/prctl.h>
#endif

using time_point = std::chrono::steady_clock::time_point;

struct work_entry
{
    explicit work_entry(std::function< void() > func_)
        : func(std::move(func_))
#ifdef TimerEnabled
        , expiry(time_point())
        , from_timer(false)
#endif
    {
    }

    work_entry(std::function< void() > func_, time_point expiry_)
        : func(std::move(func_))
#ifdef TimerEnabled
        , expiry(expiry_)
        , from_timer(true)
#endif
    {
    }

    std::function< void() > func;
#ifdef TimerEnabled
    time_point expiry;
    bool from_timer;
#endif
};

#ifdef TimerEnabled
bool operator >(work_entry const &lhs, work_entry const &rhs)
{
    return lhs.expiry > rhs.expiry;
}
#endif

struct dispatch_queue::impl
{
    impl(std::string name);
    std::string name;
    
    static void dispatch_thread_proc(impl *self);
#ifdef TimerEnabled
    static void timer_thread_proc(impl *self);
#endif
    std::mutex work_queue_mtx;
    std::condition_variable work_queue_cond;
    std::deque< work_entry > work_queue;

#ifdef TimerEnabled
    std::mutex timer_mtx;
    std::condition_variable timer_cond;
    std::priority_queue< work_entry, std::vector< work_entry >, std::greater< work_entry > > timers;
#endif
    std::thread work_queue_thread;
#ifdef TimerEnabled
    std::thread timer_thread;
#endif
    std::atomic< bool > quit;
    std::atomic< bool > work_queue_thread_started;
#ifdef TimerEnabled
    std::atomic< bool > timer_thread_started;
#endif
    using work_queue_lock = std::unique_lock< decltype(work_queue_mtx) >;
#ifdef TimerEnabled
    using timer_lock = std::unique_lock< decltype(timer_mtx) >;
#endif
};

void dispatch_queue::impl::dispatch_thread_proc(dispatch_queue::impl *self)
{
#if defined(__APPLE__)
    pthread_setname_np(self->name.c_str());
#else
    prctl(PR_SET_NAME, (unsigned long)self->name.c_str());
#endif
    work_queue_lock work_queue_lock(self->work_queue_mtx);
    self->work_queue_cond.notify_one();
    self->work_queue_thread_started = true;

    while (self->quit == false)
    {
        self->work_queue_cond.wait(work_queue_lock, [&] { return !self->work_queue.empty(); });

        while (!self->work_queue.empty()) {
            auto work = self->work_queue.back();
            self->work_queue.pop_back();

            work_queue_lock.unlock();
#if defined(__APPLE__)
            @autoreleasepool {
                work.func();
            }
#else
            work.func();
#endif
            work_queue_lock.lock();
        }
    }
}

#ifdef TimerEnabled
void dispatch_queue::impl::timer_thread_proc(dispatch_queue::impl *self)
{
    timer_lock timer_lock(self->timer_mtx);
    self->timer_cond.notify_one();
    self->timer_thread_started = true;

    while (self->quit == false)
    {
        if (self->timers.empty()) {
            self->timer_cond.wait(timer_lock, [&] { return self->quit || !self->timers.empty(); });
        }

        while (!self->timers.empty())
        {
            auto const& work = self->timers.top();
            if (self->timer_cond.wait_until(timer_lock, work.expiry, [&] { return self->quit.load(); })) {
                break;
            }

            {
                work_queue_lock _(self->work_queue_mtx);
                auto where = std::find_if(self->work_queue.rbegin(),
                                          self->work_queue.rend(),
                                          [] (work_entry const &w) { return !w.from_timer; });
                self->work_queue.insert(where.base(), work);
                self->timers.pop();
                self->work_queue_cond.notify_one();
            }
        }
    }
}
#endif

dispatch_queue::impl::impl(std::string name)
    : quit(false)
    , name(name)
    , work_queue_thread_started(false)
#ifdef TimerEnabled
    , timer_thread_started(false)
#endif
{
    work_queue_lock work_queue_lock(work_queue_mtx);
#ifdef TimerEnabled
    timer_lock timer_lock(timer_mtx);
#endif

    work_queue_thread = std::thread(dispatch_thread_proc, this);
#ifdef TimerEnabled
    timer_thread = std::thread(timer_thread_proc, this);
#endif

    work_queue_cond.wait(work_queue_lock, [this] { return work_queue_thread_started.load(); });
#ifdef TimerEnabled
    timer_cond.wait(timer_lock, [this] { return timer_thread_started.load(); });
#endif
}

dispatch_queue::dispatch_queue(std::string name) : m(new impl(name))
{
    thread_id = m->work_queue_thread.get_id();
}

dispatch_queue::~dispatch_queue()
{
    dispatch_async([this] { m->quit = true; });
    m->work_queue_thread.join();
#ifdef TimerEnabled
    {
        impl::timer_lock _(m->timer_mtx);
        m->timer_cond.notify_one();
    }
    m->timer_thread.join();
#endif
    
}

void dispatch_queue::dispatch_async(std::function< void() > func)
{
    impl::work_queue_lock _(m->work_queue_mtx);
    m->work_queue.push_front(work_entry(func));
    m->work_queue_cond.notify_one();
}

void dispatch_queue::dispatch_sync(std::function< void() > func)
{
    std::mutex sync_mtx;
    impl::work_queue_lock work_queue_lock(sync_mtx);
    std::condition_variable sync_cond;
    std::atomic< bool > completed(false);

    {
        impl::work_queue_lock _(m->work_queue_mtx);
        m->work_queue.push_front(work_entry(func));
        m->work_queue.push_front(work_entry([&] {
            std::unique_lock< decltype(sync_mtx) > sync_cb_lock(sync_mtx);
            completed = true;
            sync_cond.notify_one();
        }));

        m->work_queue_cond.notify_one();
    }

    sync_cond.wait(work_queue_lock, [&] { return completed.load(); });
}

#ifdef TimerEnabled
void dispatch_queue::dispatch_after(int msec, std::function< void() > func)
{
    impl::timer_lock _(m->timer_mtx);
    m->timers.push(work_entry(func, std::chrono::steady_clock::now() + std::chrono::milliseconds(msec)));
    m->timer_cond.notify_one();
}
#endif

void dispatch_queue::dispatch_flush()
{
    dispatch_sync([]{});
}

bool dispatch_queue::isCurrent()
{
    bool result = std::this_thread::get_id() == thread_id;
    return result;
}

