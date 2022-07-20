//
// Created by  jormin on 2021/6/23.
//

#ifndef OPIPE_GLTHREADDISPATCH_H
#define OPIPE_GLTHREADDISPATCH_H


#include "thread"


namespace Opipe {

    typedef std::function<void(void *id, std::function<void(void)> &func)> DispatchAsyncFunction;


    class GLThreadDispatch {

    public:
        GLThreadDispatch(std::thread::id glThreadId, DispatchAsyncFunction dispatchAsyncFunction);

        void runSync(void *host, std::function<void(void)> func);

        void runAsync(void *host, std::function<void(void)> func);

    private :
        DispatchAsyncFunction _dispatchAsync = nullptr;
        std::thread::id _glThreadId;
    };

}


#endif //OPIPE_GLTHREADDISPATCH_H
