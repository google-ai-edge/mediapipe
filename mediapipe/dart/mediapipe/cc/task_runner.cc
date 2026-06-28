#include "third_party/mediapipe/tasks/cc/core/task_runner.h"

// Inspired by: https://source.corp.google.com/piper///depot/google3/third_party/mediapipe/tasks/python/core/pybind/task_runner.cc;l=42;rcl=549487063

#ifdef __cplusplus
extern "C"
{
#endif

    TaskRunner task_runner_create(CalculatorGraphConfig graph_config)
    {
    }

    Map task_runner_process(TaskRunner runner, Map input_packets)
    {
    }

    void task_runner_send(TaskRunner runner, Map input_packets)
    {
    }

    void task_runner_send(TaskRunner runner, Map input_packets)
    {
    }

    void task_runner_close(TaskRunner runner)
    {
    }

    void task_runner_restart(TaskRunner runner)
    {
    }

    CalculatorGraphConfig task_runner_get_graph_config(TaskRunner runner)
    {
    }

#ifdef __cplusplus
}
#endif