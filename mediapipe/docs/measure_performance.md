# Measuring Performance

*Coming soon.*

MediaPipe includes APIs for gathering aggregate performance data and
event timing data for CPU and GPU operations.  These API's can be found at:

<!-- TODO: Update the source code URL's to local or public URL's -->

   * [`GraphProfiler`](https://github.com/google/mediapipe/tree/master/mediapipe/framework/profiler/graph_profiler.h):
     Accumulates for each running calculator a histogram of latencies for
     Process calls.
   * [`GraphTracer`](https://github.com/google/mediapipe/tree/master/mediapipe/framework/profiler/graph_tracer.h):
     Records for each running calculator and each processed packet a series
     of timed events including the start and finish of each Process call.

Future mediapipe releases will include tools for visualizing and analysing
the latency histograms and timed events captured by these API's.
