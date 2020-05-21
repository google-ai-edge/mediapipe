# Profiler Configuration Settings

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'mhays' reviewed: '2020-05-08' }
*-->

[TOC]

The following settings are used when setting up [MediaPipe Tracing](tracer.md)
Many of them are advanced and not recommended for general usage. Consult
[MediaPipe Tracing](tracer.md) for a friendlier introduction.

histogram_interval_size_usec :Specifies the size of the runtimes histogram
intervals (in microseconds) to generate the histogram of the Process() time. The
last interval extends to +inf. If not specified, the interval is 1000000 usec =
1 sec.

num_histogram_intervals :Specifies the number of intervals to generate the
histogram of the `Process()` runtime. If not specified, one interval is used.

enable_profiler
:   If true, the profiler starts profiling when graph is initialized.

enable_stream_latency
:   If true, the profiler also profiles the stream latency and input-output
    latency. No-op if enable_profiler is false.

use_packet_timestamp_for_added_packet
:   If true, the profiler uses packet timestamp (as production time and source
    production time) for packets added by calling
    `CalculatorGraph::AddPacketToInputStream()`. If false, uses the profiler's
    clock.

trace_log_capacity
:   The maximum number of trace events buffered in memory. The default value
    buffers up to 20000 events.

trace_event_types_disabled
:   Trace event types that are not logged.

trace_log_path
:   The output directory and base-name prefix for trace log files. Log files are
    written to: StrCat(trace_log_path, index, "`.binarypb`")

trace_log_count
:   The number of trace log files retained. The trace log files are named
    "`trace_0.log`" through "`trace_k.log`". The default value specifies 2
    output files retained.

trace_log_interval_usec
:   The interval in microseconds between trace log output. The default value
    specifies trace log output once every 0.5 sec.

trace_log_margin_usec
:   The interval in microseconds between TimeNow and the highest times included
    in trace log output. This margin allows time for events to be appended to
    the TraceBuffer.

trace_log_duration_events
:   False specifies an event for each calculator invocation. True specifies a
    separate event for each start and finish time.

trace_log_interval_count
:   The number of trace log intervals per file. The total log duration is:
    `trace_log_interval_usec * trace_log_file_count * trace_log_interval_count`.
    The default value specifies 10 intervals per file.

trace_log_disabled
:   An option to turn ON/OFF writing trace files to disk. Saving trace files to
    disk is enabled by default.

trace_enabled
:   If true, tracer timing events are recorded and reported.
