# Framework Architecture

## Scheduling mechanics

Data processing in a MediaPipe graph occurs inside processing nodes defined as
[`CalculatorBase`] subclasses. The scheduling system decides when each
calculator should run.

Each graph has at least one **scheduler queue**. Each scheduler queue has
exactly one **executor**. Nodes are statically assigned to a queue (and
therefore to an executor). By default there is one queue, whose executor is a
thread pool with a number of threads based on the system’s capabilities.

Each node has a scheduling state, which can be *not ready*, *ready*, or
*running*. A readiness function determines whether a node is ready to run. This
function is invoked at graph initialization, whenever a node finishes running,
and whenever the state of a node’s inputs changes.

The readiness function used depends on the type of node. A node with no stream
inputs is known as a **source node**; source nodes are always ready to run,
until they tell the framework they have no more data to output, at which point
they are closed.

Non-source nodes are ready if they have inputs to process, and if those inputs
form a valid input set according to the conditions set by the node’s **input
policy** (discussed below). Most nodes use the default input policy, but some
nodes specify a different one.

Note: Because changing the input policy changes the guarantees the calculator’s
code can expect from its inputs, it is not generally possible to mix and match
calculators with arbitrary input policies. Thus a calculator that uses a special
input policy should be written for it, and declare it in its contract.

When a node becomes ready, a task is added to the corresponding scheduler queue,
which is a priority queue. The priority function is currently fixed, and takes
into account static properties of the nodes and their topological sorting within
the graph. For example, nodes closer to the output side of the graph have higher
priority, while source nodes have the lowest priority.

Each queue is served by an executor, which is responsible for actually running
the task by invoking the calculator’s code. Different executors can be provided
and configured; this can be used to customize the use of execution resources,
e.g. by running certain nodes on lower-priority threads.

## Timestamp Synchronization

MediaPipe graph execution is decentralized: there is no global clock, and
different nodes can process data from different timestamps at the same time.
This allows higher throughput via pipelining.

However, time information is very important for many perception workflows. Nodes
that receive multiple input streams generally need to coordinate them in some
way. For example, an object detector may output a list of boundary rectangles
from a frame, and this information may be fed into a rendering node, which
should process it together with the original frame.

Therefore, one of the key responsibilities of the MediaPipe framework is to
provide input synchronization for nodes. In terms of framework mechanics, the
primary role of a timestamp is to serve as a **synchronization key**.

Furthermore, MediaPipe is designed to support deterministic operations, which is
important in many scenarios (testing, simulation, batch processing, etc.), while
allowing graph authors to relax determinism where needed to meet real-time
constraints.

The two objectives of synchronization and determinism underlie several design
choices. Notably, the packets pushed into a given stream must have monotonically
increasing timestamps: this is not just a useful assumption for many nodes, but
it is also relied upon by the synchronization logic. Each stream has a
**timestamp bound**, which is the lowest possible timestamp allowed for a new
packet on the stream. When a packet with timestamp `T` arrives, the bound
automatically advances to `T+1`, reflecting the monotonic requirement. This
allows the framework to know for certain that no more packets with timestamp
lower than `T` will arrive.

## Input policies

Synchronization is handled locally on each node, using the input policy
specified by the node.

The default input policy, defined by [`DefaultInputStreamHandler`], provides
deterministic synchronization of inputs, with the following guarantees:

*   If packets with the same timestamp are provided on multiple input streams,
    they will always be processed together regardless of their arrival order in
    real time.

*   Input sets are processed in strictly ascending timestamp order.

*   No packets are dropped, and the processing is fully deterministic.

*   The node becomes ready to process data as soon as possible given the
    guarantees above.

Note: An important consequence of this is that if the calculator always uses the
current input timestamp when outputting packets, the output will inherently obey
the monotonically increasing timestamp requirement.

Warning: On the other hand, it is not guaranteed that an input packet will
always be available for all streams.

To explain how it works, we need to introduce the definition of a settled
timestamp. We say that a timestamp in a stream is *settled* if it lower than the
timestamp bound. In other words, a timestamp is settled for a stream once the
state of the input at that timestamp is irrevocably known: either there is a
packet, or there is the certainty that a packet with that timestamp will not
arrive.

Note: For this reason, MediaPipe also allows a stream producer to explicitly
advance the timestamp bound farther that what the last packet implies, i.e. to
provide a tighter bound. This can allow the downstream nodes to settle their
inputs sooner.

A timestamp is settled across multiple streams if it is settled on each of those
streams. Furthermore, if a timestamp is settled it implies that all previous
timestamps are also settled. Thus settled timestamps can be processed
deterministically in ascending order.

Given this definition, a calculator with the default input policy is ready if
there is a timestamp which is settled across all input streams and contains a
packet on at least one input stream. The input policy provides all available
packets for a settled timestamp as a single *input set* to the calculator.

One consequence of this deterministic behavior is that, for nodes with multiple
input streams, there can be a theoretically unbounded wait for a timestamp to be
settled, and an unbounded number of packets can be buffered in the meantime.
(Consider a node with two input streams, one of which keeps sending packets
while the other sends nothing and does not advance the bound.)

Therefore, we also provide for custom input policies: for example, splitting the
inputs in different synchronization sets defined by
[`SyncSetInputStreamHandler`], or avoiding synchronization altogether and
processing inputs immediately as they arrive defined by
[`ImmediateInputStreamHandler`].

## Flow control

There are two main flow control mechanisms. A backpressure mechanism throttles
the execution of upstream nodes when the packets buffered on a stream reach a
(configurable) limit defined by [`CalculatorGraphConfig::max_queue_size`]. This
mechanism maintains deterministic behavior, and includes a deadlock avoidance
system that relaxes configured limits when needed.

The second system consists of inserting special nodes which can drop packets
according to real-time constraints (typically using custom input policies)
defined by [`FlowLimiterCalculator`]. For example, a common pattern places a
flow-control node at the input of a subgraph, with a loopback connection from
the final output to the flow-control node. The flow-control node is thus able to
keep track of how many timestamps are being processed in the downstream graph,
and drop packets if this count hits a (configurable) limit; and since packets
are dropped upstream, we avoid the wasted work that would result from partially
processing a timestamp and then dropping packets between intermediate stages.

This calculator-based approach gives the graph author control of where packets
can be dropped, and allows flexibility in adapting and customizing the graph’s
behavior depending on resource constraints.

[`CalculatorBase`]: https://github.com/google/mediapipe/tree/master/mediapipe/framework/calculator_base.h
[`DefaultInputStreamHandler`]: https://github.com/google/mediapipe/tree/master/mediapipe/framework/stream_handler/default_input_stream_handler.h
[`SyncSetInputStreamHandler`]: https://github.com/google/mediapipe/tree/master/mediapipe/framework/stream_handler/sync_set_input_stream_handler.h
[`ImmediateInputStreamHandler`]: https://github.com/google/mediapipe/tree/master/mediapipe/framework/stream_handler/immediate_input_stream_handler.h
[`CalculatorGraphConfig::max_queue_size`]: https://github.com/google/mediapipe/tree/master/mediapipe/framework/calculator.proto
[`FlowLimiterCalculator`]: https://github.com/google/mediapipe/tree/master/mediapipe/calculators/core/flow_limiter_calculator.cc
