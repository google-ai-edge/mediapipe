---
layout: forward
target: https://developers.google.com/mediapipe/framework/framework_concepts/graphs_cpp
title: Building Graphs in C++
parent: Graphs
nav_order: 1
---

# Building Graphs in C++
{: .no_toc }

1. TOC
{:toc}
---

**Attention:** *Thanks for your interest in MediaPipe! We have moved to
[https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)
as the primary developer documentation site for MediaPipe as of April 3, 2023.*

----

C++ graph builder is a powerful tool for:

*   Building complex graphs
*   Parametrizing graphs (e.g. setting a delegate on `InferenceCalculator`,
    enabling/disabling parts of the graph)
*   Deduplicating graphs (e.g. instead of CPU and GPU dedicated graphs in pbtxt
    you can have a single code that constructs required graphs, sharing as much
    as possible)
*   Supporting optional graph inputs/outputs
*   Customizing graphs per platform

## Basic Usage

Let's see how C++ graph builder can be used for a simple graph:

```proto
# Graph inputs.
input_stream: "input_tensors"
input_side_packet: "model"

# Graph outputs.
output_stream: "output_tensors"

node {
  calculator: "InferenceCalculator"
  input_stream: "TENSORS:input_tensors"
  input_side_packet: "MODEL:model"
  output_stream: "TENSORS:output_tensors"
  node_options: {
    [type.googleapis.com/mediapipe.InferenceCalculatorOptions] {
      # Requesting GPU delegate.
      delegate { gpu {} }
    }
  }
}
```

Function to build the above `CalculatorGraphConfig` may look like:

```c++
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Graph inputs.
  Stream<std::vector<Tensor>> input_tensors =
      graph.In(0).SetName("input_tensors").Cast<std::vector<Tensor>>();
  SidePacket<TfLiteModelPtr> model =
      graph.SideIn(0).SetName("model").Cast<TfLiteModelPtr>();

  auto& inference_node = graph.AddNode("InferenceCalculator");
  auto& inference_opts =
      inference_node.GetOptions<InferenceCalculatorOptions>();
  // Requesting GPU delegate.
  inference_opts.mutable_delegate()->mutable_gpu();
  input_tensors.ConnectTo(inference_node.In("TENSORS"));
  model.ConnectTo(inference_node.SideIn("MODEL"));
  Stream<std::vector<Tensor>> output_tensors =
      inference_node.Out("TENSORS").Cast<std::vector<Tensor>>();

  // Graph outputs.
  output_tensors.SetName("output_tensors").ConnectTo(graph.Out(0));

  // Get `CalculatorGraphConfig` to pass it into `CalculatorGraph`
  return graph.GetConfig();
}
```

Short summary:

*   Use `Graph::In/SideIn` to get graph inputs as `Stream/SidePacket`
*   Use `Node::Out/SideOut` to get node outputs as `Stream/SidePacket`
*   Use `Stream/SidePacket::ConnectTo` to connect streams and side packets to
    node inputs (`Node::In/SideIn`) and graph outputs (`Graph::Out/SideOut`)
    *   There's a "shortcut" operator `>>` that you can use instead of
        `ConnectTo` function (E.g. `x >> node.In("IN")`).
*   `Stream/SidePacket::Cast` is used to cast stream or side packet of `AnyType`
    (E.g. `Stream<AnyType> in = graph.In(0);`) to a particular type
    *   Using actual types instead of `AnyType` sets you on a better path for
        unleashing graph builder capabilities and improving your graphs
        readability.

## Advanced Usage

### Utility Functions

Let's extract inference construction code into a dedicated utility function to
help for readability and code reuse:

```c++
// Updates graph to run inference.
Stream<std::vector<Tensor>> RunInference(
    Stream<std::vector<Tensor>> tensors, SidePacket<TfLiteModelPtr> model,
    const InferenceCalculatorOptions::Delegate& delegate, Graph& graph) {
  auto& inference_node = graph.AddNode("InferenceCalculator");
  auto& inference_opts =
      inference_node.GetOptions<InferenceCalculatorOptions>();
  *inference_opts.mutable_delegate() = delegate;
  tensors.ConnectTo(inference_node.In("TENSORS"));
  model.ConnectTo(inference_node.SideIn("MODEL"));
  return inference_node.Out("TENSORS").Cast<std::vector<Tensor>>();
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Graph inputs.
  Stream<std::vector<Tensor>> input_tensors =
      graph.In(0).SetName("input_tensors").Cast<std::vector<Tensor>>();
  SidePacket<TfLiteModelPtr> model =
      graph.SideIn(0).SetName("model").Cast<TfLiteModelPtr>();

  InferenceCalculatorOptions::Delegate delegate;
  delegate.mutable_gpu();
  Stream<std::vector<Tensor>> output_tensors =
      RunInference(input_tensors, model, delegate, graph);

  // Graph outputs.
  output_tensors.SetName("output_tensors").ConnectTo(graph.Out(0));

  return graph.GetConfig();
}
```

As a result, `RunInference` provides a clear interface stating what are the
inputs/outputs and their types.

It can be easily reused, e.g. it's only a few lines if you want to run an extra
model inference:

```c++
  // Run first inference.
  Stream<std::vector<Tensor>> output_tensors =
      RunInference(input_tensors, model, delegate, graph);
  // Run second inference on the output of the first one.
  Stream<std::vector<Tensor>> extra_output_tensors =
      RunInference(output_tensors, extra_model, delegate, graph);
```

And you don't need to duplicate names and tags (`InferenceCalculator`,
`TENSORS`, `MODEL`) or introduce dedicated constants here and there - those
details are localized to `RunInference` function.

Tip: extracting `RunInference` and similar functions to dedicated modules (e.g.
inference.h/cc which depends on the inference calculator) enables reuse in
graphs construction code and helps automatically pull in calculator dependencies
(e.g. no need to manually add `:inference_calculator` dep, just let your IDE
include `inference.h` and build cleaner pull in corresponding dependency).

### Utility Classes

And surely, it's not only about functions, in some cases it's beneficial to
introduce utility classes which can help making your graph construction code
more readable and less error prone.

MediaPipe offers `PassThroughCalculator` calculator, which is simply passing
through its inputs:

```
input_stream: "float_value"
input_stream: "int_value"
input_stream: "bool_value"

output_stream: "passed_float_value"
output_stream: "passed_int_value"
output_stream: "passed_bool_value"

node {
  calculator: "PassThroughCalculator"
  input_stream: "float_value"
  input_stream: "int_value"
  input_stream: "bool_value"
  # The order must be the same as for inputs (or you can use explicit indexes)
  output_stream: "passed_float_value"
  output_stream: "passed_int_value"
  output_stream: "passed_bool_value"
}
```

Let's see the straightforward C++ construction code to create the above graph:

```c++
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Graph inputs.
  Stream<float> float_value = graph.In(0).SetName("float_value").Cast<float>();
  Stream<int> int_value = graph.In(1).SetName("int_value").Cast<int>();
  Stream<bool> bool_value = graph.In(2).SetName("bool_value").Cast<bool>();

  auto& pass_node = graph.AddNode("PassThroughCalculator");
  float_value.ConnectTo(pass_node.In("")[0]);
  int_value.ConnectTo(pass_node.In("")[1]);
  bool_value.ConnectTo(pass_node.In("")[2]);
  Stream<float> passed_float_value = pass_node.Out("")[0].Cast<float>();
  Stream<int> passed_int_value = pass_node.Out("")[1].Cast<int>();
  Stream<bool> passed_bool_value = pass_node.Out("")[2].Cast<bool>();

  // Graph outputs.
  passed_float_value.SetName("passed_float_value").ConnectTo(graph.Out(0));
  passed_int_value.SetName("passed_int_value").ConnectTo(graph.Out(1));
  passed_bool_value.SetName("passed_bool_value").ConnectTo(graph.Out(2));

  // Get `CalculatorGraphConfig` to pass it into `CalculatorGraph`
  return graph.GetConfig();
}
```

While `pbtxt` representation maybe error prone (when we have many inputs to pass
through), C++ code looks even worse: repeated empty tags and `Cast` calls. Let's
see how we can do better by introducing a `PassThroughNodeBuilder`:

```c++
class PassThroughNodeBuilder {
 public:
  explicit PassThroughNodeBuilder(Graph& graph)
      : node_(graph.AddNode("PassThroughCalculator")) {}

  template <typename T>
  Stream<T> PassThrough(Stream<T> stream) {
    stream.ConnectTo(node_.In(index_));
    return node_.Out(index_++).Cast<T>();
  }

 private:
  int index_ = 0;
  GenericNode& node_;
};
```

And now graph construction code can look like:

```c++
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Graph inputs.
  Stream<float> float_value = graph.In(0).SetName("float_value").Cast<float>();
  Stream<int> int_value = graph.In(1).SetName("int_value").Cast<int>();
  Stream<bool> bool_value = graph.In(2).SetName("bool_value").Cast<bool>();

  PassThroughNodeBuilder pass_node_builder(graph);
  Stream<float> passed_float_value = pass_node_builder.PassThrough(float_value);
  Stream<int> passed_int_value = pass_node_builder.PassThrough(int_value);
  Stream<bool> passed_bool_value = pass_node_builder.PassThrough(bool_value);

  // Graph outputs.
  passed_float_value.SetName("passed_float_value").ConnectTo(graph.Out(0));
  passed_int_value.SetName("passed_int_value").ConnectTo(graph.Out(1));
  passed_bool_value.SetName("passed_bool_value").ConnectTo(graph.Out(2));

  // Get `CalculatorGraphConfig` to pass it into `CalculatorGraph`
  return graph.GetConfig();
}
```

Now you can't have incorrect order or index in your pass through construction
code and save some typing by guessing the type for `Cast` from the `PassThrough`
input.

Tip: the same as for the `RunInference` function, extracting
`PassThroughNodeBuilder` and similar utility classes into dedicated modules
enables reuse in graph construction code and helps to automatically pull in the
corresponding calculator dependencies.

## Dos and Don'ts

### Define graph inputs at the very beginning if possible

```c++ {.bad}
Stream<D> RunSomething(Stream<A> a, Stream<B> b, Graph& graph) {
  Stream<C> c = graph.In(2).SetName("c").Cast<C>();  // Bad.
  // ...
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  Stream<A> a = graph.In(0).SetName("a").Cast<A>();
  // 10/100/N lines of code.
  Stream<B> b = graph.In(1).SetName("b").Cast<B>()  // Bad.
  Stream<D> d = RunSomething(a, b, graph);
  // ...

  return graph.GetConfig();
}

```

In the above code:

*   It can be hard to guess how many inputs you have in the graph.
*   Can be error prone overall and hard to maintain in future (e.g. is it a
    correct index? name? what if some inputs are removed or made optional?
    etc.).
*   `RunSomething` reuse is limited because other graphs may have different
    inputs

Instead, define your graph inputs at the very beginning of your graph builder:

```c++ {.good}
Stream<D> RunSomething(Stream<A> a, Stream<B> b, Stream<C> c, Graph& graph) {
  // ...
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).SetName("a").Cast<A>();
  Stream<B> b = graph.In(1).SetName("b").Cast<B>();
  Stream<C> c = graph.In(2).SetName("c").Cast<C>();

  // 10/100/N lines of code.
  Stream<D> d = RunSomething(a, b, c, graph);
  // ...

  return graph.GetConfig();
}
```

Use `std::optional` if you have an input stream or side packet that is not
always defined and put it at the very beginning:

```c++ {.good}
std::optional<Stream<A>> a;
if (needs_a) {
  a = graph.In(0).SetName(a).Cast<A>();
}
```

Note: of course, there can be exceptions - for example, there can be a use case
where calling `RunSomething1(..., graph)`, ..., `RunSomethingN(..., graph)` is
**intended to add new inputs**, so afterwards you can iterate over them and feed
only added inputs into the graph. However, in any case, try to make it easy for
readers to find out what graph inputs it has or may have.

### Define graph outputs at the very end

```c++ {.bad}
void RunSomething(Stream<Input> input, Graph& graph) {
  // ...
  node.Out("OUTPUT_F")
      .SetName("output_f").ConnectTo(graph.Out(2));  // Bad.
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // 10/100/N lines of code.
  node.Out("OUTPUT_D")
      .SetName("output_d").ConnectTo(graph.Out(0));  // Bad.
  // 10/100/N lines of code.
  node.Out("OUTPUT_E")
      .SetName("output_e").ConnectTo(graph.Out(1));  // Bad.
  // 10/100/N lines of code.
  RunSomething(input, graph);
  // ...

  return graph.GetConfig();
}
```

In the above code:

*   It can be hard to guess how many outputs you have in the graph.
*   Can be error prone overall and hard to maintain in future (e.g. is it a
    correct index? name? what if some outpus are removed or made optional?
    etc.).
*   `RunSomething` reuse is limited as other graphs may have different outputs

Instead, define your graph outputs at the very end of your graph builder:

```c++ {.good}
Stream<F> RunSomething(Stream<Input> input, Graph& graph) {
  // ...
  return node.Out("OUTPUT_F").Cast<F>();
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // 10/100/N lines of code.
  Stream<D> d = node.Out("OUTPUT_D").Cast<D>();
  // 10/100/N lines of code.
  Stream<E> e = node.Out("OUTPUT_E").Cast<E>();
  // 10/100/N lines of code.
  Stream<F> f = RunSomething(input, graph);
  // ...

  // Outputs.
  d.SetName("output_d").ConnectTo(graph.Out(0));
  e.SetName("output_e").ConnectTo(graph.Out(1));
  f.SetName("output_f").ConnectTo(graph.Out(2));

  return graph.GetConfig();
}
```

### Keep nodes decoupled from each other

In MediaPipe, packet streams and side packets are as meaningful as processing
nodes. And any node input requirements and output products are expressed clearly
and independently in terms of the streams and side packets it consumes and
produces.

```c++ {.bad}
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();

  auto& node1 = graph.AddNode("Calculator1");
  a.ConnectTo(node1.In("INPUT"));

  auto& node2 = graph.AddNode("Calculator2");
  node1.Out("OUTPUT").ConnectTo(node2.In("INPUT"));  // Bad.

  auto& node3 = graph.AddNode("Calculator3");
  node1.Out("OUTPUT").ConnectTo(node3.In("INPUT_B"));  // Bad.
  node2.Out("OUTPUT").ConnectTo(node3.In("INPUT_C"));  // Bad.

  auto& node4 = graph.AddNode("Calculator4");
  node1.Out("OUTPUT").ConnectTo(node4.In("INPUT_B"));  // Bad.
  node2.Out("OUTPUT").ConnectTo(node4.In("INPUT_C"));  // Bad.
  node3.Out("OUTPUT").ConnectTo(node4.In("INPUT_D"));  // Bad.

  // Outputs.
  node1.Out("OUTPUT").SetName("b").ConnectTo(graph.Out(0));  // Bad.
  node2.Out("OUTPUT").SetName("c").ConnectTo(graph.Out(1));  // Bad.
  node3.Out("OUTPUT").SetName("d").ConnectTo(graph.Out(2));  // Bad.
  node4.Out("OUTPUT").SetName("e").ConnectTo(graph.Out(3));  // Bad.

  return graph.GetConfig();
}
```

In the above code:

*   Nodes are coupled to each other, e.g. `node4` knows where its inputs are
    coming from (`node1`, `node2`, `node3`) and it complicates refactoring,
    maintenance and code reuse
    *   Such usage pattern is a downgrade from proto representation, where nodes
        are decoupled by default.
*   `node#.Out("OUTPUT")` calls are duplicated and readability suffers as you
    could use cleaner names instead and also provide an actual type.

So, to fix the above issues you can write the following graph construction code:

```c++ {.good}
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();

  // `node1` usage is limited to 3 lines below.
  auto& node1 = graph.AddNode("Calculator1");
  a.ConnectTo(node1.In("INPUT"));
  Stream<B> b = node1.Out("OUTPUT").Cast<B>();

  // `node2` usage is limited to 3 lines below.
  auto& node2 = graph.AddNode("Calculator2");
  b.ConnectTo(node2.In("INPUT"));
  Stream<C> c = node2.Out("OUTPUT").Cast<C>();

  // `node3` usage is limited to 4 lines below.
  auto& node3 = graph.AddNode("Calculator3");
  b.ConnectTo(node3.In("INPUT_B"));
  c.ConnectTo(node3.In("INPUT_C"));
  Stream<D> d = node3.Out("OUTPUT").Cast<D>();

  // `node4` usage is limited to 5 lines below.
  auto& node4 = graph.AddNode("Calculator4");
  b.ConnectTo(node4.In("INPUT_B"));
  c.ConnectTo(node4.In("INPUT_C"));
  d.ConnectTo(node4.In("INPUT_D"));
  Stream<E> e = node4.Out("OUTPUT").Cast<E>();

  // Outputs.
  b.SetName("b").ConnectTo(graph.Out(0));
  c.SetName("c").ConnectTo(graph.Out(1));
  d.SetName("d").ConnectTo(graph.Out(2));
  e.SetName("e").ConnectTo(graph.Out(3));

  return graph.GetConfig();
}
```

Now, if needed, you can easily remove `node1` and make `b` a graph input and no
updates are needed to `node2`, `node3`, `node4` (same as in proto representation
by the way), because they are decoupled from each other.

Overall, the above code replicates the proto graph more closely:

```proto
input_stream: "a"

node {
  calculator: "Calculator1"
  input_stream: "INPUT:a"
  output_stream: "OUTPUT:b"
}

node {
  calculator: "Calculator2"
  input_stream: "INPUT:b"
  output_stream: "OUTPUT:C"
}

node {
  calculator: "Calculator3"
  input_stream: "INPUT_B:b"
  input_stream: "INPUT_C:c"
  output_stream: "OUTPUT:d"
}

node {
  calculator: "Calculator4"
  input_stream: "INPUT_B:b"
  input_stream: "INPUT_C:c"
  input_stream: "INPUT_D:d"
  output_stream: "OUTPUT:e"
}

output_stream: "b"
output_stream: "c"
output_stream: "d"
output_stream: "e"
```

On top of that, now you can extract utility functions for further reuse in other graphs:

```c++ {.good}
Stream<B> RunCalculator1(Stream<A> a, Graph& graph) {
  auto& node = graph.AddNode("Calculator1");
  a.ConnectTo(node.In("INPUT"));
  return node.Out("OUTPUT").Cast<B>();
}

Stream<C> RunCalculator2(Stream<B> b, Graph& graph) {
  auto& node = graph.AddNode("Calculator2");
  b.ConnectTo(node.In("INPUT"));
  return node.Out("OUTPUT").Cast<C>();
}

Stream<D> RunCalculator3(Stream<B> b, Stream<C> c, Graph& graph) {
  auto& node = graph.AddNode("Calculator3");
  b.ConnectTo(node.In("INPUT_B"));
  c.ConnectTo(node.In("INPUT_C"));
  return node.Out("OUTPUT").Cast<D>();
}

Stream<E> RunCalculator4(Stream<B> b, Stream<C> c, Stream<D> d, Graph& graph) {
  auto& node = graph.AddNode("Calculator4");
  b.ConnectTo(node.In("INPUT_B"));
  c.ConnectTo(node.In("INPUT_C"));
  d.ConnectTo(node.In("INPUT_D"));
  return node.Out("OUTPUT").Cast<E>();
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();

  Stream<B> b = RunCalculator1(a, graph);
  Stream<C> c = RunCalculator2(b, graph);
  Stream<D> d = RunCalculator3(b, c, graph);
  Stream<E> e = RunCalculator4(b, c, d, graph);

  // Outputs.
  b.SetName("b").ConnectTo(graph.Out(0));
  c.SetName("c").ConnectTo(graph.Out(1));
  d.SetName("d").ConnectTo(graph.Out(2));
  e.SetName("e").ConnectTo(graph.Out(3));

  return graph.GetConfig();
}
```

### Separate nodes for better readability

```c++ {.bad}
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();
  auto& node1 = graph.AddNode("Calculator1");
  a.ConnectTo(node1.In("INPUT"));
  Stream<B> b = node1.Out("OUTPUT").Cast<B>();
  auto& node2 = graph.AddNode("Calculator2");
  b.ConnectTo(node2.In("INPUT"));
  Stream<C> c = node2.Out("OUTPUT").Cast<C>();
  auto& node3 = graph.AddNode("Calculator3");
  b.ConnectTo(node3.In("INPUT_B"));
  c.ConnectTo(node3.In("INPUT_C"));
  Stream<D> d = node3.Out("OUTPUT").Cast<D>();
  auto& node4 = graph.AddNode("Calculator4");
  b.ConnectTo(node4.In("INPUT_B"));
  c.ConnectTo(node4.In("INPUT_C"));
  d.ConnectTo(node4.In("INPUT_D"));
  Stream<E> e = node4.Out("OUTPUT").Cast<E>();
  // Outputs.
  b.SetName("b").ConnectTo(graph.Out(0));
  c.SetName("c").ConnectTo(graph.Out(1));
  d.SetName("d").ConnectTo(graph.Out(2));
  e.SetName("e").ConnectTo(graph.Out(3));

  return graph.GetConfig();
}
```

In the above code, it can be hard to grasp the idea where each node begins and
ends. To improve this and help your code readers, you can simply have blank
lines before and after each node:

```c++ {.good}
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();

  auto& node1 = graph.AddNode("Calculator1");
  a.ConnectTo(node1.In("INPUT"));
  Stream<B> b = node1.Out("OUTPUT").Cast<B>();

  auto& node2 = graph.AddNode("Calculator2");
  b.ConnectTo(node2.In("INPUT"));
  Stream<C> c = node2.Out("OUTPUT").Cast<C>();

  auto& node3 = graph.AddNode("Calculator3");
  b.ConnectTo(node3.In("INPUT_B"));
  c.ConnectTo(node3.In("INPUT_C"));
  Stream<D> d = node3.Out("OUTPUT").Cast<D>();

  auto& node4 = graph.AddNode("Calculator4");
  b.ConnectTo(node4.In("INPUT_B"));
  c.ConnectTo(node4.In("INPUT_C"));
  d.ConnectTo(node4.In("INPUT_D"));
  Stream<E> e = node4.Out("OUTPUT").Cast<E>();

  // Outputs.
  b.SetName("b").ConnectTo(graph.Out(0));
  c.SetName("c").ConnectTo(graph.Out(1));
  d.SetName("d").ConnectTo(graph.Out(2));
  e.SetName("e").ConnectTo(graph.Out(3));

  return graph.GetConfig();
}
```

Also, the above representation matches `CalculatorGraphConfig` proto
representation better.

If you extract nodes into utility functions, they are scoped within functions
already and it's clear where they begin and end, so it's completely fine to
have:

```c++ {.good}
CalculatorGraphConfig BuildGraph() {
  Graph graph;

  // Inputs.
  Stream<A> a = graph.In(0).Cast<A>();

  Stream<B> b = RunCalculator1(a, graph);
  Stream<C> c = RunCalculator2(b, graph);
  Stream<D> d = RunCalculator3(b, c, graph);
  Stream<E> e = RunCalculator4(b, c, d, graph);

  // Outputs.
  b.SetName("b").ConnectTo(graph.Out(0));
  c.SetName("c").ConnectTo(graph.Out(1));
  d.SetName("d").ConnectTo(graph.Out(2));
  e.SetName("e").ConnectTo(graph.Out(3));

  return graph.GetConfig();
}
```
