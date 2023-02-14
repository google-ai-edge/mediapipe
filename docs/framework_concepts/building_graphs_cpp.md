---
layout: default
title: Building Graphs in C++
parent: Graphs
nav_order: 1
---

# Building Graphs in C++
{: .no_toc }

1. TOC
{:toc}
---

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
// Graph inputs.
input_stream: "input_tensors"
input_side_packet: "model"

// Graph outputs.
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
  // The order must be the same as for inputs (or you can use explicit indexes)
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
}

```

In the above code:

*   It can be hard to guess how many inputs you have in the graph.
*   Can be error prone overall and hard to maintain in future (e.g. is it a
    correct index? name? what if some inputs are removed or made optional?
    etc.).

Instead, simply define your graph inputs at the very beginning of your graph
builder:

```c++ {.good}
Stream<int> RunSomething(Stream<A> a, Stream<B> b, Stream<C> c, Graph& graph) {
  // ...
}

CalculatorGraphConfig BuildGraph() {
  Graph graph;

  Stream<A> a = graph.In(0).SetName("a").Cast<A>();
  Stream<B> b = graph.In(1).SetName("b").Cast<B>();
  Stream<C> c = graph.In(2).SetName("c").Cast<C>();

  // 10/100/N lines of code.
  Stream<D> d = RunSomething(a, b, c, graph);
  // ...
}
```

And if you have an input stream or side packet that is not always defined -
simply use `std::optional` and put it at the very beginning as well:

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
