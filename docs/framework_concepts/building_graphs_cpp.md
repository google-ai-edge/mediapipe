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

## C++ Graph Builder

C++ graph builder is a powerful tool for:

*   Building complex graphs
*   Parametrizing graphs (e.g. setting a delegate on
    `InferenceCalculator`, enabling/disabling parts of the graph)
*   Deduplicating graphs (e.g. instead of CPU and GPU dedicated graphs in pbtxt
    you can have a single code that constructs required graphs, sharing as much
    as possible)
*   Supporting optional graph inputs/outputs
*   Customizing graphs per platform

### Basic Usage

Let's see how C++ graph builder can be used for a simple graph:

```proto
// Graph inputs.
input_stream: "input_tensors"
input_side_packet: "model"

// Graph outputs.
output_stream: "output_tensors"

// Nodes.
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

  // Nodes.
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
*   Use `Stream/SidePacket::ConnectTo` to connect streams and side packets to node
    inputs (`Node::In/SideIn`) and graph outputs (`Graph::Out/SideOut`)
    *   There's a "shortcut" operator `>>` that you can use instead of
        `ConnectTo` function (E.g. `x >> node.In("IN")`).
*   `Stream/SidePacket::Cast` is used to cast stream or side packet of `AnyType` (E.g. `Stream<AnyType> in = graph.In(0);`) to a particular type
    *   Using actual types instead of `AnyType` sets you on a better path for unleashing graph
        builder capabilities and improving your graphs readability.


