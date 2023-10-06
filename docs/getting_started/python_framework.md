---
layout: forward
target: https://developers.google.com/mediapipe/framework/getting_started/python_framework
parent: MediaPipe in Python
grand_parent: Getting Started
nav_order: 1
---

# MediaPipe Python Framework
{: .no_toc }

1. TOC
{:toc}
---
**Attention:** *Thanks for your interest in MediaPipe! We have moved to
[https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)
as the primary developer documentation site for MediaPipe as of April 3, 2023.*

----

The MediaPipe Python framework grants direct access to the core components of
the MediaPipe C++ framework such as Timestamp, Packet, and CalculatorGraph,
whereas the
[ready-to-use Python solutions](./python.md#ready-to-use-python-solutions) hide
the technical details of the framework and simply return the readable model
inference results back to the callers.

MediaPipe framework sits on top of
[the pybind11 library](https://pybind11.readthedocs.io/en/stable/index.html).
The C++ core framework is exposed in Python via a C++/Python language binding.
The content below assumes that the reader already has a basic understanding of
the MediaPipe C++ framework. Otherwise, you can find useful information in
[Framework Concepts](../framework_concepts/framework_concepts.md).

### Packet

The packet is the basic data flow unit in MediaPipe. A packet consists of a
numeric timestamp and a shared pointer to an immutable payload. In Python, a
MediaPipe packet can be created by calling one of the packet creator methods in
the
[`mp.packet_creator`](https://github.com/google/mediapipe/tree/master/mediapipe/python/pybind/packet_creator.cc)
module. Correspondingly, the packet payload can be retrieved by using one of the
packet getter methods in the
[`mp.packet_getter`](https://github.com/google/mediapipe/tree/master/mediapipe/python/pybind/packet_getter.cc)
module. Note that the packet payload becomes **immutable** after packet
creation. Thus, the modification of the retrieved packet content doesn't affect
the actual payload in the packet. MediaPipe framework Python API supports the
most commonly used data types of MediaPipe (e.g., ImageFrame, Matrix, Protocol
Buffers, and the primitive data types) in the core binding. The comprehensive
table below shows the type mappings between the Python and the C++ data type
along with the packet creator and the content getter method for each data type
supported by the MediaPipe Python framework API.

Python Data Type                     | C++ Data Type                 | Packet Creator                                                                                                                                               | Content Getter
------------------------------------ | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------
bool                                 | bool                          | create_bool(True)                                                                                                                                            | get_bool(packet)
int or np.intc                       | int_t                         | create_int(1)                                                                                                                                                | get_int(packet)
int or np.int8                       | int8_t                        | create_int8(2**7-1)                                                                                                                                          | get_int(packet)
int or np.int16                      | int16_t                       | create_int16(2**15-1)                                                                                                                                        | get_int(packet)
int or np.int32                      | int32_t                       | create_int32(2**31-1)                                                                                                                                        | get_int(packet)
int or np.int64                      | int64_t                       | create_int64(2**63-1)                                                                                                                                        | get_int(packet)
int or np.uint8                      | uint8_t                       | create_uint8(2**8-1)                                                                                                                                         | get_uint(packet)
int or np.uint16                     | uint16_t                      | create_uint16(2**16-1)                                                                                                                                       | get_uint(packet)
int or np.uint32                     | uint32_t                      | create_uint32(2**32-1)                                                                                                                                       | get_uint(packet)
int or np.uint64                     | uint64_t                      | create_uint64(2**64-1)                                                                                                                                       | get_uint(packet)
float or np.float32                  | float                         | create_float(1.1)                                                                                                                                            | get_float(packet)
float or np.double                   | double                        | create_double(1.1)                                                                                                                                           | get_float(packet)
str (UTF-8)                          | std::string                   | create_string('abc')                                                                                                                                         | get_str(packet)
bytes                                | std::string                   | create_string(b'\xd0\xd0\xd0')                                                                                                                               | get_bytes(packet)
mp.Packet                            | mp::Packet                    | create_packet(p)                                                                                                                                             | get_packet(packet)
List\[bool\]                         | std::vector\<bool\>           | create_bool_vector(\[True, False\])                                                                                                                          | get_bool_list(packet)
List\[int\] or List\[np.intc\]       | int\[\]                       | create_int_array(\[1, 2, 3\])                                                                                                                                | get_int_list(packet, size=10)
List\[int\] or List\[np.intc\]       | std::vector\<int\>            | create_int_vector(\[1, 2, 3\])                                                                                                                               | get_int_list(packet)
List\[float\] or List\[np.float\]    | float\[\]                     | create_float_arrary(\[0.1, 0.2\])                                                                                                                            | get_float_list(packet, size=10)
List\[float\] or List\[np.float\]    | std::vector\<float\>          | create_float_vector(\[0.1, 0.2\])                                                                                                                            | get_float_list(packet, size=10)
List\[str\]                          | std::vector\<std::string\>    | create_string_vector(\['a'\])                                                                                                                                | get_str_list(packet)
List\[mp.Packet\]                    | std::vector\<mp::Packet\>     | create_packet_vector(<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\[packet1, packet2\])                                                               | get_packet_list(p)
Mapping\[str, Packet\]               | std::map<std::string, Packet> | create_string_to_packet_map(<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{'a': packet1, 'b': packet2})                                                | get_str_to_packet_dict(packet)
np.ndarray<br>(cv.mat and PIL.Image) | mp::ImageFrame                | create_image_frame(<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;format=ImageFormat.SRGB,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;data=mat) | get_image_frame(packet)
np.ndarray                           | mp::Matrix                    | create_matrix(data)                                                                                                                                          | get_matrix(packet)
Google Proto Message                 | Google Proto Message          | create_proto(proto)                                                                                                                                          | get_proto(packet)
List\[Proto\]                        | std::vector\<Proto\>          | n/a                                                                                                                                                          | get_proto_list(packet)

It's not uncommon that users create custom C++ classes and send those into
the graphs and calculators. To allow the custom classes to be used in Python
with MediaPipe, you may extend the Packet API for a new data type in the
following steps:

1.  Write the pybind11
    [class binding code](https://pybind11.readthedocs.io/en/stable/advanced/classes.html)
    or
    [a custom type caster](https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html?highlight=custom%20type%20caster)
    for the custom type in a cc file.

    ```c++
    #include "path/to/my_type/header/file.h"
    #include "pybind11/pybind11.h"

    namespace py = pybind11;

    PYBIND11_MODULE(my_type_binding, m) {
      // Write binding code or a custom type caster for MyType.
      py::class_<MyType>(m, "MyType")
          .def(py::init<>())
          .def(...);
    }
    ```

2.  Create a new packet creator and getter method of the custom type in a
    separate cc file.

    ```c++
    #include "path/to/my_type/header/file.h"
    #include "mediapipe/framework/packet.h"
    #include "pybind11/pybind11.h"

    namespace mediapipe {
    namespace py = pybind11;

    PYBIND11_MODULE(my_packet_methods, m) {
      m.def(
          "create_my_type",
          [](const MyType& my_type) { return MakePacket<MyType>(my_type); });

      m.def(
          "get_my_type",
          [](const Packet& packet) {
            if(!packet.ValidateAsType<MyType>().ok()) {
              PyErr_SetString(PyExc_ValueError, "Packet data type mismatch.");
              return py::error_already_set();
            }
            return packet.Get<MyType>();
          });
    }
    }  // namespace mediapipe
    ```

3.  Add two bazel build rules for the custom type binding and the new packet
    methods in the BUILD file.

    ```
    load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

    pybind_extension(
        name = "my_type_binding",
        srcs = ["my_type_binding.cc"],
        deps = [":my_type"],
    )

    pybind_extension(
        name = "my_packet_methods",
        srcs = ["my_packet_methods.cc"],
        deps = [
            ":my_type",
            "//mediapipe/framework:packet"
        ],
    )
    ```

4.  Build the pybind extension targets (with the suffix .so) by Bazel and move the generated dynamic libraries into one of the $LD_LIBRARY_PATH dirs.

5.  Use the binding modules in Python.

    ```python
    import my_type_binding
    import my_packet_methods

    packet = my_packet_methods.create_my_type(my_type_binding.MyType())
    my_type = my_packet_methods.get_my_type(packet)
    ```

### Timestamp

Each packet contains a timestamp that is in units of microseconds. In Python,
the Packet API provides a convenience method `packet.at()` to define the numeric
timestamp of a packet. More generally, `packet.timestamp` is the packet class
property for accessing the underlying timestamp. To convert an Unix epoch to a
MediaPipe timestamp,
[the Timestamp API](https://github.com/google/mediapipe/tree/master/mediapipe/python/pybind/timestamp.cc)
offers a method `mp.Timestamp.from_seconds()` for this purpose.

### ImageFrame

ImageFrame is the container for storing an image or a video frame. Formats
supported by ImageFrame are listed in
[the ImageFormat enum](https://github.com/google/mediapipe/tree/master/mediapipe/python/pybind/image_frame.cc#l=170).
Pixels are encoded row-major with interleaved color components, and ImageFrame
supports uint8, uint16, and float as its data types. MediaPipe provides
[an ImageFrame Python API](https://github.com/google/mediapipe/tree/master/mediapipe/python/pybind/image_frame.cc)
to access the ImageFrame C++ class. In Python, the easiest way to retrieve the
pixel data is to call `image_frame.numpy_view()` to get a numpy ndarray. Note
that the returned numpy ndarray, a reference to the internal pixel data, is
unwritable. If the callers need to modify the numpy ndarray, it's required to
explicitly call a copy operation to obtain a copy. When MediaPipe takes a numpy
ndarray to make an ImageFrame, it assumes that the data is stored contiguously.
Correspondingly, the pixel data of an ImageFrame will be realigned to be
contiguous when it's returned to the Python side.

### Graph

In MediaPipe, all processing takes places within the context of a
CalculatorGraph.
[The CalculatorGraph Python API](https://github.com/google/mediapipe/tree/master/mediapipe/python/pybind/calculator_graph.cc)
is a direct binding to the C++ CalculatorGraph class. The major difference is
the CalculatorGraph Python API raises a Python error instead of returning a
non-OK Status when an error occurs. Therefore, as a Python user, you can handle
the exceptions as you normally do. The life cycle of a CalculatorGraph contains
three stages: initialization and setup, graph run, and graph shutdown.

1.  Initialize a CalculatorGraph with a CalculatorGraphConfig protobuf or binary
    protobuf file, and provide callback method(s) to observe the output
    stream(s).

    Option 1. Initialize a CalculatorGraph with a CalculatorGraphConfig protobuf
    or its text representation, and observe the output stream(s):

    ```python
    import mediapipe as mp

    config_text = """
      input_stream: 'in_stream'
      output_stream: 'out_stream'
      node {
        calculator: 'PassThroughCalculator'
        input_stream: 'in_stream'
        output_stream: 'out_stream'
      }
    """
    graph = mp.CalculatorGraph(graph_config=config_text)
    output_packets = []
    graph.observe_output_stream(
        'out_stream',
        lambda stream_name, packet:
            output_packets.append(mp.packet_getter.get_str(packet)))
    ```

    Option 2. Initialize a CalculatorGraph with a binary protobuf file, and
    observe the output stream(s).

    ```python
    import mediapipe as mp
    # resources dependency

    graph = mp.CalculatorGraph(
        binary_graph=os.path.join(
            resources.GetRunfilesDir(), 'path/to/your/graph.binarypb'))
    graph.observe_output_stream(
        'out_stream',
        lambda stream_name, packet: print(f'Get {packet} from {stream_name}'))
    ```

2.  Start the graph run and feed packets into the graph.

    ```python
    graph.start_run()

    graph.add_packet_to_input_stream(
        'in_stream', mp.packet_creator.create_string('abc').at(0))

    rgb_img = cv2.cvtColor(cv2.imread('/path/to/your/image.png'), cv2.COLOR_BGR2RGB)
    graph.add_packet_to_input_stream(
        'in_stream',
        mp.packet_creator.create_image_frame(image_format=mp.ImageFormat.SRGB,
                                             data=rgb_img).at(1))
    ```

3.  Close the graph after finish. You may restart the graph for another graph
    run after the call to `close()`.

    ```python
    graph.close()
    ```

The Python script can be run by your local Python runtime.
