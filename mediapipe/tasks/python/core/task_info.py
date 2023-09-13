# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MediaPipe Tasks' task info data class."""

import dataclasses

from typing import Any, List

from mediapipe.calculators.core import flow_limiter_calculator_pb2
from mediapipe.framework import calculator_options_pb2
from mediapipe.framework import calculator_pb2
from mediapipe.tasks.python.core.optional_dependencies import doc_controls


@doc_controls.do_not_generate_docs
@dataclasses.dataclass
class TaskInfo:
  """Specifications of a MediaPipe task graph.

  Attributes:
    task_graph: The MediaPipe task graph name.
    input_streams: The list of graph input stream info strings in the form
      TAG:name.
    output_streams: The list of graph output stream info strings in the form
      TAG:name.
    task_options: The task-specific options object that can be converted to a
      protobuf object.
  """
  task_graph: str
  input_streams: List[str]
  output_streams: List[str]
  task_options: Any

  def generate_graph_config(
      self,
      enable_flow_limiting: bool = False
  ) -> calculator_pb2.CalculatorGraphConfig:
    """Generates a MediaPipe Task CalculatorGraphConfig proto from TaskInfo.

    Args:
      enable_flow_limiting: Whether to add a flow limiter calculator into the
        graph config to lower the overall graph latency for live streaming use
        case.

    Raises:
      ValueError: Any required data fields (namely, `task_graph`,
        `task_options`, `input_streams`, and  `output_streams`) is not
        specified or `task_options` is not able to be converted to a protobuf
        object.

    Returns:
      A CalculatorGraphConfig proto of the task graph.
    """

    def strip_tag_index(tag_index_name):
      return tag_index_name.split(':')[-1]

    def add_stream_name_prefix(tag_index_name):
      splitted = tag_index_name.split(':')
      splitted[-1] = 'throttled_' + splitted[-1]
      return ':'.join(splitted)

    if not self.task_graph or not self.task_options:
      raise ValueError('Please provide both `task_graph` and `task_options`.')
    if not self.input_streams or not self.output_streams:
      raise ValueError(
          'Both `input_streams` and `output_streams` must be non-empty.')
    if not hasattr(self.task_options, 'to_pb2'):
      raise ValueError(
          '`task_options` doesn`t provide `to_pb2()` method to convert itself to be a protobuf object.'
      )
    task_subgraph_options = calculator_options_pb2.CalculatorOptions()
    task_options_proto = self.task_options.to_pb2()
    task_subgraph_options.Extensions[task_options_proto.ext].CopyFrom(
        task_options_proto)
    if not enable_flow_limiting:
      return calculator_pb2.CalculatorGraphConfig(
          node=[
              calculator_pb2.CalculatorGraphConfig.Node(
                  calculator=self.task_graph,
                  input_stream=self.input_streams,
                  output_stream=self.output_streams,
                  options=task_subgraph_options)
          ],
          input_stream=self.input_streams,
          output_stream=self.output_streams)
    # When a FlowLimiterCalculator is inserted to lower the overall graph
    # latency, the task doesn't guarantee that each input must have the
    # corresponding output.
    task_subgraph_inputs = [
        add_stream_name_prefix(stream) for stream in self.input_streams
    ]
    finished_stream = 'FINISHED:' + strip_tag_index(self.output_streams[0])
    flow_limiter_options = calculator_options_pb2.CalculatorOptions()
    flow_limiter_options.Extensions[
        flow_limiter_calculator_pb2.FlowLimiterCalculatorOptions.ext].CopyFrom(
            flow_limiter_calculator_pb2.FlowLimiterCalculatorOptions(
                max_in_flight=1, max_in_queue=1))
    flow_limiter = calculator_pb2.CalculatorGraphConfig.Node(
        calculator='FlowLimiterCalculator',
        input_stream_info=[
            calculator_pb2.InputStreamInfo(
                tag_index='FINISHED', back_edge=True)
        ],
        input_stream=[strip_tag_index(stream) for stream in self.input_streams]
        + [finished_stream],
        output_stream=[
            strip_tag_index(stream) for stream in task_subgraph_inputs
        ],
        options=flow_limiter_options)
    config = calculator_pb2.CalculatorGraphConfig(
        node=[
            calculator_pb2.CalculatorGraphConfig.Node(
                calculator=self.task_graph,
                input_stream=task_subgraph_inputs,
                output_stream=self.output_streams,
                options=task_subgraph_options), flow_limiter
        ],
        input_stream=self.input_streams,
        output_stream=self.output_streams)
    return config
