# Copyright 2024 The MediaPipe Authors.
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

"""Utility library that helps create the converter instances."""
from mediapipe.tasks.python.genai.converter import converter_base
from mediapipe.tasks.python.genai.converter import pytorch_converter
from mediapipe.tasks.python.genai.converter import safetensors_converter
from mediapipe.tasks.python.genai.converter import weight_bins_writer


def create_ckpt_loader(
    ckpt_format: str, *args, **kwargs
) -> converter_base.CkptLoaderBase:
  """Creates the checkpoint loader.

  Args:
    ckpt_format: A string that indicates which input checkpoint format is.
    *args: Additional arguments to be passed into the loader.
    **kwargs: Additional arguments to be passed into the loader.

  Returns:
    A created CkptLoader instance.
  """
  del args
  if ckpt_format == "pytorch":
    return pytorch_converter.PytorchCkptLoader(
        ckpt_path=kwargs["ckpt_path"],
        is_symmetric=kwargs["is_symmetric"],
        attention_quant_bits=kwargs["attention_quant_bits"],
        feedforward_quant_bits=kwargs["feedforward_quant_bits"],
        embedding_quant_bits=kwargs["embedding_quant_bits"],
        special_model=kwargs["special_model"],
        backend=kwargs["backend"],
    )
  elif ckpt_format == "safetensors":
    return safetensors_converter.SafetensorsCkptLoader(
        ckpt_path=kwargs["ckpt_path"],
        is_symmetric=kwargs["is_symmetric"],
        attention_quant_bits=kwargs["attention_quant_bits"],
        feedforward_quant_bits=kwargs["feedforward_quant_bits"],
        embedding_quant_bits=kwargs["embedding_quant_bits"],
        special_model=kwargs["special_model"],
        backend=kwargs["backend"],
    )
  else:
    raise ValueError(f"Unknown checkpoint format: {ckpt_format}")


def create_writer(
    writer_type: str, *args, **kwargs
) -> converter_base.ModelWriterBase:
  """Creates the model writer.

  Args:
    writer_type: A string the indicates which model writer to create.
    *args: Additional arguments to be passed into the loader.
    **kwargs: Additional arguments to be passed into the loader.

  Returns:
    A created ModelWriter instance.
  """
  del args
  if writer_type == "weight_bins":
    return weight_bins_writer.WeightBinsWriter(
        output_dir=kwargs["output_dir"], backend=kwargs["backend"]
    )
  else:
    raise ValueError(f"Unknown writer type: {writer_type}")
