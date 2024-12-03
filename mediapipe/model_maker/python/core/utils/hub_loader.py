# Copyright 2023 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Handles both V1 and V2 modules."""

import tensorflow_hub as hub


class HubKerasLayerV1V2(hub.KerasLayer):
  """Class to loads TF v1 and TF v2 hub modules that could be fine-tuned.

  Since TF v1 modules couldn't be retrained in hub.KerasLayer. This class
  provides a workaround for retraining the whole tf1 model in tf2. In
  particular, it extract self._func._self_unconditional_checkpoint_dependencies
  into trainable variable in tf1.

  Doesn't update moving-mean/moving-variance for BatchNormalization during
  fine-tuning.
  """

  def _setup_layer(self, trainable=False, **kwargs):
    if self._is_hub_module_v1:
      self._setup_layer_v1(trainable, **kwargs)
    else:
      # call _setup_layer from the base class for v2.
      super(HubKerasLayerV1V2, self)._setup_layer(trainable, **kwargs)

  def _check_trainability(self):
    if self._is_hub_module_v1:
      self._check_trainability_v1()
    else:
      # call _check_trainability from the base class for v2.
      super(HubKerasLayerV1V2, self)._check_trainability()

  def _setup_layer_v1(self, trainable=False, **kwargs):
    """Constructs keras layer with relevant weights and losses."""
    # Initialize an empty layer, then add_weight() etc. as needed.
    super(hub.KerasLayer, self).__init__(trainable=trainable, **kwargs)

    if not self._is_hub_module_v1:
      raise ValueError(
          'Only supports to set up v1 hub module in this function.'
      )

    # v2 trainable_variable:
    if hasattr(self._func, 'trainable_variables'):
      for v in self._func.trainable_variables:
        self._add_existing_weight(v, trainable=True)
      trainable_variables = {id(v) for v in self._func.trainable_variables}
    else:
      trainable_variables = set()

    if not hasattr(self._func, '_self_unconditional_checkpoint_dependencies'):
      raise ValueError(
          "_func doesn't contains attribute "
          '_self_unconditional_checkpoint_dependencies.'
      )
    dependencies = self._func._self_unconditional_checkpoint_dependencies  # pylint: disable=protected-access

    # Adds trainable variables.
    for dep in dependencies:
      if dep.name == 'variables':
        for v in dep.ref:
          if id(v) not in trainable_variables:
            self._add_existing_weight(v, trainable=True)
            trainable_variables.add(id(v))

    # Adds non-trainable variables.
    if hasattr(self._func, 'variables'):
      for v in self._func.variables:
        if id(v) not in trainable_variables:
          self._add_existing_weight(v, trainable=False)

    # Forward the callable's regularization losses (if any).
    if hasattr(self._func, 'regularization_losses'):
      for l in self._func.regularization_losses:
        if not callable(l):
          raise ValueError(
              'hub.KerasLayer(obj) expects obj.regularization_losses to be an '
              'iterable of callables, each returning a scalar loss term.'
          )
        self.add_loss(self._call_loss_if_trainable(l))  # Supports callables.

  def _check_trainability_v1(self):
    """Ignores trainability checks for V1."""
    if self._is_hub_module_v1:
      return  # Nothing to do.
