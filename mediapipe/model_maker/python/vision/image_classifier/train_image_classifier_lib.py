# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
"""Library to train model."""

import os
import tensorflow as tf

from mediapipe.model_maker.python.core.utils import model_util
from mediapipe.model_maker.python.vision.image_classifier import hyperparameters as hp


def _create_optimizer(init_lr: float, decay_steps: int,
                      warmup_steps: int) -> tf.keras.optimizers.Optimizer:
  """Creates an optimizer with learning rate schedule.

  Uses Keras CosineDecay schedule for the learning rate by default.

  Args:
    init_lr: Initial learning rate.
    decay_steps: Number of steps to decay over.
    warmup_steps: Number of steps to do warmup for.

  Returns:
    A tf.keras.optimizers.Optimizer for model training.
  """
  learning_rate_fn = tf.keras.experimental.CosineDecay(
      initial_learning_rate=init_lr, decay_steps=decay_steps, alpha=0.0)
  if warmup_steps:
    learning_rate_fn = model_util.WarmUp(
        initial_learning_rate=init_lr,
        decay_schedule_fn=learning_rate_fn,
        warmup_steps=warmup_steps)
  optimizer = tf.keras.optimizers.RMSprop(
      learning_rate=learning_rate_fn, rho=0.9, momentum=0.9, epsilon=0.001)

  return optimizer


def train_model(model: tf.keras.Model, hparams: hp.HParams,
                train_ds: tf.data.Dataset,
                validation_ds: tf.data.Dataset) -> tf.keras.callbacks.History:
  """Trains model with the input data and hyperparameters.

  Args:
    model: Input tf.keras.Model.
    hparams: Hyperparameters for training image classifier.
    train_ds: tf.data.Dataset, training data to be fed in tf.keras.Model.fit().
    validation_ds: tf.data.Dataset, validation data to be fed in
      tf.keras.Model.fit().

  Returns:
    The tf.keras.callbacks.History object returned by tf.keras.Model.fit().
  """

  # Learning rate is linear to batch size.
  learning_rate = hparams.learning_rate * hparams.batch_size / 256

  # Get decay steps.
  # NOMUTANTS--(b/256493858):Plan to test it in the unified training library.
  total_training_steps = hparams.steps_per_epoch * hparams.epochs
  default_decay_steps = hparams.decay_samples // hparams.batch_size
  decay_steps = max(total_training_steps, default_decay_steps)

  warmup_steps = hparams.warmup_epochs * hparams.steps_per_epoch
  optimizer = _create_optimizer(
      init_lr=learning_rate, decay_steps=decay_steps, warmup_steps=warmup_steps)

  loss = tf.keras.losses.CategoricalCrossentropy(
      label_smoothing=hparams.label_smoothing)
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  summary_dir = os.path.join(hparams.export_dir, 'summaries')
  summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
  # Save checkpoint every 5 epochs.
  checkpoint_path = os.path.join(hparams.export_dir, 'checkpoint')
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      os.path.join(checkpoint_path, 'model-{epoch:04d}'),
      save_weights_only=True,
      period=5)

  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
  if latest_checkpoint:
    print(f'Resuming from {latest_checkpoint}')
    model.load_weights(latest_checkpoint)

  # Train the model.
  return model.fit(
      x=train_ds,
      epochs=hparams.epochs,
      validation_data=validation_ds,
      callbacks=[summary_callback, checkpoint_callback])
