# Copyright 2026 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Regression tests for path-traversal hardening in weight_bins_writer.

Exercises the real weight_bins_writer._safe_join. Before importing the
module, we register placeholder packages in sys.modules so that
absolute imports inside weight_bins_writer resolve to our stubs
instead of triggering mediapipe.tasks.python.__init__, which eagerly
imports cv2, matplotlib, and jax for unrelated vision utilities.
"""

import importlib.util
import os
import pathlib
import sys
import tempfile
import types
import unittest


def _load_real_weight_bins_writer():
  """Load weight_bins_writer.py directly, shortcutting package __init__."""
  src = (pathlib.Path(__file__).resolve().parent.parent
         / "weight_bins_writer.py")

  # Register placeholder packages for each level of the import path so
  # `from mediapipe.tasks.python.genai.converter import ...` inside
  # weight_bins_writer.py finds an already-initialised package and does
  # not execute the real __init__.py chain.
  pkg_names = [
      "mediapipe",
      "mediapipe.tasks",
      "mediapipe.tasks.python",
      "mediapipe.tasks.python.genai",
      "mediapipe.tasks.python.genai.converter",
  ]
  for name in pkg_names:
    if name not in sys.modules:
      placeholder = types.ModuleType(name)
      placeholder.__path__ = []  # mark as package
      sys.modules[name] = placeholder

  # Minimal base class so `class WeightBinsWriter(converter_base.ModelWriterBase)`
  # resolves at class-definition time.
  class _ModelWriterBaseStub:
    def __init__(self, *args, **kwargs):
      pass

  stubs = {
      "mediapipe.tasks.python.genai.converter.converter_base": {
          "ModelWriterBase": _ModelWriterBaseStub,
      },
      "mediapipe.tasks.python.genai.converter.external_dependencies": {
          "jnp": None,
      },
      "mediapipe.tasks.python.genai.converter.quantization_util": {},
  }
  for stub_name, attrs in stubs.items():
    mod = types.ModuleType(stub_name)
    for k, v in attrs.items():
      setattr(mod, k, v)
    sys.modules[stub_name] = mod

  spec = importlib.util.spec_from_file_location(
      "mediapipe.tasks.python.genai.converter.weight_bins_writer", src)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[
      "mediapipe.tasks.python.genai.converter.weight_bins_writer"] = mod
  spec.loader.exec_module(mod)
  return mod


wbw = _load_real_weight_bins_writer()


class PathSafetyTest(unittest.TestCase):

  def setUp(self):
    self._tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self._tmp.cleanup)
    self.output_dir = self._tmp.name

  def test_relative_traversal_rejected(self):
    with self.assertRaises(ValueError):
      wbw._safe_join(self.output_dir, "../escape.bin")

  def test_absolute_path_rejected(self):
    with self.assertRaises(ValueError):
      wbw._safe_join(self.output_dir, "/tmp/escape.bin")

  def test_deep_traversal_rejected(self):
    with self.assertRaises(ValueError):
      wbw._safe_join(self.output_dir, "../../../../../tmp/escape.bin")

  def test_embedded_separator_rejected(self):
    with self.assertRaises(ValueError):
      wbw._safe_join(self.output_dir, "subdir/file.bin")

  def test_backslash_separator_rejected(self):
    with self.assertRaises(ValueError):
      wbw._safe_join(self.output_dir, "subdir\\file.bin")

  def test_safe_names_accepted(self):
    for name in ["layer0.w", "emb-table_1.bin", "alpha-beta", "a.b.c"]:
      got = wbw._safe_join(self.output_dir, name)
      self.assertTrue(got.startswith(os.path.realpath(self.output_dir)))


if __name__ == "__main__":
  unittest.main()
