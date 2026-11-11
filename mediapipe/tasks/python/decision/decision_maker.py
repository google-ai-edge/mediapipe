# Copyright 2026 The MediaPipe Authors.
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
"""MediaPipe semantic decision maker task."""

from __future__ import annotations

import collections.abc
import ctypes
import dataclasses
import enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from mediapipe.tasks.python.core import (
    base_options as base_options_module,
)
from mediapipe.tasks.python.core import (
    base_options_c as base_options_c_module,
)
from mediapipe.tasks.python.core import mediapipe_c_bindings
from mediapipe.tasks.python.core import mediapipe_c_utils
from mediapipe.tasks.python.core import serial_dispatcher


class _MpBooleanQuestionC(ctypes.Structure):
  _fields_ = [
      ("condition", ctypes.c_char_p),
      ("threshold", ctypes.c_float),
      ("temperature", ctypes.c_float),
      ("normalize_prior", ctypes.c_bool),
      ("positive_examples", ctypes.POINTER(ctypes.c_char_p)),
      ("positive_examples_count", ctypes.c_int),
      ("negative_examples", ctypes.POINTER(ctypes.c_char_p)),
      ("negative_examples_count", ctypes.c_int),
      ("input_prompt", ctypes.c_int),
  ]


class _MpChoiceQuestionC(ctypes.Structure):
  _fields_ = [
      ("keys", ctypes.POINTER(ctypes.c_char_p)),
      ("descriptions", ctypes.POINTER(ctypes.c_char_p)),
      ("count", ctypes.c_int),
      ("temperature", ctypes.c_float),
      ("instructions", ctypes.c_char_p),
      ("scoring_mode", ctypes.c_int),
      ("normalize_prior", ctypes.c_bool),
      ("image_bytes", ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))),
      ("image_bytes_lengths", ctypes.POINTER(ctypes.c_int)),
      ("audio_samples", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
      ("audio_samples_lengths", ctypes.POINTER(ctypes.c_int)),
      ("examples", ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p))),
      ("example_counts", ctypes.POINTER(ctypes.c_int)),
      ("input_prompt", ctypes.c_int),
      ("omit_balanced_class_weights", ctypes.c_bool),
  ]


class _MpScoreQuestionC(ctypes.Structure):
  _fields_ = [
      ("rubric", ctypes.POINTER(ctypes.c_char_p)),
      ("count", ctypes.c_int),
      ("temperature", ctypes.c_float),
      ("instructions", ctypes.c_char_p),
  ]


class _MpBooleanResultC(ctypes.Structure):
  _fields_ = [
      ("value", ctypes.c_bool),
      ("probability_true", ctypes.c_float),
      ("confidence", ctypes.c_float),
  ]


class _MpChoiceResultC(ctypes.Structure):
  _fields_ = [
      ("selected_key", ctypes.c_char_p),
      ("keys", ctypes.POINTER(ctypes.c_char_p)),
      ("probabilities", ctypes.POINTER(ctypes.c_float)),
      ("count", ctypes.c_int),
      ("confidence", ctypes.c_float),
  ]


class _MpScoreResultC(ctypes.Structure):
  _fields_ = [
      ("selected_key", ctypes.c_char_p),
      ("expected_score", ctypes.c_float),
      ("probabilities", ctypes.POINTER(ctypes.c_float)),
      ("count", ctypes.c_int),
      ("confidence", ctypes.c_float),
  ]


_MP_QUESTION_TYPE_BOOLEAN = 0
_MP_QUESTION_TYPE_CHOICE = 1
_MP_QUESTION_TYPE_SCORE = 2


class _MpSchemaQuestionC(ctypes.Structure):
  _fields_ = [
      ("name", ctypes.c_char_p),
      ("question_type", ctypes.c_int),
      ("boolean_question", _MpBooleanQuestionC),
      ("choice_question", _MpChoiceQuestionC),
      ("score_question", _MpScoreQuestionC),
  ]


class _MpSchemaQuestionResultC(ctypes.Structure):
  _fields_ = [
      ("question_type", ctypes.c_int),
      ("boolean_result", _MpBooleanResultC),
      ("choice_result", _MpChoiceResultC),
      ("score_result", _MpScoreResultC),
  ]


class _MpDecisionMakerOptionsC(ctypes.Structure):
  _fields_ = [
      ("base_options", base_options_c_module.MpBaseOptionsC),
      ("max_num_tokens", ctypes.c_int),
  ]


class ScoringMode:
  SCORING_AUTO = 0
  SCORING_SINGLE_TOKEN = 1
  SCORING_FULL_OPTION = 2


class InputPrompt(enum.IntEnum):
  """How examples and inputs are embedded. Only used with examples."""

  # "task: classification | query: <text>". Best for sentences.
  CLASSIFICATION = 0
  # The raw text. Best for single words and short phrases.
  NONE = 1


@dataclasses.dataclass
class DecisionContext:
  text: str = ""


# For all question types, `temperature <= 0` (the default) uses the model's
# calibrated temperature; a positive value overrides it.


def _check_example_list(name: str, examples: Sequence[str]) -> None:
  # A one-shot iterable would be used up here, before the question is built.
  if not isinstance(examples, collections.abc.Sequence) or isinstance(
      examples, (str, bytes)
  ):
    raise TypeError(f"{name} must be a sequence of strings.")
  for example in examples:
    if not isinstance(example, str):
      raise TypeError(
          f"{name} must contain only strings, not {type(example).__name__}."
      )


@dataclasses.dataclass
class BooleanQuestion:
  """A yes/no question.

  With both `positive_examples` and `negative_examples` (inputs for which the
  answer is true / false), embedding models fit a classifier on them and
  `condition` becomes optional. `input_prompt` is only used with examples.
  """

  condition: str = ""
  threshold: float = 0.5
  temperature: float = 0.0
  normalize_prior: bool = False
  positive_examples: Sequence[str] = ()
  negative_examples: Sequence[str] = ()
  input_prompt: InputPrompt = InputPrompt.CLASSIFICATION

  def __post_init__(self):
    _check_example_list("positive_examples", self.positive_examples)
    _check_example_list("negative_examples", self.negative_examples)
    if bool(self.positive_examples) != bool(self.negative_examples):
      raise ValueError(
          "A BooleanQuestion with examples needs both positive_examples and"
          " negative_examples."
      )


@dataclasses.dataclass
class ChoiceQuestion:
  """A question with one answer out of several choices.

  `criteria` maps each choice key to its description. With `examples` (choice
  key to example inputs), embedding models fit a classifier on them; then every
  choice needs at least one example, and `criteria` may be left out.
  `input_prompt` and `balance_class_weights` are only used with examples.
  """

  criteria: Dict[str, str] = dataclasses.field(default_factory=dict)
  temperature: float = 0.0
  scoring_mode: int = ScoringMode.SCORING_AUTO
  normalize_prior: bool = False
  instructions: str = ""
  examples: Optional[Dict[str, Sequence[str]]] = None
  input_prompt: InputPrompt = InputPrompt.CLASSIFICATION
  # With examples, weights each choice by N / (K * N_c), so that choices with
  # fewer examples aren't penalized.
  balance_class_weights: bool = True

  def __post_init__(self):
    if not isinstance(self.criteria, Mapping):
      raise TypeError("criteria must be a mapping from choice key to text.")
    if self.examples is None:
      return
    if not isinstance(self.examples, Mapping):
      raise TypeError("examples must be a mapping from choice key to inputs.")
    for key, examples in self.examples.items():
      _check_example_list(f"examples['{key}']", examples)
    if not self.has_examples():
      # As in the C API, no example at all means a question without examples.
      return
    keys = self.criteria.keys() or self.examples.keys()
    if len(keys) < 2:
      raise ValueError(
          "A ChoiceQuestion with examples needs at least 2 choices."
      )
    if set(self.examples.keys()) != set(keys):
      raise ValueError("examples and criteria must list the same choices.")
    for key in keys:
      if not self.examples[key]:
        raise ValueError(f"Choice '{key}' needs at least one example.")

  def has_examples(self) -> bool:
    """Returns whether any choice has an example."""
    return any(self.examples.values()) if self.examples else False

  def choice_keys(self) -> List[str]:
    """Returns the choice keys, from `criteria` or else from `examples`.

    Without examples (`examples` empty or all lists empty), only `criteria`
    gives choices.
    """
    if self.criteria or not self.has_examples():
      return list(self.criteria.keys())
    return list((self.examples or {}).keys())


@dataclasses.dataclass
class ScoreQuestion:
  rubric: List[str]
  temperature: float = 0.0
  instructions: str = ""


@dataclasses.dataclass
class BooleanResult:
  value: bool
  probability_true: float
  confidence: float


@dataclasses.dataclass
class ChoiceResult:
  selected_key: str
  probabilities: Dict[str, float]
  confidence: float
  prediction_set: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ScoreResult:
  expected_score: float
  probabilities: List[float]
  confidence: float
  selected_key: str = ""


def _compute_prediction_set(
    probabilities: Dict[str, float], coverage: float = 0.90
) -> List[str]:
  """Computes the conformal prediction set covering `coverage` mass."""
  sorted_items = sorted(probabilities.items(), key=lambda kv: (-kv[1], kv[0]))
  pred_set: List[str] = []
  cum = 0.0
  clamped_coverage = min(1.0, max(0.0, coverage))
  for k, p in sorted_items:
    pred_set.append(k)
    cum += p
    if cum >= clamped_coverage:
      break
  return pred_set


QuestionType = Union[BooleanQuestion, ChoiceQuestion, ScoreQuestion]
QuestionResultType = Union[BooleanResult, ChoiceResult, ScoreResult]
DecisionResult = Dict[str, QuestionResultType]


def _encode_string_array(
    strings: Sequence[str],
) -> Tuple[Optional[ctypes.Array[ctypes.c_char_p]], List[bytes]]:
  """Returns a `char*` array of UTF-8 `strings` (or None) and its buffers."""
  encoded = [s.encode("utf-8") for s in strings]
  if not encoded:
    return None, encoded
  return (ctypes.c_char_p * len(encoded))(*encoded), encoded


def _build_boolean_question_c(
    question: BooleanQuestion,
) -> Tuple[_MpBooleanQuestionC, List[object]]:
  """Builds `_MpBooleanQuestionC` and the objects it points into."""
  cond_bytes = question.condition.encode("utf-8")
  c_pos, pos_bytes = _encode_string_array(question.positive_examples)
  c_neg, neg_bytes = _encode_string_array(question.negative_examples)
  q_c = _MpBooleanQuestionC(
      condition=cond_bytes,
      threshold=question.threshold,
      temperature=question.temperature,
      normalize_prior=question.normalize_prior,
      positive_examples=c_pos,
      positive_examples_count=len(pos_bytes),
      negative_examples=c_neg,
      negative_examples_count=len(neg_bytes),
      input_prompt=int(question.input_prompt),
  )
  return q_c, [cond_bytes, c_pos, pos_bytes, c_neg, neg_bytes]


def _build_choice_question_c(
    question: ChoiceQuestion,
) -> Tuple[_MpChoiceQuestionC, List[object]]:
  """Builds `_MpChoiceQuestionC` and the objects it points into."""
  keys = question.choice_keys()
  c_keys, key_bytes = _encode_string_array(keys)
  c_descs, desc_bytes = _encode_string_array(
      [question.criteria.get(k, "") for k in keys]
  )
  inst_bytes = (
      question.instructions.encode("utf-8") if question.instructions else None
  )
  keepalive: List[object] = [c_keys, key_bytes, c_descs, desc_bytes, inst_bytes]
  c_examples = None
  c_counts = None
  examples = question.examples
  if examples is not None and question.has_examples() and keys:
    per_choice = [_encode_string_array(examples[k]) for k in keys]
    keepalive.append(per_choice)
    c_examples = (ctypes.POINTER(ctypes.c_char_p) * len(keys))(*[
        ctypes.cast(arr, ctypes.POINTER(ctypes.c_char_p))
        if arr is not None
        else ctypes.POINTER(ctypes.c_char_p)()
        for arr, _ in per_choice
    ])
    c_counts = (ctypes.c_int * len(keys))(
        *[len(encoded) for _, encoded in per_choice]
    )
    keepalive.extend([c_examples, c_counts])
  q_c = _MpChoiceQuestionC(
      keys=c_keys,
      descriptions=c_descs,
      count=len(keys),
      temperature=question.temperature,
      instructions=inst_bytes,
      scoring_mode=question.scoring_mode,
      normalize_prior=question.normalize_prior,
      examples=c_examples,
      example_counts=c_counts,
      input_prompt=int(question.input_prompt),
      omit_balanced_class_weights=not question.balance_class_weights,
  )
  return q_c, keepalive


def _build_schema_questions_c(
    questions: Mapping[str, QuestionType],
) -> Tuple[ctypes.Array[_MpSchemaQuestionC], List[str], List[object]]:
  """Builds a ctypes array of `_MpSchemaQuestionC` and a GC keepalive list."""
  names = list(questions.keys())
  n = len(names)
  q_arr = (_MpSchemaQuestionC * n)()
  keepalive: List[object] = []
  for i, name in enumerate(names):
    if not name:
      raise ValueError("Schema question name must not be empty.")
    name_bytes = name.encode("utf-8")
    keepalive.append(name_bytes)
    q_arr[i].name = name_bytes
    question = questions[name]
    if isinstance(question, BooleanQuestion):
      q_c, q_keepalive = _build_boolean_question_c(question)
      keepalive.append(q_keepalive)
      q_arr[i].question_type = _MP_QUESTION_TYPE_BOOLEAN
      q_arr[i].boolean_question = q_c
    elif isinstance(question, ChoiceQuestion):
      q_c, q_keepalive = _build_choice_question_c(question)
      keepalive.append(q_keepalive)
      q_arr[i].question_type = _MP_QUESTION_TYPE_CHOICE
      q_arr[i].choice_question = q_c
    elif isinstance(question, ScoreQuestion):
      rubric_bytes = [r.encode("utf-8") for r in question.rubric]
      c_rubric = (ctypes.c_char_p * len(question.rubric))(*rubric_bytes)
      inst_bytes = (
          question.instructions.encode("utf-8")
          if question.instructions
          else None
      )
      keepalive.extend([rubric_bytes, c_rubric, inst_bytes])
      q_arr[i].question_type = _MP_QUESTION_TYPE_SCORE
      q_arr[i].score_question = _MpScoreQuestionC(
          rubric=c_rubric,
          count=len(question.rubric),
          temperature=question.temperature,
          instructions=inst_bytes,
      )
    else:
      raise TypeError(f"Unsupported question type: {type(question)}")
  return q_arr, names, keepalive


@dataclasses.dataclass
class DecisionMakerOptions:
  """Options for configuring the Decision task."""

  base_options: Optional[base_options_module.BaseOptions] = None
  max_num_tokens: int = 4096

  def to_ctypes(self) -> _MpDecisionMakerOptionsC:
    """Converts the options to a ctypes struct."""
    # Always go through `BaseOptions.to_ctypes()` so that the host environment,
    # host version and CA bundle path used for usage logging are populated
    # even when no base options are provided.
    base_options = self.base_options or base_options_module.BaseOptions()
    base_options_c = base_options.to_ctypes()
    return _MpDecisionMakerOptionsC(
        base_options=base_options_c,
        max_num_tokens=self.max_num_tokens,
    )


_CTYPES_SIGNATURES = (
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerCreate",
        [
            ctypes.POINTER(_MpDecisionMakerOptionsC),
            ctypes.POINTER(ctypes.c_void_p),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerEvaluateBoolean",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_MpBooleanQuestionC),
            ctypes.POINTER(_MpBooleanResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerEvaluateChoice",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_MpChoiceQuestionC),
            ctypes.POINTER(_MpChoiceResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerEvaluateScore",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_MpScoreQuestionC),
            ctypes.POINTER(_MpScoreResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerEvaluateSchema",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_MpSchemaQuestionC),
            ctypes.c_int,
            ctypes.POINTER(_MpSchemaQuestionResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerEvaluateBooleanBatch",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.POINTER(_MpBooleanQuestionC),
            ctypes.POINTER(_MpBooleanResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerEvaluateChoiceBatch",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.POINTER(_MpChoiceQuestionC),
            ctypes.POINTER(_MpChoiceResultC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerEvaluateScoreBatch",
        [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.POINTER(_MpScoreQuestionC),
            ctypes.POINTER(_MpScoreResultC),
        ],
    ),
    mediapipe_c_utils.CFunction(
        "MpDecisionMakerCloseChoiceResult",
        [ctypes.POINTER(_MpChoiceResultC)],
        None,
    ),
    mediapipe_c_utils.CFunction(
        "MpDecisionMakerCloseChoiceResultBatch",
        [ctypes.POINTER(_MpChoiceResultC), ctypes.c_int],
        None,
    ),
    mediapipe_c_utils.CFunction(
        "MpDecisionMakerCloseScoreResult",
        [ctypes.POINTER(_MpScoreResultC)],
        None,
    ),
    mediapipe_c_utils.CFunction(
        "MpDecisionMakerCloseScoreResultBatch",
        [ctypes.POINTER(_MpScoreResultC), ctypes.c_int],
        None,
    ),
    mediapipe_c_utils.CFunction(
        "MpDecisionMakerCloseSchemaResult",
        [ctypes.POINTER(_MpSchemaQuestionResultC), ctypes.c_int],
        None,
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerPrewarmBoolean",
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpBooleanQuestionC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerPrewarmChoice",
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpChoiceQuestionC),
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerPrewarmSchema",
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpSchemaQuestionC),
            ctypes.c_int,
        ],
    ),
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerClose",
        [
            ctypes.c_void_p,
        ],
    ),
)


class DecisionMaker:
  """MediaPipe Decision task."""

  def __init__(
      self,
      lib: serial_dispatcher.SerialDispatcher,
      handle: ctypes.c_void_p,
  ):
    self._lib = lib
    self._handle = handle

  @classmethod
  def create_from_model_path(cls, model_path: str) -> "DecisionMaker":
    """Creates a Decision task from a model file path."""
    base_options = base_options_module.BaseOptions(model_asset_path=model_path)
    options = DecisionMakerOptions(base_options=base_options)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(
      cls, options: Optional[DecisionMakerOptions] = None
  ) -> "DecisionMaker":
    """Creates a Decision task from options."""
    if options is None:
      options = DecisionMakerOptions()

    lib = mediapipe_c_bindings.load_shared_library(_CTYPES_SIGNATURES)
    options_c = options.to_ctypes()
    handle = ctypes.c_void_p()

    lib.MpDecisionMakerCreate(  # pyrefly: ignore[missing-attribute]
        ctypes.byref(options_c), ctypes.byref(handle)
    )
    return cls(lib, handle)

  def prewarm_boolean(self, question: BooleanQuestion) -> None:
    """Prewarms a boolean question (for example, fits its examples)."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")
    q_c, keepalive = _build_boolean_question_c(question)
    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerPrewarmBoolean(self._handle, ctypes.byref(q_c))
    del keepalive

  def prewarm(self, question: ChoiceQuestion) -> None:
    """Prewarms the static prompt prefix and KV cache for a choice question."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")
    q_c, keepalive = _build_choice_question_c(question)
    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerPrewarmChoice(self._handle, ctypes.byref(q_c))
    del keepalive

  def prewarm_choice(self, question: ChoiceQuestion) -> None:
    """Deprecated alias for `prewarm`."""
    self.prewarm(question)

  def prewarm_schema(self, questions: Mapping[str, QuestionType]) -> None:
    """Prewarms the prompt prefix and caches for a multi-question schema."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")
    if not questions:
      return
    q_arr, _, keepalive = _build_schema_questions_c(questions)
    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerPrewarmSchema(self._handle, q_arr, len(questions))
    del keepalive

  def evaluate_schema(
      self,
      text: Union[str, DecisionContext],
      questions: Mapping[str, QuestionType],
  ) -> DecisionResult:
    """Evaluates a multi-question schema simultaneously against input text."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")
    if not questions:
      return {}
    text_str = text.text if isinstance(text, DecisionContext) else text
    text_bytes = text_str.encode("utf-8")
    q_arr, names, keepalive = _build_schema_questions_c(questions)
    n = len(names)
    res_arr = (_MpSchemaQuestionResultC * n)()

    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerEvaluateSchema(
        self._handle,
        text_bytes,
        q_arr,
        n,
        res_arr,
    )
    del keepalive

    try:
      out: DecisionResult = {}
      for i, name in enumerate(names):
        q_type = res_arr[i].question_type
        if q_type == _MP_QUESTION_TYPE_BOOLEAN:
          b_res = res_arr[i].boolean_result
          out[name] = BooleanResult(
              value=b_res.value,
              probability_true=b_res.probability_true,
              confidence=b_res.confidence,
          )
        elif q_type == _MP_QUESTION_TYPE_CHOICE:
          c_res = res_arr[i].choice_result
          probs = {
              c_res.keys[j].decode("utf-8"): c_res.probabilities[j]
              for j in range(c_res.count)
          }
          selected = (
              c_res.selected_key.decode("utf-8") if c_res.selected_key else ""
          )
          out[name] = ChoiceResult(
              selected_key=selected,
              probabilities=probs,
              confidence=c_res.confidence,
              prediction_set=_compute_prediction_set(probs),
          )
        elif q_type == _MP_QUESTION_TYPE_SCORE:
          s_res = res_arr[i].score_result
          probs = [s_res.probabilities[j] for j in range(s_res.count)]
          selected = (
              s_res.selected_key.decode("utf-8") if s_res.selected_key else ""
          )
          out[name] = ScoreResult(
              expected_score=s_res.expected_score,
              probabilities=probs,
              confidence=s_res.confidence,
              selected_key=selected,
          )
      return out
    finally:
      # pyrefly: ignore[missing-attribute]
      self._lib.MpDecisionMakerCloseSchemaResult(res_arr, n)

  def evaluate_boolean(
      self, text: str, question: BooleanQuestion
  ) -> BooleanResult:
    """Evaluates a Boolean against input text."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")

    q_c, keepalive = _build_boolean_question_c(question)
    res_c = _MpBooleanResultC()
    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerEvaluateBoolean(
        self._handle,
        text.encode("utf-8"),
        ctypes.byref(q_c),
        ctypes.byref(res_c),
    )
    del keepalive

    return BooleanResult(
        value=res_c.value,
        probability_true=res_c.probability_true,
        confidence=res_c.confidence,
    )

  def evaluate_choice(
      self, text: str, question: ChoiceQuestion
  ) -> ChoiceResult:
    """Evaluates a Choice against input text."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")

    q_c, keepalive = _build_choice_question_c(question)
    res_c = _MpChoiceResultC()

    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerEvaluateChoice(
        self._handle,
        text.encode("utf-8"),
        ctypes.byref(q_c),
        ctypes.byref(res_c),
    )
    del keepalive

    try:
      probs = {}
      for i in range(res_c.count):
        k = res_c.keys[i].decode("utf-8")
        probs[k] = res_c.probabilities[i]

      selected = (
          res_c.selected_key.decode("utf-8") if res_c.selected_key else ""
      )
      conf = res_c.confidence
    finally:
      # pyrefly: ignore[missing-attribute]
      self._lib.MpDecisionMakerCloseChoiceResult(ctypes.byref(res_c))

    return ChoiceResult(
        selected_key=selected,
        probabilities=probs,
        confidence=conf,
        prediction_set=_compute_prediction_set(probs),
    )

  def evaluate_score(self, text: str, question: ScoreQuestion) -> ScoreResult:
    """Evaluates an Score rubric against input text."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")

    c_rubric = (ctypes.c_char_p * len(question.rubric))(
        *(r.encode("utf-8") for r in question.rubric)
    )

    q_c = _MpScoreQuestionC(
        rubric=c_rubric,
        count=len(question.rubric),
        temperature=question.temperature,
        instructions=(
            question.instructions.encode("utf-8")
            if question.instructions
            else None
        ),
    )
    res_c = _MpScoreResultC()

    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerEvaluateScore(
        self._handle,
        text.encode("utf-8"),
        ctypes.byref(q_c),
        ctypes.byref(res_c),
    )

    try:
      probs = [res_c.probabilities[i] for i in range(res_c.count)]
      expected = res_c.expected_score
      conf = res_c.confidence
      selected = (
          res_c.selected_key.decode("utf-8") if res_c.selected_key else ""
      )
    finally:
      # pyrefly: ignore[missing-attribute]
      self._lib.MpDecisionMakerCloseScoreResult(ctypes.byref(res_c))

    return ScoreResult(
        expected_score=expected,
        probabilities=probs,
        confidence=conf,
        selected_key=selected,
    )

  def evaluate_boolean_batch(
      self,
      texts: Sequence[Union[str, DecisionContext]],
      question: BooleanQuestion,
      shared_prefix: Optional[str] = None,
  ) -> List[BooleanResult]:
    """Evaluates a Boolean against a batch of input texts."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")
    n = len(texts)
    if n == 0:
      return []

    str_texts = [t.text if isinstance(t, DecisionContext) else t for t in texts]
    c_texts = (ctypes.c_char_p * n)(*(t.encode("utf-8") for t in str_texts))
    q_c, keepalive = _build_boolean_question_c(question)
    res_arr = (_MpBooleanResultC * n)()
    prefix_c = shared_prefix.encode("utf-8") if shared_prefix else None

    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerEvaluateBooleanBatch(
        self._handle,
        prefix_c,
        c_texts,
        n,
        ctypes.byref(q_c),
        res_arr,
    )
    del keepalive
    return [
        BooleanResult(
            value=res_arr[i].value,
            probability_true=res_arr[i].probability_true,
            confidence=res_arr[i].confidence,
        )
        for i in range(n)
    ]

  def evaluate_choice_batch(
      self,
      texts: Sequence[Union[str, DecisionContext]],
      question: ChoiceQuestion,
      shared_prefix: Optional[str] = None,
  ) -> List[ChoiceResult]:
    """Evaluates a Choice against a batch of input texts."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")
    n = len(texts)
    if n == 0:
      return []

    str_texts = [t.text if isinstance(t, DecisionContext) else t for t in texts]
    c_texts = (ctypes.c_char_p * n)(*(t.encode("utf-8") for t in str_texts))
    q_c, keepalive = _build_choice_question_c(question)
    res_arr = (_MpChoiceResultC * n)()
    prefix_c = shared_prefix.encode("utf-8") if shared_prefix else None

    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerEvaluateChoiceBatch(
        self._handle,
        prefix_c,
        c_texts,
        n,
        ctypes.byref(q_c),
        res_arr,
    )
    del keepalive

    try:
      results = []
      for i in range(n):
        probs = {}
        for j in range(res_arr[i].count):
          k = res_arr[i].keys[j].decode("utf-8")
          probs[k] = res_arr[i].probabilities[j]
        selected = (
            res_arr[i].selected_key.decode("utf-8")
            if res_arr[i].selected_key
            else ""
        )
        conf = res_arr[i].confidence
        results.append(
            ChoiceResult(
                selected_key=selected,
                probabilities=probs,
                confidence=conf,
                prediction_set=_compute_prediction_set(probs),
            )
        )
    finally:
      # pyrefly: ignore[missing-attribute]
      self._lib.MpDecisionMakerCloseChoiceResultBatch(res_arr, n)
    return results

  def evaluate_score_batch(
      self,
      texts: Sequence[Union[str, DecisionContext]],
      question: ScoreQuestion,
      shared_prefix: Optional[str] = None,
  ) -> List[ScoreResult]:
    """Evaluates an Score rubric against a batch of input texts."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")
    n = len(texts)
    if n == 0:
      return []

    str_texts = [t.text if isinstance(t, DecisionContext) else t for t in texts]
    c_texts = (ctypes.c_char_p * n)(*(t.encode("utf-8") for t in str_texts))
    c_rubric = (ctypes.c_char_p * len(question.rubric))(
        *(r.encode("utf-8") for r in question.rubric)
    )
    q_c = _MpScoreQuestionC(
        rubric=c_rubric,
        count=len(question.rubric),
        temperature=question.temperature,
        instructions=(
            question.instructions.encode("utf-8")
            if question.instructions
            else None
        ),
    )
    res_arr = (_MpScoreResultC * n)()
    prefix_c = shared_prefix.encode("utf-8") if shared_prefix else None

    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerEvaluateScoreBatch(
        self._handle,
        prefix_c,
        c_texts,
        n,
        ctypes.byref(q_c),
        res_arr,
    )

    try:
      results = []
      for i in range(n):
        probs = [res_arr[i].probabilities[j] for j in range(res_arr[i].count)]
        expected = res_arr[i].expected_score
        conf = res_arr[i].confidence
        selected = (
            res_arr[i].selected_key.decode("utf-8")
            if res_arr[i].selected_key
            else ""
        )
        results.append(
            ScoreResult(
                expected_score=expected,
                probabilities=probs,
                confidence=conf,
                selected_key=selected,
            )
        )
    finally:
      # pyrefly: ignore[missing-attribute]
      self._lib.MpDecisionMakerCloseScoreResultBatch(res_arr, n)
    return results

  def evaluate_batch(
      self,
      contexts: Sequence[Union[str, DecisionContext]],
      questions: Dict[str, QuestionType],
  ) -> List[DecisionResult]:
    """Evaluates N independent contexts against M questions in a batch."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")
    n = len(contexts)
    if n == 0:
      return []

    batch_results: List[DecisionResult] = [{} for _ in range(n)]
    for q_name, question in questions.items():
      if isinstance(question, BooleanQuestion):
        q_results = self.evaluate_boolean_batch(contexts, question)
      elif isinstance(question, ChoiceQuestion):
        q_results = self.evaluate_choice_batch(contexts, question)
      elif isinstance(question, ScoreQuestion):
        q_results = self.evaluate_score_batch(contexts, question)
      else:
        raise TypeError(f"Unsupported question type: {type(question)}")
      for i in range(n):
        batch_results[i][q_name] = q_results[i]
    return batch_results

  def evaluate_candidates(
      self,
      shared_context: Union[str, DecisionContext],
      candidates: Sequence[Union[str, DecisionContext]],
      questions: Dict[str, QuestionType],
  ) -> List[DecisionResult]:
    """Evaluates 1 shared prefix + N candidates against M questions."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")
    n = len(candidates)
    if n == 0:
      return []

    prefix_str = (
        shared_context.text
        if isinstance(shared_context, DecisionContext)
        else shared_context
    )
    batch_results: List[DecisionResult] = [{} for _ in range(n)]
    for q_name, question in questions.items():
      if isinstance(question, BooleanQuestion):
        q_results = self.evaluate_boolean_batch(
            candidates, question, shared_prefix=prefix_str
        )
      elif isinstance(question, ChoiceQuestion):
        q_results = self.evaluate_choice_batch(
            candidates, question, shared_prefix=prefix_str
        )
      elif isinstance(question, ScoreQuestion):
        q_results = self.evaluate_score_batch(
            candidates, question, shared_prefix=prefix_str
        )
      else:
        raise TypeError(f"Unsupported question type: {type(question)}")
      for i in range(n):
        batch_results[i][q_name] = q_results[i]
    return batch_results

  def close(self):
    """Releases resources associated with the task."""
    if self._handle:
      # pyrefly: ignore[missing-attribute]
      self._lib.MpDecisionMakerClose(self._handle)
      self._handle = None  # pyrefly: ignore[bad-assignment]
      self._lib.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
