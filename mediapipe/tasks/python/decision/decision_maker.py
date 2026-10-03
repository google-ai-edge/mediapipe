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

import ctypes
import dataclasses
from typing import Dict, List, Optional, Sequence, Union

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


class _MpDecisionMakerOptionsC(ctypes.Structure):
  _fields_ = [
      ("base_options", base_options_c_module.MpBaseOptionsC),
      ("max_num_tokens", ctypes.c_int),
  ]


class ScoringMode:
  SCORING_AUTO = 0
  SCORING_SINGLE_TOKEN = 1
  SCORING_FULL_OPTION = 2


@dataclasses.dataclass
class DecisionContext:
  text: str = ""


# For all question types, `temperature <= 0` (the default) uses the model's
# calibrated temperature; a positive value overrides it.


@dataclasses.dataclass
class BooleanQuestion:
  condition: str
  threshold: float = 0.5
  temperature: float = 0.0
  normalize_prior: bool = False


@dataclasses.dataclass
class ChoiceQuestion:
  criteria: Dict[str, str]
  temperature: float = 0.0
  scoring_mode: int = ScoringMode.SCORING_AUTO
  normalize_prior: bool = False
  instructions: str = ""


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
  sorted_items = sorted(
      probabilities.items(), key=lambda kv: kv[1], reverse=True
  )
  pred_set: List[str] = []
  cum = 0.0
  for k, p in sorted_items:
    pred_set.append(k)
    cum += p
    if cum >= coverage:
      break
  return pred_set


QuestionType = Union[BooleanQuestion, ChoiceQuestion, ScoreQuestion]
QuestionResultType = Union[BooleanResult, ChoiceResult, ScoreResult]
DecisionResult = Dict[str, QuestionResultType]


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
    mediapipe_c_utils.CStatusFunction(
        "MpDecisionMakerPrewarmChoice",
        [
            ctypes.c_void_p,
            ctypes.POINTER(_MpChoiceQuestionC),
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

  def prewarm_choice(self, question: ChoiceQuestion) -> None:
    """Prewarms the static prompt prefix and KV cache for a choice question."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")
    keys = list(question.criteria.keys())
    descs = [question.criteria[k] for k in keys]
    c_keys = (ctypes.c_char_p * len(keys))(*(k.encode("utf-8") for k in keys))
    c_descs = (ctypes.c_char_p * len(descs))(
        *(d.encode("utf-8") for d in descs)
    )
    q_c = _MpChoiceQuestionC(
        keys=c_keys,
        descriptions=c_descs,
        count=len(keys),
        temperature=question.temperature,
        instructions=(
            question.instructions.encode("utf-8")
            if question.instructions
            else None
        ),
        scoring_mode=question.scoring_mode,
        normalize_prior=question.normalize_prior,
    )
    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerPrewarmChoice(self._handle, ctypes.byref(q_c))

  def evaluate_boolean(
      self, text: str, question: BooleanQuestion
  ) -> BooleanResult:
    """Evaluates a Boolean against input text."""
    if not self._handle:
      raise RuntimeError("DecisionMaker is closed.")

    q_c = _MpBooleanQuestionC(
        condition=question.condition.encode("utf-8"),
        threshold=question.threshold,
        temperature=question.temperature,
        normalize_prior=question.normalize_prior,
    )
    res_c = _MpBooleanResultC()
    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerEvaluateBoolean(
        self._handle,
        text.encode("utf-8"),
        ctypes.byref(q_c),
        ctypes.byref(res_c),
    )

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

    keys = list(question.criteria.keys())
    descs = [question.criteria[k] for k in keys]

    c_keys = (ctypes.c_char_p * len(keys))(*(k.encode("utf-8") for k in keys))
    c_descs = (ctypes.c_char_p * len(descs))(
        *(d.encode("utf-8") for d in descs)
    )

    q_c = _MpChoiceQuestionC(
        keys=c_keys,
        descriptions=c_descs,
        count=len(keys),
        temperature=question.temperature,
        instructions=(
            question.instructions.encode("utf-8")
            if question.instructions
            else None
        ),
        scoring_mode=question.scoring_mode,
        normalize_prior=question.normalize_prior,
    )
    res_c = _MpChoiceResultC()

    # pyrefly: ignore[missing-attribute]
    self._lib.MpDecisionMakerEvaluateChoice(
        self._handle,
        text.encode("utf-8"),
        ctypes.byref(q_c),
        ctypes.byref(res_c),
    )

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
    q_c = _MpBooleanQuestionC(
        condition=question.condition.encode("utf-8"),
        threshold=question.threshold,
        temperature=question.temperature,
        normalize_prior=question.normalize_prior,
    )
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
    keys = list(question.criteria.keys())
    descs = [question.criteria[k] for k in keys]
    c_keys = (ctypes.c_char_p * len(keys))(*(k.encode("utf-8") for k in keys))
    c_descs = (ctypes.c_char_p * len(descs))(
        *(d.encode("utf-8") for d in descs)
    )

    q_c = _MpChoiceQuestionC(
        keys=c_keys,
        descriptions=c_descs,
        count=len(keys),
        temperature=question.temperature,
        instructions=(
            question.instructions.encode("utf-8")
            if question.instructions
            else None
        ),
        scoring_mode=question.scoring_mode,
        normalize_prior=question.normalize_prior,
    )
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
