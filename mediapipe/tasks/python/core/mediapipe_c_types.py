"""Common MediaPipe CTypes definitions."""

import dataclasses
from typing import Any, List


@dataclasses.dataclass(frozen=True)
class CFunction:
  """Stores the ctypes signature information for a C function.

  Attributes:
    func_name: The name of the C function.
    argtypes: A list of ctypes types for the function arguments.
    restype: The ctypes type for the function's return value.
  """

  func_name: str
  argtypes: List[Any]
  restype: Any
