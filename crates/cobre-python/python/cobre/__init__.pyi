from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union

from . import errors as errors
from . import io as io
from . import model as model
from . import results as results
from . import run as run
from . import schema as schema
from .model import System

__version__: str

def version_info() -> dict[str, Any]: ...

class Study:
    def __init__(
        self,
        case_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        threads: Optional[int] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        cpu_bind: Optional[Literal["none", "core", "numa"]] = None,
    ) -> None: ...
    @property
    def output_dir(self) -> str: ...
    @property
    def system(self) -> System: ...
    def validate(self) -> dict[str, Any]: ...
    def train(
        self,
        on_iteration: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> "Policy": ...
    def load_policy(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "Policy": ...
    def simulate(
        self,
        policy: "Policy",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> dict[str, Any]: ...

class Policy:
    @property
    def iterations(self) -> int: ...
    @property
    def final_lower_bound(self) -> float: ...
    @property
    def final_upper_bound(self) -> float: ...
    def evaluate(self, stage: int, state: Sequence[float]) -> float: ...
    def cut_matrix(self, stage: int) -> tuple[Any, Any]: ...
