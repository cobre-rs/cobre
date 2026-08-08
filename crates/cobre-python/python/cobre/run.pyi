from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Union

from ._types import RunResult

def run(
    case_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    threads: Optional[int] = None,
    skip_simulation: Optional[bool] = None,
    config_overrides: Optional[Mapping[str, Any]] = None,
    on_iteration: Optional[Callable[[dict[str, Any]], Optional[bool]]] = None,
    cpu_bind: Optional[Literal["none", "core", "numa"]] = None,
) -> RunResult: ...
