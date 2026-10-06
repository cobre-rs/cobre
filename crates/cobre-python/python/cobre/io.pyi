from pathlib import Path
from typing import Any, Mapping, Optional, Union

from . import model as model

def load_case(path: Union[str, Path]) -> model.System: ...
def validate(
    path: Union[str, Path],
    config_overrides: Optional[Mapping[str, Any]] = None,
    *,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]: ...
