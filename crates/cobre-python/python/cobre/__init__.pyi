from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from . import errors as errors
from . import io as io
from . import model as model
from . import results as results
from . import run as run
from . import schema as schema
from ._types import HydroModelsSummary, ProvenanceReport, StochasticSummary
from .model import System

__version__: str

def version_info() -> dict[str, Any]: ...
def write_policy_checkpoint(
    path: Union[str, Path],
    stage_cuts: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    stage_bases: Optional[Sequence[Mapping[str, Any]]] = None,
    stage_states: Optional[Sequence[Mapping[str, Any]]] = None,
    inflow_lag_depth: Optional[int] = None,
) -> None:
    """Write a policy checkpoint from plain Python dicts.

    metadata may include an optional "season_manifest" dict with keys:
    - "cycle_code": int (0 monthly / 1 weekly / 2 custom / 255 absent)
    - "n_seasons": int
    - "hydro_orders": list of {"hydro_id": int, "orders": list[int]} dicts

    Omitted, the checkpoint carries the absent descriptor (cycle_code=255).

    The checkpoint always records the running cobre version; a "cobre_version"
    key in metadata is ignored.
    """
    ...

class Study:
    def __init__(
        self,
        case_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        threads: Optional[int] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None: ...
    @property
    def output_dir(self) -> str: ...
    @property
    def system(self) -> System: ...
    @property
    def stochastic(self) -> StochasticSummary: ...
    @property
    def hydro_models(self) -> HydroModelsSummary: ...
    @property
    def provenance(self) -> ProvenanceReport: ...
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
