"""Canonical data layer: schema, provenance, storage, campaigns and splits."""

from pcsp.data.campaign import CampaignResult, RunRecord, load_campaign, read_manifest, run_campaign
from pcsp.data.doe import ParameterSpace, Range, full_factorial, latin_hypercube, sobol
from pcsp.data.io import load, save
from pcsp.data.provenance import collect_provenance, is_stamped, stamp
from pcsp.data.schema import (
    COMPONENTS,
    FIDELITIES,
    SCHEMA_VERSION,
    Coord,
    SchemaError,
    Var,
    get_params,
    new_dataset,
    param_hash,
    validate,
)
from pcsp.data.splits import (
    Box,
    Split,
    SplitError,
    Threshold,
    apply_split,
    check_folds,
    check_split,
    holdout_split,
    kfold_split,
    random_split,
)

__all__ = [
    "COMPONENTS",
    "FIDELITIES",
    "SCHEMA_VERSION",
    "Box",
    "CampaignResult",
    "Coord",
    "ParameterSpace",
    "Range",
    "RunRecord",
    "SchemaError",
    "Split",
    "SplitError",
    "Threshold",
    "Var",
    "apply_split",
    "check_folds",
    "check_split",
    "collect_provenance",
    "full_factorial",
    "get_params",
    "holdout_split",
    "is_stamped",
    "kfold_split",
    "latin_hypercube",
    "load",
    "load_campaign",
    "new_dataset",
    "param_hash",
    "random_split",
    "read_manifest",
    "run_campaign",
    "save",
    "sobol",
    "stamp",
    "validate",
]
