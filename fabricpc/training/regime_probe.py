"""
``RegimeProbe``: track the ε-Hessian spectrum, the ePC regime, and the
weight norms during ``train``.

``EPCInference``'s safe range of ``eta_infer`` and ``infer_steps`` is set by
the curvature of the energy in error coordinates, and that curvature grows
with the weights during training: on the muPC ResNet-18 demo λ_max grew from
16 at init past 2/eta_infer, and the run collapsed to chance right after.
A verdict at init therefore describes init. ``RegimeProbe`` re-measures the
spectrum every ``every`` weight updates from inside ``train`` through the
``iter_callback`` hook, which hands the callback an ``IterContext`` carrying
the parameters after each update. Each probe records
:class:`~fabricpc.core.epsilon_spectrum.EpsilonSpectrum` on a fixed probe
batch (or the training batch), the :class:`~fabricpc.core.inference_epc.Regime`
flags when the trainer is PC with an ``EPCInference`` solver, the Frobenius
norm of every weight, and the batch's training energy; ``on_epoch`` adds a
row with the epoch's test accuracy so the collapse can be dated against the
spectrum. Under ``algorithm="backprop"`` the spectrum and the norms are
recorded and the regime columns stay empty, which is the control run that
separates "the curvature grows regardless of the solver" from "ePC's
relaxation drives the growth".

Cost. Supplying ``iter_callback`` to ``train`` forces a device sync on every
batch, probed or not (the trainer materializes the batch metrics to floats
before calling it); the probe itself costs one feedforward initialization
and ``iters`` Hessian-vector products on the probe batch every ``every``
updates, compiled once. The callback reads ``ctx.params`` only during the
call and stores floats, so it retains no device buffers.

CSV schema (``write_csv``, ``read_regime_csv``). Metadata columns first,
constant per row: ``trainer``, ``eta_infer``, ``infer_steps`` (empty under
backprop), ``every``, ``probe_batch`` (the probe batch size, or ``train``
when the training batch is probed). Then ``update`` (weight updates applied
so far), ``epoch`` (0-based), ``lambda_max``, ``lambda_min``, ``f_weighted``,
``f_max``, ``negative_weight``, ``growth_min``, ``residual_max``,
``residual_min``, ``eta_lambda_max``, ``output_gradient_reverses``,
``unstable``, ``train_energy``, ``test_accuracy``; then one
``wnorm:<key>`` column per weight, keyed by the edge key for edge-keyed
weights (``x->h1:in``) and by ``<node>.<key>`` otherwise. Probe rows leave
``test_accuracy`` empty; epoch rows leave the spectrum columns empty.
Readers take η, T, and the trainer from the columns; the file name is a
label.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.epsilon_spectrum import make_epsilon_spectrum
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.trainer import build_clamps

METADATA_COLUMNS = ("trainer", "eta_infer", "infer_steps", "every", "probe_batch")
SPECTRUM_COLUMNS = (
    "lambda_max",
    "lambda_min",
    "f_weighted",
    "f_max",
    "negative_weight",
    "growth_min",
    "residual_max",
    "residual_min",
    "eta_lambda_max",
    "output_gradient_reverses",
    "unstable",
)
DATA_COLUMNS = (
    ("update", "epoch") + SPECTRUM_COLUMNS + ("train_energy", "test_accuracy")
)
BOOL_COLUMNS = ("output_gradient_reverses", "unstable")
WNORM_PREFIX = "wnorm:"


def weight_norm_columns(params: GraphParams) -> Dict[str, Tuple[str, str]]:
    """``wnorm:<key>`` column name -> (node, weight key) for every weight in
    ``params``: the edge key alone when the key is one (it names source,
    target, and slot, so it is unique), ``<node>.<key>`` otherwise."""
    columns: Dict[str, Tuple[str, str]] = {}
    for node, node_params in params.nodes.items():
        for key in node_params.weights:
            label = key if "->" in key else f"{node}.{key}"
            columns[f"{WNORM_PREFIX}{label}"] = (node, key)
    return columns


class RegimeProbe:
    """Record the excited ε-spectrum, the regime flags, and the weight norms
    every ``every`` updates of a ``train`` run.

    Args:
        structure: the graph being trained.
        probe_clamps: fixed clamps (node name -> array) measured on every
            probe; ``None`` measures on the training batch through
            ``build_clamps(ctx.batch, structure, clamp_target=True)``.
        every: probe after every ``every``-th weight update.
        inference: the ``EPCInference`` whose (eta_infer, infer_steps) the
            regime judges; ``None`` reads ``structure.config["inference"]``
            when that is an ``EPCInference``. The regime is recorded only
            under ``algorithm="pc"`` with such a solver.
        iters: Lanczos steps per probe.
        key: PRNG key for the probe's latent initialization and the random
            Lanczos start when the gradient is zero.
        csv_path: when given, ``on_epoch`` rewrites this file after every
            epoch, so a run that dies mid-way leaves its rows on disk.

    Use as ``train(..., iter_callback=probe.on_iter)``, and call
    ``probe.on_epoch(ctx, accuracy)`` from the epoch callback (with the
    accuracy the caller measured, or ``None``). Both return ``None``, so the
    trainer's result lists are unchanged.
    """

    def __init__(
        self,
        structure: GraphStructure,
        probe_clamps: Optional[Mapping[str, Any]] = None,
        *,
        every: int,
        inference: Optional[EPCInference] = None,
        iters: int = 30,
        key: jax.Array,
        csv_path: Optional[Any] = None,
    ):
        if int(every) < 1:
            raise ValueError(f"every must be a positive update count, got {every}")
        self.structure = structure
        self.probe_clamps = None if probe_clamps is None else dict(probe_clamps)
        self.every = int(every)
        self.iters = int(iters)
        self.key = key
        self.csv_path = None if csv_path is None else Path(csv_path)
        if inference is None:
            configured = structure.config.get("inference")
            inference = configured if isinstance(configured, EPCInference) else None
        self.inference = inference
        self.rows: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self._wnorm_columns: Optional[Dict[str, Tuple[str, str]]] = None
        spectrum_fn = make_epsilon_spectrum(structure, self.iters)

        def probe(params, clamps, key):
            batch_size = next(iter(clamps.values())).shape[0]
            state = initialize_graph_state(
                structure, batch_size, key, clamps=clamps, params=params
            )
            norms = {
                node: {
                    k: jnp.sqrt(jnp.sum(w * w)) for k, w in node_params.weights.items()
                }
                for node, node_params in params.nodes.items()
            }
            return spectrum_fn(params, state, clamps, key), norms

        self._probe = jax.jit(probe)

    # ------------------------------------------------------------------ hooks

    def _set_metadata(self, algorithm: str) -> None:
        if self.metadata:
            return
        regime_active = algorithm == "pc" and self.inference is not None
        self.metadata = {
            "trainer": algorithm,
            "eta_infer": (
                float(self.inference.config["eta_infer"]) if regime_active else None
            ),
            "infer_steps": (
                int(self.inference.config["infer_steps"]) if regime_active else None
            ),
            "every": self.every,
            "probe_batch": (
                "train"
                if self.probe_clamps is None
                else int(next(iter(self.probe_clamps.values())).shape[0])
            ),
        }

    def _regime_active(self) -> bool:
        return self.metadata.get("eta_infer") is not None

    def on_iter(self, ctx) -> None:
        """``iter_callback``: probe when ``ctx.step`` is a multiple of ``every``."""
        self._set_metadata(ctx.algorithm)
        if ctx.step % self.every:
            return None
        clamps = (
            self.probe_clamps
            if self.probe_clamps is not None
            else build_clamps(ctx.batch, self.structure, clamp_target=True)
        )
        spectrum, norms = self._probe(ctx.params, clamps, self.key)
        spectrum = spectrum.host()
        if self._wnorm_columns is None:
            self._wnorm_columns = weight_norm_columns(ctx.params)
        row: Dict[str, Any] = {
            "update": int(ctx.step),
            "epoch": int(ctx.epoch_idx),
            "lambda_max": spectrum.lambda_max,
            "lambda_min": spectrum.lambda_min,
            "negative_weight": spectrum.negative_weight,
            "residual_max": spectrum.residual_max,
            "residual_min": spectrum.residual_min,
            "train_energy": float(ctx.metrics["energy"]),
            "test_accuracy": None,
        }
        if self._regime_active():
            regime = self.inference.regime(spectrum)
            row.update(
                f_weighted=regime.f_weighted,
                f_max=regime.f_max,
                growth_min=regime.growth_min,
                eta_lambda_max=regime.eta_lambda_max,
                output_gradient_reverses=regime.output_gradient_reverses,
                unstable=regime.unstable,
            )
        else:
            row.update(
                f_weighted=None,
                f_max=None,
                growth_min=None,
                eta_lambda_max=None,
                output_gradient_reverses=None,
                unstable=None,
            )
        for column, (node, key) in self._wnorm_columns.items():
            row[column] = float(norms[node][key])
        self.rows.append(row)
        return None

    def on_epoch(self, ctx, accuracy: Optional[float] = None) -> None:
        """Record the epoch row (``test_accuracy`` = ``accuracy``) and rewrite
        ``csv_path`` when one was given."""
        self._set_metadata(ctx.algorithm)
        row: Dict[str, Any] = {column: None for column in DATA_COLUMNS}
        row.update(
            update=int(ctx.step),
            epoch=int(ctx.epoch_idx),
            train_energy=(
                float(ctx.metrics["energy"]) if "energy" in ctx.metrics else None
            ),
            test_accuracy=None if accuracy is None else float(accuracy),
        )
        self.rows.append(row)
        if self.csv_path is not None:
            self.write_csv()
        return None

    # --------------------------------------------------------------- readouts

    def probe_rows(self) -> List[Dict[str, Any]]:
        return [row for row in self.rows if row.get("lambda_max") is not None]

    def epoch_rows(self) -> List[Dict[str, Any]]:
        return [row for row in self.rows if row.get("lambda_max") is None]

    def first_reversal(self) -> Optional[Tuple[int, int]]:
        """(update, epoch) of the first probe with ``output_gradient_reverses``."""
        return _first(self.probe_rows(), "output_gradient_reverses")

    def first_crossing(self) -> Optional[Tuple[int, int]]:
        """(update, epoch) of the first probe with ``unstable`` (η·λ_max > 2)."""
        return _first(self.probe_rows(), "unstable")

    def first_chance(self, chance: float, margin: float = 0.05) -> Optional[int]:
        """Epoch (0-based) of the first recorded test accuracy below
        ``chance + margin``; ``chance`` is 1/num_classes, supplied by the
        caller."""
        for row in self.epoch_rows():
            acc = row.get("test_accuracy")
            if acc is not None and acc < chance + margin:
                return int(row["epoch"])
        return None

    def growth_phases(self) -> List[Tuple[int, float, Optional[float]]]:
        """Per epoch (0-based): the largest λ_max probed in it and the ratio to
        the previous epoch's largest (``None`` for the first)."""
        per_epoch: Dict[int, float] = {}
        for row in self.probe_rows():
            epoch = int(row["epoch"])
            per_epoch[epoch] = max(
                per_epoch.get(epoch, float("-inf")), row["lambda_max"]
            )
        phases: List[Tuple[int, float, Optional[float]]] = []
        previous: Optional[float] = None
        for epoch in sorted(per_epoch):
            value = per_epoch[epoch]
            ratio = None if previous in (None, 0.0) else value / previous
            phases.append((epoch, value, ratio))
            previous = value
        return phases

    def summary(self, chance: Optional[float] = None, margin: float = 0.05) -> str:
        """Reversal update, crossing update, chance epoch (when ``chance`` is
        given), and the growth phases, epochs printed 1-indexed."""
        lines = []
        if self._regime_active():
            reversal, crossing = self.first_reversal(), self.first_crossing()
            lines.append(
                "output-gradient reversal (odd T, eta*(lambda_max - 1) > 1 at T = 1): "
                + (
                    f"first flagged at update {reversal[0]} (epoch {reversal[1] + 1})"
                    if reversal
                    else "never flagged"
                )
            )
            lines.append(
                "stability bound eta*lambda_max > 2: "
                + (
                    f"first crossed at update {crossing[0]} (epoch {crossing[1] + 1})"
                    if crossing
                    else "never crossed"
                )
            )
        else:
            lines.append(
                f"trainer {self.metadata.get('trainer')}: spectrum and weight norms "
                f"recorded, no ePC regime"
            )
        if chance is not None:
            epoch = self.first_chance(chance, margin)
            lines.append(
                f"test accuracy within {margin:g} of chance ({chance:g}): "
                + (f"first at epoch {epoch + 1}" if epoch is not None else "never")
            )
        phases = self.growth_phases()
        if phases:
            lines.append(
                "lambda_max per epoch (max over probes; ratio to previous epoch):"
            )
            for epoch, value, ratio in phases:
                ratio_text = "" if ratio is None else f"  x{ratio:.2f}"
                lines.append(f"  epoch {epoch + 1:3d}: {value:12.4g}{ratio_text}")
        return "\n".join(lines)

    # -------------------------------------------------------------------- csv

    def columns(self) -> List[str]:
        return (
            list(METADATA_COLUMNS)
            + list(DATA_COLUMNS)
            + sorted(self._wnorm_columns or {})
        )

    def write_csv(self, path: Optional[Any] = None) -> Path:
        """Write every row (metadata repeated per row); returns the path."""
        target = Path(path) if path is not None else self.csv_path
        if target is None:
            raise ValueError("write_csv needs a path: none given and no csv_path set")
        columns = self.columns()
        with target.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for row in self.rows:
                merged = {**self.metadata, **row}
                writer.writerow([_cell(merged.get(column)) for column in columns])
        return target


def _first(rows: Sequence[Mapping[str, Any]], flag: str) -> Optional[Tuple[int, int]]:
    for row in rows:
        if row.get(flag):
            return int(row["update"]), int(row["epoch"])
    return None


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return repr(value) if isinstance(value, float) else str(value)


def read_regime_csv(path: Any) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read a ``RegimeProbe`` CSV: ``(metadata, rows)`` with the metadata taken
    from the first row's metadata columns and every data cell typed back
    (floats, ints, bools, ``None`` for empty)."""
    with Path(path).open(newline="") as fh:
        raw = list(csv.DictReader(fh))
    if not raw:
        return {}, []
    first = raw[0]
    metadata = {
        "trainer": first["trainer"],
        "eta_infer": _float_or_none(first["eta_infer"]),
        "infer_steps": _int_or_none(first["infer_steps"]),
        "every": _int_or_none(first["every"]),
        "probe_batch": (
            "train"
            if first["probe_batch"] == "train"
            else _int_or_none(first["probe_batch"])
        ),
    }
    rows = []
    for record in raw:
        row: Dict[str, Any] = {}
        for column, text in record.items():
            if column in METADATA_COLUMNS:
                continue
            if column in ("update", "epoch"):
                row[column] = _int_or_none(text)
            elif column in BOOL_COLUMNS:
                row[column] = None if text == "" else text == "True"
            else:
                row[column] = _float_or_none(text)
        rows.append(row)
    return metadata, rows


def _float_or_none(text: str) -> Optional[float]:
    return None if text == "" else float(text)


def _int_or_none(text: str) -> Optional[int]:
    return None if text == "" else int(text)


__all__ = [
    "RegimeProbe",
    "read_regime_csv",
    "weight_norm_columns",
    "METADATA_COLUMNS",
    "DATA_COLUMNS",
    "SPECTRUM_COLUMNS",
    "WNORM_PREFIX",
]
