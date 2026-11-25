"""Rocking scans from European XFEL AGIPD data, with every X-ray pulse kept apart.

A Bragg rocking scan at an XFEL records many trains per rocking-angle step and several X-ray
pulses per train. In a pump-probe run some of those pulses follow a laser pulse and some do not,
so summing over pulses mixes pumped and unpumped diffraction. This module groups corrected
("proc") AGIPD frames by scan step and by X-ray pulse, and keeps for every pixel both the sum
of the valid values and the number of frames that contributed: a pixel that was never valid
has a count of 0 and no value, rather than an intensity of 0.

Pulse identification. The AGIPD ``pulseId`` of a frame is the detector's own label, not the
bunch-pattern position of the X-ray pulse: the two differ by a constant offset, which is found
from the data by :func:`match_pulse_offset` (the offset that puts the X-ray pattern on the
frames carrying signal). Frames that match no X-ray pulse are dropped.

Pixels are valid where the calibration mask is 0, the value is finite and the pixel is not on
an ASIC seam (:func:`agipd_asic_seams`). The X-ray gas monitor (XGM) value of each pulse is
summed over the same trains and stored, not applied. Which pulses were pumped is stored only
when it is given; it is not derived from the laser bunch pattern, whose position relative to
the X-ray pattern is a timing convention to be checked against the data.

The pure functions work on numpy arrays. :func:`extract_rocking_run` reads a run with
``extra_data`` and ``extra`` (European XFEL software, imported only there).
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

AGIPD_MODULE_SHAPE = (512, 128)

Roi = tuple[int, int, int, int]
StepReader = Callable[[npt.NDArray, npt.NDArray], tuple[npt.NDArray, ...]]


def agipd_asic_seams() -> npt.NDArray[np.bool_]:
    """Double-size pixels at the joins between ASICs of one AGIPD module, as in EXtra-geom."""
    seams = np.zeros(AGIPD_MODULE_SHAPE, dtype=bool)
    seams[64::64] = True
    seams[63:511:64] = True
    return seams


def match_pulse_offset(
    frame_pulse_ids: npt.ArrayLike,
    frame_signal: npt.ArrayLike,
    fel_pulse_ids: npt.ArrayLike,
    min_contrast: float = 5.0,
) -> tuple[int, float]:
    """Offset ``d`` such that the frames with ``pulseId == fel + d`` carry the X-ray signal.

    Every offset that places all X-ray pulses on recorded frames is tried; the one whose matched
    frames have the largest mean signal over the remaining frames wins. Returns the offset and
    that contrast (matched mean minus unmatched mean, in units of the unmatched spread). A best
    contrast below ``min_contrast`` raises ``ValueError``: on frames without diffraction any
    offset fits about as well as any other.
    """
    frame_pulse_ids = np.asarray(frame_pulse_ids)
    frame_signal = np.asarray(frame_signal, dtype=float)
    fel_pulse_ids = np.asarray(fel_pulse_ids)
    best: tuple[int, float] | None = None
    for offset in np.unique(frame_pulse_ids - fel_pulse_ids[0]):
        matched = np.isin(frame_pulse_ids, fel_pulse_ids + offset)
        if matched.sum() != fel_pulse_ids.size or matched.all():
            continue
        rest = frame_signal[~matched]
        contrast = (frame_signal[matched].mean() - rest.mean()) / (rest.std() or 1.0)
        if best is None or contrast > best[1]:
            best = (int(offset), float(contrast))
    if best is None:
        raise ValueError("no offset puts every X-ray pulse on a recorded frame")
    if best[1] < min_contrast:
        raise ValueError(
            f"X-ray pulses not found in the frames: best offset {best[0]} has contrast "
            f"{best[1]:.2f} < {min_contrast}"
        )
    return best


def frame_pulse_index(
    frame_pulse_ids: npt.ArrayLike, fel_pulse_ids: npt.ArrayLike, offset: int
) -> npt.NDArray[np.intp]:
    """Index into ``fel_pulse_ids`` of each frame, -1 for frames without an X-ray pulse."""
    targets = np.asarray(fel_pulse_ids) + offset
    frame_pulse_ids = np.asarray(frame_pulse_ids)
    index = np.searchsorted(targets, frame_pulse_ids)
    index = np.clip(index, 0, targets.size - 1)
    return np.where(targets[index] == frame_pulse_ids, index, -1)


def valid_pixels(
    data: npt.NDArray, mask: npt.NDArray, seams: npt.NDArray[np.bool_] | None = None
) -> npt.NDArray[np.bool_]:
    """Pixels with a zero calibration mask, a finite value and, given seams, off the seams."""
    valid = (mask == 0) & np.isfinite(data)
    return valid & ~seams if seams is not None else valid


def accumulate_pulses(
    frames: npt.NDArray, valid: npt.NDArray[np.bool_], pulse_index: npt.NDArray, n_pulses: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Per pulse, the sum of valid values and the number of valid frames at each pixel."""
    total = np.zeros((n_pulses, *frames.shape[1:]))
    count = np.zeros((n_pulses, *frames.shape[1:]), dtype=np.int64)
    for pulse in range(n_pulses):
        take = pulse_index == pulse
        total[pulse] = np.where(valid[take], frames[take], 0.0).sum(axis=0)
        count[pulse] = valid[take].sum(axis=0)
    return total, count


def _crop(arr: npt.NDArray, roi: Roi | None) -> npt.NDArray:
    if roi is None:
        return arr
    y0, y1, x0, x1 = roi
    return arr[..., y0:y1, x0:x1]


@dataclass
class RockingScan:
    """Per scan step and X-ray pulse: summed intensity, valid-frame counts and XGM.

    ``sum`` and ``count`` have axes (step, pulse, y, x); ``xgm_sum`` (step, pulse) sums the XGM
    pulse energy over the trains that contributed frames, ``n_trains`` counts those trains.
    ``theta`` and ``theta_std`` are the mean and spread of the scan motor within each step.
    ``train_*`` keep the per-train record: train id, step index and XGM per pulse.
    """

    sum: npt.NDArray[np.float32]
    count: npt.NDArray[np.uint16]
    xgm_sum: npt.NDArray[np.float64]
    n_trains: npt.NDArray[np.int64]
    theta: npt.NDArray[np.float64]
    theta_std: npt.NDArray[np.float64]
    fel_pulse_id: npt.NDArray[np.int64]
    detector_pulse_id: npt.NDArray[np.int64]
    train_id: npt.NDArray[np.uint64]
    train_step: npt.NDArray[np.int64]
    train_xgm: npt.NDArray[np.float32]
    seams: npt.NDArray[np.bool_]
    calibration_bits: npt.NDArray[np.uint32]
    pumped: npt.NDArray[np.bool_] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def mean(self) -> npt.NDArray[np.float64]:
        """Mean over valid frames, NaN where no frame was valid."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.count > 0, self.sum / self.count, np.nan)


def build_rocking_scan(
    steps: Sequence[tuple[float, float, npt.ArrayLike]],
    read_step: StepReader,
    fel_pulse_ids: npt.ArrayLike,
    offset: int,
    *,
    xgm_train_ids: npt.ArrayLike | None = None,
    xgm: npt.ArrayLike | None = None,
    roi: Roi | None = None,
    seams: npt.NDArray[np.bool_] | None = None,
    pumped: npt.ArrayLike | None = None,
    attrs: dict[str, Any] | None = None,
) -> RockingScan:
    """Accumulate a rocking scan step by step.

    ``steps`` holds ``(theta_mean, theta_std, train_ids)`` per step. ``read_step(train_ids,
    detector_pulse_ids)`` returns ``(data, mask, frame_train_ids, frame_pulse_ids)`` for those
    trains, frames first, with at least the requested pulses. ``xgm`` is (train, pulse) for the
    trains in ``xgm_train_ids``, its columns in the order of ``fel_pulse_ids``.
    """
    fel_pulse_ids = np.asarray(fel_pulse_ids, dtype=np.int64)
    if np.any(np.diff(fel_pulse_ids) <= 0):
        raise ValueError("fel_pulse_ids must be strictly increasing")
    n_pulses = fel_pulse_ids.size
    detector_ids = fel_pulse_ids + offset
    seams = agipd_asic_seams() if seams is None else seams
    xgm_lookup: dict[int, npt.NDArray] = {}
    if xgm is not None and xgm_train_ids is not None:
        xgm_rows = np.asarray(xgm, dtype=np.float32)[:, :n_pulses]
        xgm_lookup = dict(zip(np.asarray(xgm_train_ids).tolist(), xgm_rows, strict=True))

    sums, counts, xgm_sums, n_trains = [], [], [], []
    train_records: list[tuple[int, int]] = []
    calibration_bits = None
    for step, (_, _, train_ids) in enumerate(steps):
        data, mask, frame_trains, frame_pulses = read_step(np.asarray(train_ids), detector_ids)
        pulse_index = frame_pulse_index(frame_pulses, fel_pulse_ids, offset)
        keep = pulse_index >= 0
        data, mask = data[keep], mask[keep]
        valid = _crop(valid_pixels(data, mask, seams), roi)
        total, count = accumulate_pulses(_crop(data, roi), valid, pulse_index[keep], n_pulses)
        if count.max(initial=0) > np.iinfo(np.uint16).max:
            raise OverflowError(f"step {step}: more than 65535 frames in one pixel count")
        bits = np.bitwise_or.reduce(_crop(mask, roi).astype(np.uint32), axis=0)
        calibration_bits = bits if calibration_bits is None else calibration_bits | bits

        used = np.unique(np.asarray(frame_trains)[keep])
        train_records += [(int(t), step) for t in used]
        step_xgm = [xgm_lookup[t] for t in used.tolist() if t in xgm_lookup]
        xgm_sums.append(np.sum(step_xgm, axis=0) if step_xgm else np.full(n_pulses, np.nan))
        sums.append(total.astype(np.float32))
        counts.append(count.astype(np.uint16))
        n_trains.append(used.size)

    train_id = np.array([t for t, _ in train_records], dtype=np.uint64)
    nan_row = np.full(n_pulses, np.nan, dtype=np.float32)
    return RockingScan(
        sum=np.stack(sums),
        count=np.stack(counts),
        xgm_sum=np.array(xgm_sums, dtype=np.float64),
        n_trains=np.array(n_trains, dtype=np.int64),
        theta=np.array([s[0] for s in steps], dtype=np.float64),
        theta_std=np.array([s[1] for s in steps], dtype=np.float64),
        fel_pulse_id=fel_pulse_ids,
        detector_pulse_id=detector_ids,
        train_id=train_id,
        train_step=np.array([s for _, s in train_records], dtype=np.int64),
        train_xgm=np.array(
            [xgm_lookup.get(int(t), nan_row) for t in train_id], dtype=np.float32
        ).reshape(-1, n_pulses),
        seams=_crop(seams, roi),
        calibration_bits=calibration_bits,
        pumped=None if pumped is None else np.asarray(pumped, dtype=bool),
        attrs={"pulse_offset": offset, "roi": list(roi) if roi else [], **(attrs or {})},
    )


def write_rocking_scan(path: Path | str, scan: RockingScan) -> None:
    """Write a :class:`RockingScan` to HDF5 (``/data``, ``/axes``, ``/trains``, ``/mask``)."""
    with h5py.File(path, "w") as f:
        f["data/sum"] = scan.sum
        f["data/count"] = scan.count
        f["data/xgm_sum"] = scan.xgm_sum
        f["data/n_trains"] = scan.n_trains
        f["axes/theta"] = scan.theta
        f["axes/theta_std"] = scan.theta_std
        f["axes/pulse/fel_pulse_id"] = scan.fel_pulse_id
        f["axes/pulse/detector_pulse_id"] = scan.detector_pulse_id
        if scan.pumped is not None:
            f["axes/pulse/pumped"] = scan.pumped
        f["trains/train_id"] = scan.train_id
        f["trains/step"] = scan.train_step
        f["trains/xgm"] = scan.train_xgm
        f["mask/asic_seams"] = scan.seams
        f["mask/calibration_bits"] = scan.calibration_bits
        f["data/sum"].attrs["axes"] = "step, pulse, y, x"
        for key, value in scan.attrs.items():
            f.attrs[key] = value


def read_rocking_scan(path: Path | str) -> RockingScan:
    """Read a file written by :func:`write_rocking_scan`."""
    with h5py.File(path, "r") as f:
        pumped = f["axes/pulse/pumped"][()] if "axes/pulse/pumped" in f else None
        return RockingScan(
            sum=f["data/sum"][()],
            count=f["data/count"][()],
            xgm_sum=f["data/xgm_sum"][()],
            n_trains=f["data/n_trains"][()],
            theta=f["axes/theta"][()],
            theta_std=f["axes/theta_std"][()],
            fel_pulse_id=f["axes/pulse/fel_pulse_id"][()],
            detector_pulse_id=f["axes/pulse/detector_pulse_id"][()],
            train_id=f["trains/train_id"][()],
            train_step=f["trains/step"][()],
            train_xgm=f["trains/xgm"][()],
            seams=f["mask/asic_seams"][()],
            calibration_bits=f["mask/calibration_bits"][()],
            pumped=pumped,
            attrs={k: v.item() if isinstance(v, np.generic) else v for k, v in f.attrs.items()},
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_rocking_run(
    proposal_dir: Path | str,
    run: int,
    module: int,
    *,
    detector: str = "MID_DET_AGIPD1M-1",
    theta: tuple[str, str] = ("MID_EXP_SAM/MDL/DATA_SELECTOR", "midExpSamMotorSsry.actualPosition"),
    delay: tuple[str, str] = ("MID_LAS_COM/CTRL/PPL_PHASE_SHIFTER", "actualPosition"),
    xgm: tuple[str, str] = ("SA2_XTD1_XGM/XGM/DOOCS:output", "data.intensityTD"),
    roi: Roi | None = None,
    steps: slice | None = None,
    trains_per_step: int | None = None,
    trains_per_read: int = 4,
    pumped: npt.ArrayLike | None = None,
) -> RockingScan:
    """Read one AGIPD module of one run from ``raw`` (control data only) and ``proc``.

    Only the raw files matching ``*DA*`` are opened, so detector raw data is never touched.
    ``steps`` selects scan steps and ``trains_per_step`` keeps the first trains of each, for a
    short test. Theta per step is the mean and spread of the motor readback over its trains.
    Default source names are those of the MID instrument; pass others for another instrument.
    """
    import extra_data
    from extra.components import OpticalLaserPulses, Scan, XrayPulses
    from extra_data import RunDirectory, by_id

    proposal_dir = Path(proposal_dir)
    raw = RunDirectory(proposal_dir / "raw" / f"r{run:04d}", include="*DA*")
    proc_dir = proposal_dir / "proc" / f"r{run:04d}"
    proc = RunDirectory(proc_dir, include=f"*AGIPD{module:02d}*")
    source = f"{detector}/CORR/{module}CH0:output"

    fel = XrayPulses(raw).get_pulse_ids()
    patterns = {tuple(g.values) for _, g in fel.groupby(level=0)}
    if len(patterns) != 1:
        raise ValueError(f"run {run}: the X-ray pulse pattern changes between trains")
    fel_pulse_ids = np.array(patterns.pop(), dtype=np.int64)
    laser_ids = OpticalLaserPulses(raw).get_pulse_ids()
    laser_first = laser_ids.loc[laser_ids.index.get_level_values(0)[0]].values

    theta_key = raw[theta]
    theta_values = dict(
        zip(theta_key.train_id_coordinates().tolist(), theta_key.ndarray(), strict=True)
    )
    scan_steps = []
    for _, train_ids in Scan(theta_key).steps:
        train_ids = np.intersect1d(np.asarray(train_ids), proc.train_ids)[:trains_per_step]
        if train_ids.size:
            angles = np.array([theta_values[t] for t in train_ids.tolist()])
            scan_steps.append((float(angles.mean()), float(angles.std()), train_ids))
    # The pulse offset is found on one train from each of up to 16 steps across the whole scan:
    # steps at the edge of the rocking curve carry almost no signal.
    frame_pulses, frame_signal = None, 0.0
    for _, _, train_ids in scan_steps[:: max(1, len(scan_steps) // 16)]:
        one = proc.select_trains(by_id[train_ids[:1]])
        pulses = one[source, "image.pulseId"].ndarray().ravel()
        if frame_pulses is not None and not np.array_equal(pulses, frame_pulses):
            raise ValueError(f"run {run}: detector pulse ids change between trains")
        frame_pulses = pulses
        data = np.nan_to_num(one[source, "image.data"].ndarray())
        frame_signal = frame_signal + np.where(
            one[source, "image.mask"].ndarray() == 0, data, 0.0
        ).sum(axis=(1, 2))
    offset, contrast = match_pulse_offset(frame_pulses, frame_signal, fel_pulse_ids)

    if steps is not None:
        scan_steps = scan_steps[steps]

    def read_step(train_ids, detector_pulse_ids):
        parts = []
        for part in proc.select_trains(by_id[train_ids]).split_trains(
            trains_per_part=trains_per_read
        ):
            pulses = part[source, "image.pulseId"].ndarray().ravel()
            keep = np.isin(pulses, detector_pulse_ids)
            parts.append(
                (
                    part[source, "image.data"].ndarray()[keep],
                    part[source, "image.mask"].ndarray()[keep],
                    part[source, "image.trainId"].ndarray().ravel()[keep],
                    pulses[keep],
                )
            )
        return tuple(np.concatenate(column) for column in zip(*parts, strict=True))

    xgm_key = raw[xgm]
    delay_values = raw[delay].ndarray()
    calibration = proc_dir / f"calibration_metadata_{detector}.yml"
    attrs = {
        "proposal_dir": str(proposal_dir),
        "run": run,
        "module": module,
        "detector_source": source,
        "theta_source": "/".join(theta),
        "xgm_source": "/".join(xgm),
        "delay_source": "/".join(delay),
        "delay_stage": float(np.mean(delay_values)),
        "delay_stage_std": float(np.std(delay_values)),
        "laser_pulse_ids_first_train": np.asarray(laser_first, dtype=np.int64),
        "pulse_offset_contrast": contrast,
        "extra_data_version": extra_data.__version__,
        "calibration_metadata_sha256": _sha256(calibration) if calibration.is_file() else "",
    }
    return build_rocking_scan(
        scan_steps,
        read_step,
        fel_pulse_ids,
        offset,
        xgm_train_ids=xgm_key.train_id_coordinates(),
        xgm=xgm_key.ndarray(),
        roi=roi,
        pumped=pumped,
        attrs=attrs,
    )
