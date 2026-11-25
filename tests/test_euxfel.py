import numpy as np
import pytest

from QoraFlow.euxfel import (
    accumulate_pulses,
    agipd_asic_seams,
    build_rocking_scan,
    frame_pulse_index,
    match_pulse_offset,
    read_rocking_scan,
    write_rocking_scan,
)

FEL = np.arange(200, 537, 24)  # 15 X-ray pulses, as in the bunch pattern
CELLS = np.arange(0, 1024, 8)  # 128 detector frames per train
OFFSET = -192


def test_seams_are_the_rows_at_asic_joins():
    seams = agipd_asic_seams()
    rows = np.flatnonzero(seams.any(axis=1))
    assert seams.shape == (512, 128)
    assert rows.tolist() == sorted(
        [64 * k for k in range(1, 8)] + [64 * k - 1 for k in range(1, 8)]
    )
    assert seams[rows].all()


def test_offset_found_from_the_frames_with_signal():
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 300, CELLS.size)
    signal[np.isin(CELLS, FEL + OFFSET)] += 1e5

    offset, contrast = match_pulse_offset(CELLS, signal, FEL)

    assert offset == OFFSET
    assert contrast > 100


def test_offset_needs_every_pulse_on_a_frame():
    with pytest.raises(ValueError, match="no offset"):
        match_pulse_offset(np.arange(4), np.ones(4), np.array([0, 10]))


def test_frame_pulse_index_marks_frames_without_pulse():
    index = frame_pulse_index(CELLS[:8], FEL[:3] - 200, 8)
    assert index.tolist() == [-1, 0, -1, -1, 1, -1, -1, 2]


def test_accumulate_keeps_counts_apart_from_values():
    frames = np.array([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])
    valid = np.array([[[True, False]], [[True, False]], [[True, True]]])

    total, count = accumulate_pulses(frames, valid, np.array([0, 0, 1]), 2)

    assert total.tolist() == [[[4.0, 0.0]], [[5.0, 6.0]]]
    assert count.tolist() == [[[2, 0]], [[1, 1]]]


def fake_run(shape=(4, 6)):
    """Two scan steps; frame value = 10 * pulse index + step, one masked and one NaN pixel."""
    trains = {0: np.array([100, 101, 102]), 1: np.array([103, 104, 105])}

    def read_step(train_ids, _detector_pulse_ids):
        step = 0 if train_ids[0] < 103 else 1
        frame_trains = np.repeat(train_ids, CELLS.size)
        frame_pulses = np.tile(CELLS, train_ids.size)
        index = frame_pulse_index(frame_pulses, FEL, OFFSET)
        data = np.broadcast_to(
            np.where(index >= 0, 10.0 * index + step, -1.0)[:, None, None],
            (frame_pulses.size, *shape),
        ).copy()
        mask = np.zeros(data.shape, np.uint32)
        mask[:, 0, 0] = 4  # always masked: never valid
        mask[frame_trains == train_ids[0], 1, 1] = 1  # masked in the first train only
        data[:, 2, 2] = np.nan
        return data, mask, frame_trains, frame_pulses

    steps = [(18.9, 0.01, trains[0]), (19.0, 0.02, trains[1])]
    xgm = np.arange(6 * FEL.size, dtype=float).reshape(6, FEL.size)
    return steps, read_step, xgm


def test_build_rocking_scan_per_step_and_pulse():
    steps, read_step, xgm = fake_run()
    seams = np.zeros((4, 6), bool)
    seams[3] = True

    scan = build_rocking_scan(
        steps,
        read_step,
        FEL,
        OFFSET,
        xgm_train_ids=np.arange(100, 106),
        xgm=xgm,
        seams=seams,
        attrs={"run": 299},
    )

    assert scan.sum.shape == (2, FEL.size, 4, 6)
    assert scan.count.dtype == np.uint16
    mean = scan.mean
    assert mean[1, 4, 1, 2] == pytest.approx(41.0)
    assert scan.count[0, 0, 1, 2] == 3
    assert scan.count[0, 0, 1, 1] == 2
    assert np.isnan(mean[:, :, 0, 0]).all() and (scan.count[:, :, 0, 0] == 0).all()
    assert np.isnan(mean[:, :, 2, 2]).all()
    assert (scan.count[:, :, 3] == 0).all()
    assert scan.calibration_bits[0, 0] == 4 and scan.calibration_bits[1, 1] == 1
    np.testing.assert_array_equal(scan.n_trains, [3, 3])
    np.testing.assert_allclose(scan.xgm_sum[1], xgm[3:].sum(axis=0))
    np.testing.assert_array_equal(scan.train_step, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(scan.detector_pulse_id, FEL + OFFSET)
    assert scan.pumped is None
    assert scan.attrs["pulse_offset"] == OFFSET and scan.attrs["run"] == 299


def test_roi_and_round_trip(tmp_path):
    steps, read_step, xgm = fake_run()
    pumped = np.arange(FEL.size) < 14

    scan = build_rocking_scan(
        steps,
        read_step,
        FEL,
        OFFSET,
        xgm_train_ids=np.arange(100, 106),
        xgm=xgm,
        roi=(1, 3, 1, 4),
        seams=np.zeros((4, 6), bool),
        pumped=pumped,
        attrs={"run": 299, "delay_stage": 812.6},
    )
    write_rocking_scan(tmp_path / "r0299_m15.h5", scan)
    back = read_rocking_scan(tmp_path / "r0299_m15.h5")

    assert back.sum.shape == (2, FEL.size, 2, 3)
    np.testing.assert_array_equal(back.sum, scan.sum)
    np.testing.assert_array_equal(back.count, scan.count)
    np.testing.assert_array_equal(back.pumped, pumped)
    np.testing.assert_array_equal(back.train_xgm, scan.train_xgm)
    assert back.attrs["delay_stage"] == 812.6
    assert list(back.attrs["roi"]) == [1, 3, 1, 4]
