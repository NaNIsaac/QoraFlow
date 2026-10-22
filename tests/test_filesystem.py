from QoraFlow.filesystem import (
    get_run_nums,
    get_run_scan_dir,
    get_scan_nums,
    make_run_scan_dir,
)


def test_get_run_scan_dir_combinations(tmp_path):
    assert get_run_scan_dir(tmp_path, 1) == tmp_path / "run=001"
    assert get_run_scan_dir(tmp_path, "2", 3) == tmp_path / "run=002" / "scan=003"
    assert (
        get_run_scan_dir(tmp_path, 1, 2, sub_path="/data/x.h5")
        == tmp_path / "run=001" / "scan=002" / "data" / "x.h5"
    )


def test_make_run_scan_dir_creates_only_run_and_scan(tmp_path):
    path = make_run_scan_dir(tmp_path, 4, 12, sub_path="out")
    assert path == tmp_path / "run=004" / "scan=012" / "out"
    assert path.parent.is_dir()
    assert not path.exists()


def test_run_and_scan_numbers_round_trip(tmp_path):
    for run, scan in [(1, 1), (1, 2), (7, 5)]:
        make_run_scan_dir(tmp_path, run, scan)
    assert sorted(get_run_nums(tmp_path)) == [1, 7]
    assert sorted(get_scan_nums(tmp_path, 1)) == [1, 2]
