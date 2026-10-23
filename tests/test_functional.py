import pytest

from QoraFlow.functional import batched, compose, identity, pipe


def test_identity():
    assert identity(3) == 3
    assert identity(1, 2) == (1, 2)


def test_compose_runs_right_to_left():
    assert compose(lambda x: x * 2, lambda x: x + 1)(3) == 8


def test_pipe_runs_left_to_right():
    assert pipe(lambda x: x * 2, lambda x: x + 1)(3) == 7


def test_batched_keeps_short_last_batch():
    assert list(batched("ABCDEFG", 3)) == [("A", "B", "C"), ("D", "E", "F"), ("G",)]


def test_batched_empty():
    assert list(batched([], 2)) == []


def test_batched_rejects_zero():
    with pytest.raises(ValueError):
        list(batched("abc", 0))
