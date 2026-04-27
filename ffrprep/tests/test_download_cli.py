"""Tests for the ffrprep-download command-line interface."""
import argparse
from pathlib import Path

import pytest

from ffrprep.download_cli import (
    _parse_subjects,
    get_parser,
    run_download,
)


def test_parser_creates_argument_parser():
    parser = get_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def test_parser_requires_subcommand():
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_rejects_unknown_subcommand():
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["bogus"])


def test_example_subcommand_defaults():
    parser = get_parser()
    args = parser.parse_args(["example"])
    assert args.dataset == "example"
    assert args.out is None


def test_example_subcommand_with_out():
    parser = get_parser()
    args = parser.parse_args(["example", "--out", "/tmp/data"])
    assert args.dataset == "example"
    assert args.out == Path("/tmp/data")


def test_raw_subcommand_defaults():
    parser = get_parser()
    args = parser.parse_args(["raw"])
    assert args.dataset == "raw"
    assert args.subjects is None
    assert args.out is None


def test_raw_subcommand_with_count_and_out():
    parser = get_parser()
    args = parser.parse_args(["raw", "--subjects", "3", "--out", "/data"])
    assert args.subjects == ["3"]
    assert args.out == Path("/data")


def test_raw_subcommand_with_subject_ids():
    parser = get_parser()
    args = parser.parse_args(["raw", "--subjects", "03", "21"])
    assert args.subjects == ["03", "21"]


def test_epoch_subcommand_defaults():
    parser = get_parser()
    args = parser.parse_args(["epoch"])
    assert args.dataset == "epoch"
    assert args.subjects is None
    assert args.out is None


def test_epoch_subcommand_with_subject_ids():
    parser = get_parser()
    args = parser.parse_args(["epoch", "--subjects", "01", "02"])
    assert args.subjects == ["01", "02"]


def test_parse_subjects_none_defaults_to_one():
    assert _parse_subjects(None) == 1


def test_parse_subjects_bare_integer_is_count():
    assert _parse_subjects(["3"]) == 3
    assert _parse_subjects(["10"]) == 10


def test_parse_subjects_zero_padded_is_subject_id():
    # BIDS subject IDs are typically zero-padded; treating "03" as a count
    # of 3 would silently misroute the user's request, so it must stay a list.
    assert _parse_subjects(["03"]) == ["03"]


def test_parse_subjects_multiple_values_returns_list():
    assert _parse_subjects(["03", "21"]) == ["03", "21"]


def test_run_download_example_invokes_download_example_data(monkeypatch):
    captured = {}

    def fake_download_example_data(dataset_path=None):
        captured["dataset_path"] = dataset_path
        return Path("/fake/example")

    monkeypatch.setattr(
        "ffrprep.download_cli.download_example_data",
        fake_download_example_data,
    )

    run_download(["example", "--out", "/tmp/example"])
    assert captured["dataset_path"] == Path("/tmp/example")


def test_run_download_example_default_out_is_none(monkeypatch):
    captured = {}

    def fake_download_example_data(dataset_path=None):
        captured["dataset_path"] = dataset_path

    monkeypatch.setattr(
        "ffrprep.download_cli.download_example_data",
        fake_download_example_data,
    )

    run_download(["example"])
    assert captured["dataset_path"] is None


def test_run_download_raw_with_count(monkeypatch):
    captured = {}

    def fake_download_raw_data(subjects=1, dataset_path=None):
        captured["subjects"] = subjects
        captured["dataset_path"] = dataset_path

    monkeypatch.setattr(
        "ffrprep.download_cli.download_raw_data",
        fake_download_raw_data,
    )

    run_download(["raw", "--subjects", "3", "--out", "/data"])
    assert captured["subjects"] == 3
    assert captured["dataset_path"] == Path("/data")


def test_run_download_raw_with_subject_ids(monkeypatch):
    captured = {}

    def fake_download_raw_data(subjects=1, dataset_path=None):
        captured["subjects"] = subjects

    monkeypatch.setattr(
        "ffrprep.download_cli.download_raw_data",
        fake_download_raw_data,
    )

    run_download(["raw", "--subjects", "03", "21"])
    assert captured["subjects"] == ["03", "21"]


def test_run_download_raw_defaults_to_one_subject(monkeypatch):
    captured = {}

    def fake_download_raw_data(subjects=1, dataset_path=None):
        captured["subjects"] = subjects
        captured["dataset_path"] = dataset_path

    monkeypatch.setattr(
        "ffrprep.download_cli.download_raw_data",
        fake_download_raw_data,
    )

    run_download(["raw"])
    assert captured["subjects"] == 1
    assert captured["dataset_path"] is None


def test_run_download_epoch_with_subject_ids(monkeypatch):
    captured = {}

    def fake_download_epoch_data(subjects=1, dataset_path=None):
        captured["subjects"] = subjects
        captured["dataset_path"] = dataset_path

    monkeypatch.setattr(
        "ffrprep.download_cli.download_epoch_data",
        fake_download_epoch_data,
    )

    run_download(["epoch", "--subjects", "01", "02", "--out", "/epochs"])
    assert captured["subjects"] == ["01", "02"]
    assert captured["dataset_path"] == Path("/epochs")


def test_run_download_epoch_defaults(monkeypatch):
    captured = {}

    def fake_download_epoch_data(subjects=1, dataset_path=None):
        captured["subjects"] = subjects
        captured["dataset_path"] = dataset_path

    monkeypatch.setattr(
        "ffrprep.download_cli.download_epoch_data",
        fake_download_epoch_data,
    )

    run_download(["epoch"])
    assert captured["subjects"] == 1
    assert captured["dataset_path"] is None
