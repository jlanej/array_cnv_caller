"""Unit tests for scripts/litmus_test.py."""

from __future__ import annotations

import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

# Ensure the scripts directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from litmus_test import (
    CLASS_DEL,
    CLASS_DUP,
    CLASS_NAMES,
    CLASS_NORMAL,
    build_dashboard,
    build_parser,
    classify_probe,
    collect_probe_data,
    compute_summary_stats,
    label_probes,
    load_truth_intervals,
    match_samples,
)


# ── Fixtures ──────────────────────────────────────────────────────────
@pytest.fixture
def sample_probes():
    """Create a small probe DataFrame for testing."""
    return pd.DataFrame(
        {
            "chrom": ["chr1"] * 10 + ["chr2"] * 5,
            "pos": [
                100, 200, 500, 1500, 2000, 3000, 4000, 5500, 6000, 7000,
                100, 300, 600, 1200, 2500,
            ],
            "lrr": [
                -0.8, -0.9, -0.7, 0.0, 0.1, -0.1, 0.05, 0.5, 0.6, 0.4,
                0.0, 0.1, -0.05, 0.55, 0.0,
            ],
            "baf": [
                0.0, 0.0, 1.0, 0.5, 0.45, 0.55, 0.48, 0.3, 0.7, 0.25,
                0.5, 0.48, 0.52, 0.35, 0.5,
            ],
        }
    )


@pytest.fixture
def truth_bed_file(tmp_path):
    """Create a truth BED file with DEL and DUP regions."""
    content = "chr1\t50\t600\tDEL\nchr1\t5000\t7500\tDUP\nchr2\t500\t1300\tDEL\n"
    bed_path = tmp_path / "test_sample.bed"
    bed_path.write_text(content)
    return str(bed_path)


@pytest.fixture
def multi_sample_truth_dir(tmp_path):
    """Create a truth directory with multiple sample BED files."""
    for name in ("SAMPLE_A", "SAMPLE_B"):
        bed = tmp_path / f"{name}.bed"
        bed.write_text("chr1\t50\t600\tDEL\nchr1\t5000\t7500\tDUP\n")
    return str(tmp_path)


@pytest.fixture
def probe_stats_df():
    """Create a probe-level DataFrame with state labels for stats tests."""
    np.random.seed(42)
    n_del, n_norm, n_dup = 100, 500, 80
    data = {
        "sample": (["S1"] * n_del + ["S1"] * n_norm + ["S1"] * n_dup),
        "chrom": ["chr1"] * (n_del + n_norm + n_dup),
        "pos": list(range(n_del + n_norm + n_dup)),
        "lrr": np.concatenate([
            np.random.normal(-0.7, 0.3, n_del),
            np.random.normal(0.0, 0.15, n_norm),
            np.random.normal(0.4, 0.25, n_dup),
        ]),
        "baf": np.concatenate([
            np.random.uniform(0, 0.15, n_del),
            np.random.normal(0.5, 0.05, n_norm),
            np.random.uniform(0.3, 0.7, n_dup),
        ]),
        "state": (["DEL"] * n_del + ["NORMAL"] * n_norm + ["DUP"] * n_dup),
        "region_size": (
            [5000] * n_del + [0] * n_norm + [3000] * n_dup
        ),
    }
    return pd.DataFrame(data)


# ── load_truth_intervals / classify_probe ─────────────────────────────
class TestLoadTruthIntervals:
    """Tests for load_truth_intervals()."""

    def test_loads_intervals(self, multi_sample_truth_dir):
        ivls = load_truth_intervals(multi_sample_truth_dir, ["SAMPLE_A"])
        assert "SAMPLE_A" in ivls
        # chr1 has DEL and DUP
        assert "chr1" in ivls["SAMPLE_A"]

    def test_sorted_starts(self, multi_sample_truth_dir):
        ivls = load_truth_intervals(multi_sample_truth_dir, ["SAMPLE_A"])
        starts = ivls["SAMPLE_A"]["chr1"][0]
        assert list(starts) == sorted(starts)

    def test_multiple_samples(self, multi_sample_truth_dir):
        ivls = load_truth_intervals(
            multi_sample_truth_dir, ["SAMPLE_A", "SAMPLE_B"]
        )
        assert len(ivls) == 2
        assert "SAMPLE_A" in ivls
        assert "SAMPLE_B" in ivls

    def test_filters_non_del_dup(self, tmp_path):
        bed = tmp_path / "S1.bed"
        bed.write_text("chr1\t100\t200\tDEL\nchr1\t300\t400\tINS\nchr1\t500\t600\tDUP\n")
        ivls = load_truth_intervals(str(tmp_path), ["S1"])
        starts = ivls["S1"]["chr1"][0]
        # Only DEL and DUP should be loaded (INS filtered)
        assert len(starts) == 2

    def test_arrays_have_correct_dtypes(self, multi_sample_truth_dir):
        ivls = load_truth_intervals(multi_sample_truth_dir, ["SAMPLE_A"])
        starts, ends, cls, sizes = ivls["SAMPLE_A"]["chr1"]
        assert starts.dtype == np.int64
        assert ends.dtype == np.int64
        assert cls.dtype == np.int64
        assert sizes.dtype == np.int64


class TestClassifyProbe:
    """Tests for classify_probe()."""

    def test_probe_in_del(self, multi_sample_truth_dir):
        ivls = load_truth_intervals(multi_sample_truth_dir, ["SAMPLE_A"])
        cls, sz = classify_probe(ivls["SAMPLE_A"], "chr1", 200)
        assert cls == CLASS_DEL
        assert sz == 550  # 600 - 50

    def test_probe_in_dup(self, multi_sample_truth_dir):
        ivls = load_truth_intervals(multi_sample_truth_dir, ["SAMPLE_A"])
        cls, sz = classify_probe(ivls["SAMPLE_A"], "chr1", 6000)
        assert cls == CLASS_DUP
        assert sz == 2500

    def test_probe_normal(self, multi_sample_truth_dir):
        ivls = load_truth_intervals(multi_sample_truth_dir, ["SAMPLE_A"])
        cls, sz = classify_probe(ivls["SAMPLE_A"], "chr1", 2000)
        assert cls == CLASS_NORMAL
        assert sz == 0

    def test_probe_at_boundary_start_inclusive(self, multi_sample_truth_dir):
        ivls = load_truth_intervals(multi_sample_truth_dir, ["SAMPLE_A"])
        cls, _ = classify_probe(ivls["SAMPLE_A"], "chr1", 50)
        assert cls == CLASS_DEL

    def test_probe_at_boundary_end_exclusive(self, multi_sample_truth_dir):
        ivls = load_truth_intervals(multi_sample_truth_dir, ["SAMPLE_A"])
        cls, _ = classify_probe(ivls["SAMPLE_A"], "chr1", 600)
        assert cls == CLASS_NORMAL

    def test_unknown_chrom(self, multi_sample_truth_dir):
        ivls = load_truth_intervals(multi_sample_truth_dir, ["SAMPLE_A"])
        cls, sz = classify_probe(ivls["SAMPLE_A"], "chrUn", 100)
        assert cls == CLASS_NORMAL
        assert sz == 0

    def test_empty_intervals(self):
        cls, sz = classify_probe({}, "chr1", 100)
        assert cls == CLASS_NORMAL
        assert sz == 0


# ── label_probes ──────────────────────────────────────────────────────
class TestLabelProbes:
    """Tests for label_probes()."""

    def test_basic_labelling(self, sample_probes, truth_bed_file):
        labels, region_sizes = label_probes(sample_probes, truth_bed_file)
        assert len(labels) == len(sample_probes)
        assert len(region_sizes) == len(sample_probes)

    def test_del_probes_labelled(self, sample_probes, truth_bed_file):
        labels, _ = label_probes(sample_probes, truth_bed_file)
        # Probes at pos 100, 200, 500 on chr1 overlap DEL [50, 600)
        assert labels[0] == CLASS_DEL  # pos 100
        assert labels[1] == CLASS_DEL  # pos 200
        assert labels[2] == CLASS_DEL  # pos 500

    def test_dup_probes_labelled(self, sample_probes, truth_bed_file):
        labels, _ = label_probes(sample_probes, truth_bed_file)
        # Probes at pos 5500, 6000, 7000 on chr1 overlap DUP [5000, 7500)
        assert labels[7] == CLASS_DUP  # pos 5500
        assert labels[8] == CLASS_DUP  # pos 6000
        assert labels[9] == CLASS_DUP  # pos 7000

    def test_normal_probes(self, sample_probes, truth_bed_file):
        labels, _ = label_probes(sample_probes, truth_bed_file)
        # Probes at pos 1500, 2000, 3000, 4000 on chr1 are outside truth
        assert labels[3] == CLASS_NORMAL  # pos 1500
        assert labels[4] == CLASS_NORMAL  # pos 2000
        assert labels[5] == CLASS_NORMAL  # pos 3000
        assert labels[6] == CLASS_NORMAL  # pos 4000

    def test_region_sizes_assigned(self, sample_probes, truth_bed_file):
        _, region_sizes = label_probes(sample_probes, truth_bed_file)
        # DEL region is [50, 600) = 550 bp
        assert region_sizes[0] == 550  # pos 100 in DEL
        # DUP region is [5000, 7500) = 2500 bp
        assert region_sizes[7] == 2500  # pos 5500 in DUP
        # NORMAL probes have region_size 0
        assert region_sizes[3] == 0

    def test_chr2_del(self, sample_probes, truth_bed_file):
        labels, region_sizes = label_probes(sample_probes, truth_bed_file)
        # chr2 DEL [500, 1300): probes at 600 and 1200
        assert labels[12] == CLASS_DEL  # chr2 pos 600
        assert labels[13] == CLASS_DEL  # chr2 pos 1200
        assert region_sizes[12] == 800

    def test_empty_truth_bed(self, sample_probes, tmp_path):
        bed = tmp_path / "empty.bed"
        bed.write_text("")
        # Empty file should label everything NORMAL
        labels, region_sizes = label_probes(sample_probes, str(bed))
        assert all(l == CLASS_NORMAL for l in labels)
        assert all(s == 0 for s in region_sizes)


# ── compute_summary_stats ─────────────────────────────────────────────
class TestComputeSummaryStats:
    """Tests for compute_summary_stats()."""

    def test_returns_dataframe(self, probe_stats_df):
        result = compute_summary_stats(probe_stats_df)
        assert isinstance(result, pd.DataFrame)

    def test_all_states_present(self, probe_stats_df):
        result = compute_summary_stats(probe_stats_df)
        states = set(result["state"])
        assert states == {"DEL", "NORMAL", "DUP"}

    def test_both_metrics(self, probe_stats_df):
        result = compute_summary_stats(probe_stats_df)
        metrics = set(result["metric"])
        assert metrics == {"lrr", "baf"}

    def test_row_count(self, probe_stats_df):
        result = compute_summary_stats(probe_stats_df)
        # 3 states × 2 metrics = 6 rows
        assert len(result) == 6

    def test_del_lrr_mean_negative(self, probe_stats_df):
        result = compute_summary_stats(probe_stats_df)
        del_lrr = result[
            (result["state"] == "DEL") & (result["metric"] == "lrr")
        ]
        assert del_lrr.iloc[0]["mean"] < 0

    def test_dup_lrr_mean_positive(self, probe_stats_df):
        result = compute_summary_stats(probe_stats_df)
        dup_lrr = result[
            (result["state"] == "DUP") & (result["metric"] == "lrr")
        ]
        assert dup_lrr.iloc[0]["mean"] > 0

    def test_expected_columns(self, probe_stats_df):
        result = compute_summary_stats(probe_stats_df)
        expected = {
            "state", "metric", "n", "mean", "median", "std",
            "q1", "q3", "iqr", "min", "max", "skew", "kurtosis",
            "pct_1", "pct_99",
        }
        assert expected == set(result.columns)

    def test_n_counts_correct(self, probe_stats_df):
        result = compute_summary_stats(probe_stats_df)
        del_lrr = result[
            (result["state"] == "DEL") & (result["metric"] == "lrr")
        ]
        assert del_lrr.iloc[0]["n"] == 100

    def test_empty_state_skipped(self):
        df = pd.DataFrame({
            "lrr": [0.0, 0.1],
            "baf": [0.5, 0.5],
            "state": ["NORMAL", "NORMAL"],
        })
        result = compute_summary_stats(df)
        assert set(result["state"]) == {"NORMAL"}


# ── build_dashboard ───────────────────────────────────────────────────
class TestBuildDashboard:
    """Tests for build_dashboard()."""

    def test_html_created(self, probe_stats_df, tmp_path):
        summary = compute_summary_stats(probe_stats_df)
        out = str(tmp_path / "test_report.html")
        build_dashboard(probe_stats_df, summary, out)
        assert os.path.isfile(out)

    def test_html_contains_plotly(self, probe_stats_df, tmp_path):
        summary = compute_summary_stats(probe_stats_df)
        out = str(tmp_path / "test_report.html")
        build_dashboard(probe_stats_df, summary, out)
        with open(out) as f:
            content = f.read()
        assert "plotly" in content.lower()

    def test_html_contains_states(self, probe_stats_df, tmp_path):
        summary = compute_summary_stats(probe_stats_df)
        out = str(tmp_path / "test_report.html")
        build_dashboard(probe_stats_df, summary, out)
        with open(out) as f:
            content = f.read()
        for state in ("DEL", "NORMAL", "DUP"):
            assert state in content

    def test_html_contains_filter_panel(self, probe_stats_df, tmp_path):
        summary = compute_summary_stats(probe_stats_df)
        out = str(tmp_path / "test_report.html")
        build_dashboard(probe_stats_df, summary, out)
        with open(out) as f:
            content = f.read()
        assert "Interactive Filtering" in content
        assert "applyFilters" in content

    def test_html_self_contained(self, probe_stats_df, tmp_path):
        summary = compute_summary_stats(probe_stats_df)
        out = str(tmp_path / "test_report.html")
        build_dashboard(probe_stats_df, summary, out)
        with open(out) as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content


# ── build_parser ──────────────────────────────────────────────────────
class TestBuildParser:
    """Tests for CLI argument parsing."""

    def test_required_args(self):
        parser = build_parser()
        args = parser.parse_args(["--bcf", "test.bcf", "--truth-dir", "beds/"])
        assert args.bcf == "test.bcf"
        assert args.truth_dir == "beds/"

    def test_default_output_dir(self):
        parser = build_parser()
        args = parser.parse_args(["--bcf", "test.bcf", "--truth-dir", "beds/"])
        assert args.output_dir == "litmus_output"

    def test_custom_output_dir(self):
        parser = build_parser()
        args = parser.parse_args([
            "--bcf", "test.bcf",
            "--truth-dir", "beds/",
            "--output-dir", "my_output/",
        ])
        assert args.output_dir == "my_output/"

    def test_max_samples(self):
        parser = build_parser()
        args = parser.parse_args([
            "--bcf", "test.bcf",
            "--truth-dir", "beds/",
            "--max-samples", "10",
        ])
        assert args.max_samples == 10

    def test_missing_required_args(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])
