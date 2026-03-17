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
    BLACKLIST_NONE,
    CLASS_DEL,
    CLASS_DUP,
    CLASS_NAMES,
    CLASS_NORMAL,
    annotate_probes_with_blacklist,
    build_blacklist_summary,
    build_dashboard,
    build_parser,
    classify_probe,
    classify_probe_blacklist,
    collect_probe_data,
    compute_summary_stats,
    label_probes,
    load_blacklist_regions,
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
        assert "Interactive Filtering Panel" in content
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

    def test_blacklist_dir_default(self):
        """--blacklist-dir defaults to the repo resources/blacklists/ directory."""
        parser = build_parser()
        args = parser.parse_args(["--bcf", "test.bcf", "--truth-dir", "beds/"])
        assert "blacklists" in args.blacklist_dir

    def test_blacklist_dir_custom(self):
        parser = build_parser()
        args = parser.parse_args([
            "--bcf", "test.bcf",
            "--truth-dir", "beds/",
            "--blacklist-dir", "/tmp/my_bl/",
        ])
        assert args.blacklist_dir == "/tmp/my_bl/"

    def test_blacklist_dir_empty_disables(self):
        """Passing --blacklist-dir '' allows disabling blacklist annotation."""
        parser = build_parser()
        args = parser.parse_args([
            "--bcf", "test.bcf",
            "--truth-dir", "beds/",
            "--blacklist-dir", "",
        ])
        assert args.blacklist_dir == ""


# ── load_blacklist_regions ────────────────────────────────────────────
class TestLoadBlacklistRegions:
    """Tests for load_blacklist_regions()."""

    @pytest.fixture
    def blacklist_dir(self, tmp_path):
        """Create a minimal blacklist directory with two BED files."""
        (tmp_path / "centromeres.bed").write_text(
            "# comment\n"
            "chr1\t100\t200\tcentromere\n"
            "chr2\t500\t1000\tcentromere\n"
        )
        (tmp_path / "telomeres.bed").write_text(
            "chr1\t0\t50\ttelomere\n"
            "chr1\t9950\t10000\ttelomere\n"
        )
        return str(tmp_path)

    def test_loads_both_tracks(self, blacklist_dir):
        bl = load_blacklist_regions(blacklist_dir)
        assert "centromeres" in bl
        assert "telomeres" in bl

    def test_skips_non_bed_files(self, tmp_path):
        (tmp_path / "SOURCES.md").write_text("# sources")
        (tmp_path / "regions.bed").write_text("chr1\t0\t100\tfoo\n")
        bl = load_blacklist_regions(str(tmp_path))
        assert "regions" in bl
        assert "SOURCES" not in bl

    def test_skips_comment_lines(self, blacklist_dir):
        bl = load_blacklist_regions(blacklist_dir)
        starts = bl["centromeres"]["chr1"][0]
        assert len(starts) == 1  # only one chr1 entry (comment skipped)

    def test_sorted_starts(self, blacklist_dir):
        bl = load_blacklist_regions(blacklist_dir)
        starts = bl["telomeres"]["chr1"][0]
        assert list(starts) == sorted(starts)

    def test_empty_dir(self, tmp_path):
        bl = load_blacklist_regions(str(tmp_path))
        assert bl == {}

    def test_missing_dir(self):
        bl = load_blacklist_regions("/nonexistent/path/")
        assert bl == {}

    def test_arrays_dtype(self, blacklist_dir):
        bl = load_blacklist_regions(blacklist_dir)
        starts, ends, idxs = bl["centromeres"]["chr1"]
        assert starts.dtype == np.int64
        assert ends.dtype == np.int64
        assert idxs.dtype == np.int64


# ── classify_probe_blacklist ──────────────────────────────────────────
class TestClassifyProbeBlacklist:
    """Tests for classify_probe_blacklist()."""

    @pytest.fixture
    def blacklist(self, tmp_path):
        (tmp_path / "centromeres.bed").write_text("chr1\t100\t200\tcentromere\n")
        (tmp_path / "telomeres.bed").write_text("chr1\t0\t50\ttelomere\n")
        return load_blacklist_regions(str(tmp_path))

    def test_probe_in_centromere(self, blacklist):
        assert classify_probe_blacklist(blacklist, "chr1", 150) == "centromeres"

    def test_probe_in_telomere(self, blacklist):
        assert classify_probe_blacklist(blacklist, "chr1", 10) == "telomeres"

    def test_probe_outside_all(self, blacklist):
        assert classify_probe_blacklist(blacklist, "chr1", 300) == BLACKLIST_NONE

    def test_unknown_chrom(self, blacklist):
        assert classify_probe_blacklist(blacklist, "chrZ", 100) == BLACKLIST_NONE

    def test_empty_blacklist(self):
        assert classify_probe_blacklist({}, "chr1", 100) == BLACKLIST_NONE

    def test_boundary_start_inclusive(self, blacklist):
        # centromere starts at 100
        assert classify_probe_blacklist(blacklist, "chr1", 100) == "centromeres"

    def test_boundary_end_exclusive(self, blacklist):
        # centromere ends at 200 (exclusive)
        assert classify_probe_blacklist(blacklist, "chr1", 200) == BLACKLIST_NONE


# ── annotate_probes_with_blacklist ────────────────────────────────────
class TestAnnotateProbesWithBlacklist:
    """Tests for annotate_probes_with_blacklist()."""

    @pytest.fixture
    def simple_df(self):
        return pd.DataFrame({
            "chrom": ["chr1", "chr1", "chr1"],
            "pos":   [10, 150, 300],
            "lrr":   [0.0, -0.5, 0.1],
            "baf":   [0.5, 0.1, 0.5],
        })

    @pytest.fixture
    def bl_dir(self, tmp_path):
        (tmp_path / "centromeres.bed").write_text("chr1\t100\t200\tcentromere\n")
        (tmp_path / "telomeres.bed").write_text("chr1\t0\t50\ttelomere\n")
        return str(tmp_path)

    def test_adds_blacklist_region_column(self, simple_df, bl_dir):
        result = annotate_probes_with_blacklist(simple_df, bl_dir)
        assert "blacklist_region" in result.columns

    def test_correct_labels(self, simple_df, bl_dir):
        result = annotate_probes_with_blacklist(simple_df, bl_dir)
        assert result.iloc[0]["blacklist_region"] == "telomeres"
        assert result.iloc[1]["blacklist_region"] == "centromeres"
        assert result.iloc[2]["blacklist_region"] == BLACKLIST_NONE

    def test_empty_blacklist_dir(self, simple_df, tmp_path):
        result = annotate_probes_with_blacklist(simple_df, str(tmp_path))
        assert all(v == BLACKLIST_NONE for v in result["blacklist_region"])

    def test_missing_dir(self, simple_df, tmp_path):
        result = annotate_probes_with_blacklist(simple_df, "/nonexistent/")
        assert all(v == BLACKLIST_NONE for v in result["blacklist_region"])


# ── build_blacklist_summary ───────────────────────────────────────────
class TestBuildBlacklistSummary:
    """Tests for build_blacklist_summary()."""

    @pytest.fixture
    def df_with_bl(self):
        return pd.DataFrame({
            "state": ["DEL"] * 30 + ["NORMAL"] * 60 + ["DUP"] * 10,
            "blacklist_region": (
                ["centromeres"] * 10 + [""] * 20
                + [""] * 60
                + ["telomeres"] * 10
            ),
        })

    def test_returns_dataframe(self, df_with_bl):
        result = build_blacklist_summary(df_with_bl)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, df_with_bl):
        result = build_blacklist_summary(df_with_bl)
        assert set(result.columns) == {"blacklist_region", "state", "n_probes", "pct_of_state"}

    def test_none_region_labelled(self, df_with_bl):
        result = build_blacklist_summary(df_with_bl)
        regions = set(result["blacklist_region"])
        assert "(none)" in regions

    def test_centromere_region_present(self, df_with_bl):
        result = build_blacklist_summary(df_with_bl)
        assert "centromeres" in set(result["blacklist_region"])

    def test_pct_calculation(self, df_with_bl):
        result = build_blacklist_summary(df_with_bl)
        cen_del = result[
            (result["blacklist_region"] == "centromeres") & (result["state"] == "DEL")
        ]
        assert not cen_del.empty
        # 10 centromere DEL out of 30 total DEL = 33.33%
        assert abs(cen_del.iloc[0]["pct_of_state"] - 33.33) < 0.5

    def test_no_blacklist_column(self):
        df = pd.DataFrame({"state": ["DEL", "NORMAL"], "lrr": [0.0, 0.1]})
        result = build_blacklist_summary(df)
        assert result.empty

    def test_counts_correct(self, df_with_bl):
        result = build_blacklist_summary(df_with_bl)
        cen_del = result[
            (result["blacklist_region"] == "centromeres") & (result["state"] == "DEL")
        ]
        assert cen_del.iloc[0]["n_probes"] == 10


# ── build_dashboard (blacklist integration) ───────────────────────────
class TestBuildDashboardBlacklist:
    """Tests for build_dashboard() when blacklist_region column is present."""

    @pytest.fixture
    def probe_stats_with_bl(self, probe_stats_df):
        np.random.seed(7)
        n = len(probe_stats_df)
        # Assign ~15% of probes to blacklist regions to exercise the summary table
        regions = np.where(
            np.random.rand(n) < 0.15,
            np.random.choice(["centromeres", "telomeres"], size=n),
            "",
        )
        probe_stats_df = probe_stats_df.copy()
        probe_stats_df["blacklist_region"] = regions
        return probe_stats_df

    def test_html_contains_blacklist_section(self, probe_stats_with_bl, tmp_path):
        summary = compute_summary_stats(probe_stats_with_bl)
        out = str(tmp_path / "test_bl.html")
        build_dashboard(
            probe_stats_with_bl, summary, out,
            blacklist_dir="/tmp/mock_bl/"  # dir absent; section still shown
        )
        with open(out) as f:
            content = f.read()
        assert "Blacklist Region Summary" in content

    def test_html_contains_t2t_reference(self, probe_stats_with_bl, tmp_path):
        summary = compute_summary_stats(probe_stats_with_bl)
        out = str(tmp_path / "test_t2t.html")
        build_dashboard(probe_stats_with_bl, summary, out)
        with open(out) as f:
            content = f.read()
        assert "T2T" in content

    def test_html_blacklist_filter_option(self, probe_stats_with_bl, tmp_path):
        summary = compute_summary_stats(probe_stats_with_bl)
        out = str(tmp_path / "test_bl_filter.html")
        build_dashboard(probe_stats_with_bl, summary, out)
        with open(out) as f:
            content = f.read()
        assert "Exclude blacklist" in content
        assert "Show only blacklist" in content

    def test_html_no_blacklist_column(self, probe_stats_df, tmp_path):
        """Dashboard should render without error even without blacklist_region col."""
        summary = compute_summary_stats(probe_stats_df)
        out = str(tmp_path / "test_no_bl.html")
        build_dashboard(probe_stats_df, summary, out)
        assert os.path.isfile(out)

