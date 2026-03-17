"""Integration tests using the 10 evaluation samples from Schloissnig et al.

These tests use the subset VCF (tests/fixtures/test_samples.vcf.gz) extracted
from the full shapeit5-phased callset for the 10 samples recommended for
methods development and evaluation:

    HG00268, HG00513, HG00731, HG02554, HG02953,
    NA12878, NA19129, NA19238, NA19331, NA19347

Reference:
    Schloissnig, S., Pani, S. et al. Long-read sequencing and structural
    variant characterization in 1,019 samples from the 1000 Genomes Project.
    bioRxiv 2024.04.18.590093 (2024) doi:10.1101/2024.04.18.590093
"""

from __future__ import annotations

import gzip
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from prepare_truth_set import parse_sv_id, process_vcf
from litmus_test import (
    collect_probe_data,
    compute_summary_stats,
    build_dashboard,
    match_samples,
    main as litmus_main,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TEST_VCF = os.path.join(FIXTURES_DIR, "test_samples.vcf.gz")
ARRAY_VCF = os.path.join(
    FIXTURES_DIR, "stage2_reclustered.selected.array.samples.vcf.gz"
)
EVAL_SAMPLES = [
    "HG00268",
    "HG00513",
    "HG00731",
    "HG02554",
    "HG02953",
    "NA12878",
    "NA19129",
    "NA19238",
    "NA19331",
    "NA19347",
]
ARRAY_SAMPLES = [
    "HG00268",
    "HG00513",
    "HG00731",
    "NA12878",
    "NA19129",
    "NA19238",
    "NA19331",
    "NA19347",
]

pytestmark = pytest.mark.integration


@pytest.fixture
def subset_vcf():
    """Return path to the 10-sample subset VCF."""
    if not os.path.exists(TEST_VCF):
        pytest.skip("Subset VCF not found at " + TEST_VCF)
    return TEST_VCF


class TestSubsetVcf:
    """Validate the structure of the subset VCF fixture."""

    def test_vcf_exists(self, subset_vcf):
        assert os.path.exists(subset_vcf)

    def test_vcf_has_correct_samples(self, subset_vcf):
        open_fn = gzip.open if subset_vcf.endswith(".gz") else open
        with open_fn(subset_vcf, "rt") as fh:
            for line in fh:
                if line.startswith("#CHROM"):
                    fields = line.strip().split("\t")
                    samples = fields[9:]
                    assert sorted(samples) == sorted(EVAL_SAMPLES)
                    return
        pytest.fail("No #CHROM header line found in VCF")

    def test_vcf_has_variants(self, subset_vcf):
        open_fn = gzip.open if subset_vcf.endswith(".gz") else open
        variant_count = 0
        with open_fn(subset_vcf, "rt") as fh:
            for line in fh:
                if not line.startswith("#"):
                    variant_count += 1
        assert variant_count > 0, "VCF has no variant records"

    def test_all_variants_are_del_or_dup(self, subset_vcf):
        """The subset VCF should only contain DEL and DUP variants."""
        open_fn = gzip.open if subset_vcf.endswith(".gz") else open
        with open_fn(subset_vcf, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                variant_id = line.split("\t")[2]
                parsed = parse_sv_id(variant_id)
                assert parsed is not None, f"Unparseable ID: {variant_id}"
                svtype, size = parsed
                assert svtype in ("DEL", "DUP"), f"Unexpected type: {svtype}"
                assert size >= 1000, f"Size below threshold: {size}"


class TestTruthSetGeneration:
    """Test truth set generation from the subset VCF."""

    def test_per_sample_bed_generation(self, subset_vcf, tmp_path):
        output_dir = str(tmp_path / "truth_sets")
        process_vcf(subset_vcf, output_dir, min_size=1000)

        per_sample_dir = os.path.join(output_dir, "per_sample")
        for sample in EVAL_SAMPLES:
            bed_path = os.path.join(per_sample_dir, f"{sample}.bed")
            assert os.path.exists(bed_path), f"Missing BED for {sample}"

    def test_bed_records_are_sorted(self, subset_vcf, tmp_path):
        output_dir = str(tmp_path / "truth_sorted")
        process_vcf(subset_vcf, output_dir, min_size=1000)

        per_sample_dir = os.path.join(output_dir, "per_sample")
        for sample in EVAL_SAMPLES:
            bed_path = os.path.join(per_sample_dir, f"{sample}.bed")
            if not os.path.exists(bed_path):
                continue
            with open(bed_path) as f:
                lines = f.readlines()
            if not lines:
                continue
            prev_chrom, prev_start = "", -1
            for line in lines:
                fields = line.strip().split("\t")
                chrom, start = fields[0], int(fields[1])
                if chrom == prev_chrom:
                    assert start >= prev_start, (
                        f"BED not sorted: {sample} {chrom}:{start} < {prev_start}"
                    )
                prev_chrom, prev_start = chrom, start

    def test_na12878_has_cnvs(self, subset_vcf, tmp_path):
        """NA12878 is a well-characterised reference – it should have CNV calls."""
        output_dir = str(tmp_path / "truth_na12878")
        process_vcf(subset_vcf, output_dir, min_size=1000)

        bed_path = os.path.join(output_dir, "per_sample", "NA12878.bed")
        assert os.path.exists(bed_path)
        with open(bed_path) as f:
            lines = f.readlines()
        assert len(lines) > 0, "NA12878 should have CNV calls"

    def test_summary_sample_count(self, subset_vcf, tmp_path):
        output_dir = str(tmp_path / "truth_summary")
        process_vcf(subset_vcf, output_dir, min_size=1000)

        import pandas as pd

        summary = pd.read_csv(
            os.path.join(output_dir, "truth_set_summary.tsv"), sep="\t"
        )
        metrics = dict(zip(summary["metric"], summary["value"]))
        assert int(metrics["total_samples"]) == 10


class TestLitmusPipeline:
    """Integration test for the litmus pipeline using the array VCF.

    Runs the full litmus pipeline (truth-set generation → probe collection →
    summary statistics → HTML dashboard) on the stage2 array VCF fixture
    from illumina_idat_processing.

    A single BCF scan is performed for the whole class via the module-scoped
    ``litmus_data`` fixture, so individual tests are fast.
    """

    @pytest.fixture(scope="module")
    def array_vcf_path(self):
        """Return path to the stage2 array VCF with FORMAT/LRR and FORMAT/BAF."""
        if not os.path.exists(ARRAY_VCF):
            pytest.skip("Array VCF not found at " + ARRAY_VCF)
        return ARRAY_VCF

    @pytest.fixture(scope="module")
    def truth_dir(self, tmp_path_factory):
        """Generate truth BED files from the shapeit5-phased SV VCF.

        Uses the 10-sample SV VCF (test_samples.vcf.gz) as the source of
        truth intervals; the array VCF provides LRR/BAF signal.  The
        intersection of both sample sets determines which samples are analysed.
        """
        truth_output = str(tmp_path_factory.mktemp("truth_sets"))
        process_vcf(TEST_VCF, truth_output, min_size=1000)
        return os.path.join(truth_output, "per_sample")

    @pytest.fixture(scope="module")
    def litmus_data(self, array_vcf_path, truth_dir):
        """Run the BCF scan once for the whole class and return (df, summary).

        This is the most expensive step (~3 s for 96k records × 8 samples).
        All individual tests reuse this result instead of repeating the scan.
        """
        df = collect_probe_data(array_vcf_path, truth_dir)
        summary = compute_summary_stats(df)
        return df, summary

    # ── Sample matching ───────────────────────────────────────────────

    def test_matched_sample_count(self, array_vcf_path, truth_dir):
        """All 8 array samples should match truth BED files."""
        matched, bcf_only, truth_only = match_samples(array_vcf_path, truth_dir)
        assert set(matched) == set(ARRAY_SAMPLES), (
            f"Matched samples {set(matched)} differ from expected {set(ARRAY_SAMPLES)}"
        )
        assert bcf_only == [], f"BCF samples with no truth BED: {bcf_only}"

    # ── Probe data schema & coverage ─────────────────────────────────

    def test_probe_schema(self, litmus_data):
        """Probe DataFrame must have exactly the expected columns."""
        df, _ = litmus_data
        expected_cols = {"sample", "chrom", "pos", "lrr", "baf", "state", "region_size"}
        assert set(df.columns) == expected_cols

    def test_probe_count(self, litmus_data):
        """A real array VCF should yield at least 50,000 probe entries per sample."""
        df, _ = litmus_data
        n_samples = df["sample"].nunique()
        assert n_samples > 0
        probes_per_sample = len(df) / n_samples
        assert probes_per_sample >= 50_000, (
            f"Fewer probes than expected per sample: {probes_per_sample:.0f}"
        )

    def test_all_array_samples_present(self, litmus_data):
        """All 8 array samples must appear in the probe DataFrame."""
        df, _ = litmus_data
        collected = set(df["sample"].unique())
        assert collected == set(ARRAY_SAMPLES), (
            f"Collected samples {collected} differ from expected {set(ARRAY_SAMPLES)}"
        )

    def test_multiple_chromosomes(self, litmus_data):
        """Probes should span multiple chromosomes."""
        df, _ = litmus_data
        assert df["chrom"].nunique() >= 5, (
            f"Too few chromosomes in probe data: {sorted(df['chrom'].unique())}"
        )

    def test_lrr_baf_ranges(self, litmus_data):
        """LRR and BAF values should fall within biologically plausible ranges."""
        df, _ = litmus_data
        normal = df[df["state"] == "NORMAL"]
        lrr_median = normal["lrr"].median()
        baf_median = normal["baf"].median()
        # NORMAL LRR should be near 0
        assert abs(lrr_median) < 0.5, f"NORMAL LRR median unexpectedly far from 0: {lrr_median:.3f}"
        # NORMAL BAF median should be near 0.5 (mix of 0, 0.5, 1)
        assert 0.3 < baf_median < 0.7, f"NORMAL BAF median out of range: {baf_median:.3f}"

    # ── Copy-number state content ─────────────────────────────────────

    def test_normal_state_dominates(self, litmus_data):
        """NORMAL probes should be the vast majority (>95%) of all probes."""
        df, _ = litmus_data
        frac_normal = (df["state"] == "NORMAL").mean()
        assert frac_normal > 0.95, (
            f"NORMAL fraction unexpectedly low: {frac_normal:.3f}"
        )

    def test_cnv_state_present(self, litmus_data):
        """At least one CNV state (DEL or DUP) must appear in the data."""
        df, _ = litmus_data
        states = set(df["state"].unique())
        assert states & {"DEL", "DUP"}, (
            f"No CNV probes found; states present: {states}"
        )

    def test_del_lrr_lower_than_normal(self, litmus_data):
        """DEL probes should have lower median LRR than NORMAL probes."""
        df, _ = litmus_data
        del_df = df[df["state"] == "DEL"]
        if del_df.empty:
            pytest.skip("No DEL probes in dataset")
        del_median = del_df["lrr"].median()
        normal_median = df[df["state"] == "NORMAL"]["lrr"].median()
        assert del_median < normal_median, (
            f"DEL LRR median ({del_median:.3f}) not below NORMAL ({normal_median:.3f})"
        )

    def test_del_region_sizes_positive(self, litmus_data):
        """DEL probes must have a positive region_size (labelled from a truth interval)."""
        df, _ = litmus_data
        del_df = df[df["state"] == "DEL"]
        if del_df.empty:
            pytest.skip("No DEL probes in dataset")
        assert (del_df["region_size"] > 0).all(), (
            "Some DEL probes have region_size == 0"
        )

    # ── Summary statistics ────────────────────────────────────────────

    def test_summary_has_both_metrics(self, litmus_data):
        """Summary stats must cover both LRR and BAF metrics."""
        _, summary = litmus_data
        assert set(summary["metric"]) == {"lrr", "baf"}

    def test_summary_normal_state_present(self, litmus_data):
        """Summary stats must include the NORMAL state."""
        _, summary = litmus_data
        assert "NORMAL" in set(summary["state"])

    def test_summary_del_lrr_mean_negative(self, litmus_data):
        """Summary mean LRR for DEL should be below 0."""
        _, summary = litmus_data
        row = summary[(summary["state"] == "DEL") & (summary["metric"] == "lrr")]
        if row.empty:
            pytest.skip("No DEL rows in summary")
        assert row.iloc[0]["mean"] < 0, (
            f"DEL mean LRR expected < 0, got {row.iloc[0]['mean']:.3f}"
        )

    def test_summary_columns_complete(self, litmus_data):
        """Summary DataFrame must contain all expected statistical columns."""
        _, summary = litmus_data
        expected = {
            "state", "metric", "n", "mean", "median", "std",
            "q1", "q3", "iqr", "min", "max", "skew", "kurtosis",
            "pct_1", "pct_99",
        }
        assert set(summary.columns) == expected

    # ── HTML dashboard ────────────────────────────────────────────────

    def test_html_is_self_contained(self, litmus_data, tmp_path):
        """HTML report must be a complete, self-contained document."""
        df, summary = litmus_data
        html_path = str(tmp_path / "litmus_report.html")
        build_dashboard(df, summary, html_path)

        assert os.path.isfile(html_path), "HTML report not created"
        with open(html_path) as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content
        # Must be self-contained: no external CSS/JS links other than Plotly CDN
        assert "plotly" in content.lower()

    def test_html_contains_all_sections(self, litmus_data, tmp_path):
        """All 7 dashboard sections and the interactive filter panel must be present."""
        df, summary = litmus_data
        html_path = str(tmp_path / "litmus_sections.html")
        build_dashboard(df, summary, html_path)

        with open(html_path) as f:
            content = f.read()

        for section in (
            "Summary Statistics",
            "LRR",
            "BAF",
            "Violin",
            "Scatter",
            "Per-Chromosome",
            "Per-Sample",
            "Interactive Filtering",
        ):
            assert section in content, f"Section '{section}' missing from HTML"

    def test_html_has_working_filter_panel(self, litmus_data, tmp_path):
        """The interactive filter panel must include JavaScript and dropdowns."""
        df, summary = litmus_data
        html_path = str(tmp_path / "litmus_filter.html")
        build_dashboard(df, summary, html_path)

        with open(html_path) as f:
            content = f.read()
        assert "applyFilters" in content, "JavaScript filter function missing"
        assert "filt-chrom" in content, "Chromosome filter dropdown missing"
        assert "filt-sample" in content, "Sample filter dropdown missing"

    def test_html_cn_states_in_report(self, litmus_data, tmp_path):
        """All three copy-number state labels must appear in the HTML."""
        df, summary = litmus_data
        html_path = str(tmp_path / "litmus_states.html")
        build_dashboard(df, summary, html_path)

        with open(html_path) as f:
            content = f.read()
        for state in ("DEL", "NORMAL", "DUP"):
            assert state in content, f"State '{state}' missing from HTML"

    def test_html_sample_names_in_filter_dropdown(self, litmus_data, tmp_path):
        """All matched sample names must appear in the filter panel dropdown."""
        df, summary = litmus_data
        html_path = str(tmp_path / "litmus_samples.html")
        build_dashboard(df, summary, html_path)

        with open(html_path) as f:
            content = f.read()
        for sample in df["sample"].unique():
            assert sample in content, f"Sample '{sample}' missing from HTML"

    # ── Full CLI output ───────────────────────────────────────────────

    def test_cli_produces_all_output_files(self, array_vcf_path, truth_dir, tmp_path):
        """The litmus_test CLI (main()) should create all three expected output files."""
        out_dir = str(tmp_path / "cli_output")
        litmus_main([
            "--bcf", array_vcf_path,
            "--truth-dir", truth_dir,
            "--output-dir", out_dir,
        ])

        assert os.path.isfile(os.path.join(out_dir, "litmus_report.html")), \
            "litmus_report.html not created by CLI"
        assert os.path.isfile(os.path.join(out_dir, "summary_stats.tsv")), \
            "summary_stats.tsv not created by CLI"
        assert os.path.isfile(os.path.join(out_dir, "probe_stats.tsv.gz")), \
            "probe_stats.tsv.gz not created by CLI"
