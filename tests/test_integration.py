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

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from prepare_truth_set import parse_sv_id, process_vcf
from litmus_test import collect_probe_data, compute_summary_stats, build_dashboard

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
    """

    @pytest.fixture
    def array_vcf(self):
        """Return path to the stage2 array VCF with FORMAT/LRR and FORMAT/BAF."""
        if not os.path.exists(ARRAY_VCF):
            pytest.skip("Array VCF not found at " + ARRAY_VCF)
        return ARRAY_VCF

    @pytest.fixture
    def truth_dir(self, array_vcf, tmp_path):
        """Generate truth BED files from the shapeit5-phased VCF for the 8 array samples."""
        truth_output = str(tmp_path / "truth_sets")
        process_vcf(TEST_VCF, truth_output, min_size=1000)
        return os.path.join(truth_output, "per_sample")

    def test_litmus_runs_to_completion(self, array_vcf, truth_dir, tmp_path):
        """The litmus pipeline should run end-to-end without errors."""
        df = collect_probe_data(array_vcf, truth_dir)
        assert len(df) > 0, "No probe data collected"

        summary = compute_summary_stats(df)
        assert len(summary) > 0, "Summary statistics are empty"

        html_path = str(tmp_path / "litmus_report.html")
        build_dashboard(df, summary, html_path)
        assert os.path.isfile(html_path), "HTML report not created"

    def test_litmus_produces_probe_data(self, array_vcf, truth_dir):
        """Probe data should contain all expected columns."""
        df = collect_probe_data(array_vcf, truth_dir)
        expected_cols = {"sample", "chrom", "pos", "lrr", "baf", "state", "region_size"}
        assert expected_cols == set(df.columns)

    def test_litmus_samples_match(self, array_vcf, truth_dir):
        """Collected probes should come from the matched array samples."""
        df = collect_probe_data(array_vcf, truth_dir)
        collected_samples = set(df["sample"].unique())
        # All collected samples must be among the expected array samples
        assert collected_samples.issubset(set(ARRAY_SAMPLES))
        assert len(collected_samples) > 0

    def test_litmus_has_all_states(self, array_vcf, truth_dir):
        """Probe data should include DEL, NORMAL, and DUP states."""
        df = collect_probe_data(array_vcf, truth_dir)
        states = set(df["state"].unique())
        assert "NORMAL" in states, "No NORMAL probes found"
        # At least one CNV state should appear
        assert states & {"DEL", "DUP"}, "No CNV probes (DEL/DUP) found"

    def test_litmus_summary_stats_complete(self, array_vcf, truth_dir):
        """Summary statistics should have rows for each state/metric pair."""
        df = collect_probe_data(array_vcf, truth_dir)
        summary = compute_summary_stats(df)
        # Should have entries for both lrr and baf
        assert set(summary["metric"]) == {"lrr", "baf"}
        # At minimum NORMAL should be present
        assert "NORMAL" in set(summary["state"])

    def test_litmus_html_is_informative(self, array_vcf, truth_dir, tmp_path):
        """The HTML dashboard should contain all key sections."""
        df = collect_probe_data(array_vcf, truth_dir)
        summary = compute_summary_stats(df)
        html_path = str(tmp_path / "litmus_report.html")
        build_dashboard(df, summary, html_path)

        with open(html_path) as f:
            content = f.read()

        # Structural checks
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content
        assert "plotly" in content.lower()

        # Key sections present
        assert "Summary Statistics" in content
        assert "LRR" in content
        assert "BAF" in content
        assert "Interactive Filtering" in content
        assert "applyFilters" in content

        # CN states appear in the report
        for state in ("DEL", "NORMAL", "DUP"):
            assert state in content

    def test_litmus_html_has_sample_info(self, array_vcf, truth_dir, tmp_path):
        """The HTML dashboard should reference at least some array sample names."""
        df = collect_probe_data(array_vcf, truth_dir)
        summary = compute_summary_stats(df)
        html_path = str(tmp_path / "litmus_report.html")
        build_dashboard(df, summary, html_path)

        with open(html_path) as f:
            content = f.read()

        # At least some sample names should appear in the filter dropdown
        collected_samples = df["sample"].unique()
        for sample in collected_samples:
            assert sample in content, f"Sample {sample} not found in HTML"
