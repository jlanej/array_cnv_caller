"""Unit tests for scripts/prepare_truth_set.py."""

from __future__ import annotations

import os
import sys
import textwrap

import pytest

# Ensure the scripts directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from prepare_truth_set import parse_sv_id, compute_sv_end, process_vcf


# ── parse_sv_id ───────────────────────────────────────────────────────
class TestParseSvId:
    """Tests for parse_sv_id()."""

    def test_del_simple(self):
        result = parse_sv_id("chr1-154808-DEL->s5339>s5341-113")
        assert result == ("DEL", 113)

    def test_dup(self):
        result = parse_sv_id("chr3-45000000-DUP->s12345>s12347-2500")
        assert result == ("DUP", 2500)

    def test_ins(self):
        result = parse_sv_id("chr5-100000-INS->s99>s100-800")
        assert result == ("INS", 800)

    def test_complex(self):
        result = parse_sv_id("chr7-200000-COMPLEX->s200>s201-300")
        assert result == ("COMPLEX", 300)

    def test_inv(self):
        result = parse_sv_id("chr1-297459-INV->s16481<s34323>s16592-1748")
        assert result == ("INV", 1748)

    def test_large_size(self):
        result = parse_sv_id("chrX-1000000-DEL->s1>s2-999999")
        assert result == ("DEL", 999999)

    def test_unparseable_returns_none(self):
        assert parse_sv_id("not-a-valid-id") is None

    def test_empty_string(self):
        assert parse_sv_id("") is None

    def test_missing_size(self):
        assert parse_sv_id("chr1-1000-DEL") is None


# ── compute_sv_end ────────────────────────────────────────────────────
class TestComputeSvEnd:
    """Tests for compute_sv_end()."""

    def test_del_end(self):
        # DEL: end = pos + len(ref) - 1
        end = compute_sv_end(pos=1000, ref_len=5001, alt_len=1, svtype="DEL", id_size=5000)
        assert end == 1000 + 5001 - 1

    def test_dup_end(self):
        # DUP: end = pos + id_size
        end = compute_sv_end(pos=50000, ref_len=1, alt_len=3001, svtype="DUP", id_size=3000)
        assert end == 50000 + 3000


# ── process_vcf (integration with tiny VCF) ──────────────────────────
class TestProcessVcf:
    """Tests for process_vcf() using a small synthetic VCF."""

    def test_process_tiny_vcf(self, tiny_vcf, tmp_path):
        output_dir = str(tmp_path / "truth_output")
        process_vcf(tiny_vcf, output_dir, min_size=1000)

        per_sample_dir = os.path.join(output_dir, "per_sample")
        assert os.path.isdir(per_sample_dir)

        # Check that SAMPLE_A and SAMPLE_B bed files were created
        sample_a_bed = os.path.join(per_sample_dir, "SAMPLE_A.bed")
        sample_b_bed = os.path.join(per_sample_dir, "SAMPLE_B.bed")

        assert os.path.exists(sample_a_bed)
        assert os.path.exists(sample_b_bed)

        # SAMPLE_A has: chr1 DEL 5000bp (kept), chr1 DEL 2000bp (kept),
        #               chr22 DEL 1500bp (kept)
        # INS 800bp is excluded by type and size
        with open(sample_a_bed) as f:
            lines = f.readlines()
        assert len(lines) == 3  # 3 DEL records >= 1000bp

        # Check that summary TSV was created
        summary_path = os.path.join(output_dir, "truth_set_summary.tsv")
        assert os.path.exists(summary_path)

    def test_min_size_filter(self, tiny_vcf, tmp_path):
        """With min_size=3000, only the 5000bp and 3000bp SVs should pass."""
        output_dir = str(tmp_path / "truth_large")
        process_vcf(tiny_vcf, output_dir, min_size=3000)

        per_sample_dir = os.path.join(output_dir, "per_sample")
        sample_a_bed = os.path.join(per_sample_dir, "SAMPLE_A.bed")

        with open(sample_a_bed) as f:
            lines = f.readlines()
        # Only chr1 DEL 5000bp should remain for SAMPLE_A
        assert len(lines) == 1

    def test_summary_tsv_contents(self, tiny_vcf, tmp_path):
        output_dir = str(tmp_path / "truth_summary")
        process_vcf(tiny_vcf, output_dir, min_size=1000)

        summary_path = os.path.join(output_dir, "truth_set_summary.tsv")
        with open(summary_path) as f:
            content = f.read()

        assert "total_variants" in content
        assert "total_samples" in content
        assert "type_kept_DEL" in content


# ── Integration test with real subset VCF ─────────────────────────────
@pytest.mark.integration
class TestProcessVcfIntegration:
    """Integration tests using the 10-sample subset VCF from resources."""

    def test_process_test_samples_vcf(self, test_vcf_path, tmp_path):
        """Process the subset VCF and verify per-sample BED output."""
        if not os.path.exists(test_vcf_path):
            pytest.skip("Test subset VCF not found")

        output_dir = str(tmp_path / "integration_truth")
        process_vcf(test_vcf_path, output_dir, min_size=1000)

        per_sample_dir = os.path.join(output_dir, "per_sample")
        assert os.path.isdir(per_sample_dir)

        # Every evaluation sample should have a BED file
        eval_samples = [
            "HG00268", "HG00513", "HG00731", "HG02554", "HG02953",
            "NA12878", "NA19129", "NA19238", "NA19331", "NA19347",
        ]

        for sample in eval_samples:
            bed_path = os.path.join(per_sample_dir, f"{sample}.bed")
            assert os.path.exists(bed_path), f"Missing BED for {sample}"

            with open(bed_path) as f:
                lines = f.readlines()
            assert len(lines) > 0, f"Empty BED for {sample}"

            # Verify BED format: 4 columns per line
            for line in lines:
                fields = line.strip().split("\t")
                assert len(fields) == 4, f"Bad BED format: {line}"
                chrom, start, end, svtype = fields
                assert chrom.startswith("chr")
                assert int(start) >= 0
                assert int(end) > int(start)
                assert svtype in ("DEL", "DUP")

    def test_summary_metrics(self, test_vcf_path, tmp_path):
        """Verify summary metrics are reasonable."""
        if not os.path.exists(test_vcf_path):
            pytest.skip("Test subset VCF not found")

        output_dir = str(tmp_path / "integration_metrics")
        process_vcf(test_vcf_path, output_dir, min_size=1000)

        summary_path = os.path.join(output_dir, "truth_set_summary.tsv")
        assert os.path.exists(summary_path)

        import pandas as pd

        summary = pd.read_csv(summary_path, sep="\t")
        metrics = dict(zip(summary["metric"], summary["value"]))

        assert int(metrics["total_samples"]) == 10
        assert int(metrics["samples_with_calls"]) > 0
        assert int(metrics["type_kept_DEL"]) > 0
        assert int(metrics.get("type_kept_DUP", 0)) >= 0
