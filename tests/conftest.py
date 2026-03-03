"""Shared fixtures for array_cnv_caller tests."""

from __future__ import annotations

import os
import textwrap

import pytest

# ── Paths ─────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
RESOURCES_DIR = os.path.join(REPO_ROOT, "resources")

FULL_VCF = os.path.join(
    RESOURCES_DIR, "shapeit5-phased-callset_final-vcf.phased.vcf.gz"
)
TEST_VCF = os.path.join(FIXTURES_DIR, "test_samples.vcf.gz")

# The 10 evaluation samples from Schloissnig, Pani et al.
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


@pytest.fixture
def fixtures_dir():
    """Return the path to the test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def test_vcf_path():
    """Return the path to the subset VCF for the 10 evaluation samples."""
    return TEST_VCF


@pytest.fixture
def full_vcf_path():
    """Return the path to the full shapeit5-phased VCF in resources/."""
    return FULL_VCF


@pytest.fixture
def tiny_vcf(tmp_path):
    """Create a minimal synthetic VCF for fast unit tests.

    Contains 5 variants (3 DEL, 1 DUP, 1 INS) across two samples.
    """
    vcf_content = textwrap.dedent("""\
        ##fileformat=VCFv4.2
        ##FILTER=<ID=PASS,Description="All filters passed">
        ##contig=<ID=chr1>
        ##contig=<ID=chr22>
        ##FORMAT=<ID=GT,Number=1,Type=String,Description="Phased genotypes">
        #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE_A\tSAMPLE_B
        chr1\t1000\tchr1-1000-DEL->s1>s2-5000\tACGTACGT\tA\t.\tPASS\t.\tGT\t0|1\t0|0
        chr1\t50000\tchr1-50000-DEL->s3>s4-2000\tACGT\tA\t.\tPASS\t.\tGT\t1|1\t0|1
        chr1\t100000\tchr1-100000-DUP->s5>s6-3000\tA\tACGTACGT\t.\tPASS\t.\tGT\t0|0\t1|0
        chr1\t200000\tchr1-200000-INS->s7>s8-800\tA\tACGTACGT\t.\tPASS\t.\tGT\t0|1\t1|0
        chr22\t500000\tchr22-500000-DEL->s9>s10-1500\tACGTACGTAC\tA\t.\tPASS\t.\tGT\t0|1\t0|1
    """)
    vcf_path = tmp_path / "tiny.vcf"
    vcf_path.write_text(vcf_content)
    return str(vcf_path)


@pytest.fixture
def tiny_truth_bed(tmp_path):
    """Create a minimal truth BED file matching the tiny_vcf DEL/DUP regions."""
    bed_content = textwrap.dedent("""\
        chr1\t999\t6000\tDEL
        chr1\t49999\t52000\tDEL
        chr22\t499999\t501500\tDEL
    """)
    bed_path = tmp_path / "sample_a.bed"
    bed_path.write_text(bed_content)
    return str(bed_path)
