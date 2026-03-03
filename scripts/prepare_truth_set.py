#!/usr/bin/env python3
"""
prepare_truth_set.py – Parse the 1000 Genomes ONT Vienna shapeit5-phased SV VCF
and produce per-sample BED truth files for training the ML CNV caller.

The phased callset is sequence-resolved (no INFO/SVTYPE field).  SV type is
derived from the variant ID, which follows the pattern::

    chr1-118734679-DEL->s65362>s65364-4638
    chr3-45000000-DUP->s12345>s12347-2500
    chr5-100000-INS->s99>s100-800
    chr7-200000-COMPLEX->s200>s201-300

We keep only **DEL** and **DUP** variants ≥ 1 kb and write per-sample BED
files with columns: ``chrom  start  end  svtype``.

Usage::

    python scripts/prepare_truth_set.py \\
        --vcf resources/shapeit5-phased-callset_final-vcf.phased.vcf.gz \\
        --output-dir truth_sets \\
        --min-size 1000

Requirements: pysam, pandas (already in requirements_ml.txt)
"""

from __future__ import annotations

import argparse
import collections
import gzip
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)

# Regex to parse the variant ID.
# Examples:
#   chr1-154808-DEL->s5339>s5341-113
#   chr1-297459-INS->s16481<s34323>s16592-1748
#   chr1-21837-COMPLEX->s65863<s38486>s1192-104
# Captures: svtype  and  trailing size (last integer after -)
_ID_RE = re.compile(
    r"^[^-]+-\d+-(?P<svtype>[A-Z]+)->.+-(?P<size>\d+)$"
)

# SV types we keep for CNV training
_KEEP_TYPES = {"DEL", "DUP"}

# Minimum SV size (bp) to include
_DEFAULT_MIN_SIZE = 1000


def parse_sv_id(variant_id: str) -> Optional[Tuple[str, int]]:
    """Extract SV type and size from the variant ID string.

    Returns
    -------
    (svtype, size) or None if the ID doesn't match the expected pattern.
    """
    m = _ID_RE.match(variant_id)
    if m is None:
        return None
    return m.group("svtype"), int(m.group("size"))


def compute_sv_end(pos: int, ref_len: int, alt_len: int, svtype: str, id_size: int) -> int:
    """Compute the end coordinate for a structural variant.

    For DELs the end is POS + len(REF) - 1 (the deleted region on the reference).
    For DUPs, we use POS + id_size as the duplicated span.
    """
    if svtype == "DEL":
        return pos + ref_len - 1
    else:
        # DUP – use the size from the ID
        return pos + id_size


def process_vcf(
    vcf_path: str,
    output_dir: str,
    min_size: int = _DEFAULT_MIN_SIZE,
) -> None:
    """Parse the phased SV VCF and write per-sample BED files + summary."""

    per_sample_dir = os.path.join(output_dir, "per_sample")
    os.makedirs(per_sample_dir, exist_ok=True)

    # ── Counters for summary ──────────────────────────────────────────────
    type_total: Dict[str, int] = collections.Counter()       # all SV types seen
    type_kept: Dict[str, int] = collections.Counter()        # DEL/DUP ≥ min_size
    type_excluded_size: Dict[str, int] = collections.Counter()  # DEL/DUP < min_size
    type_excluded_type: Dict[str, int] = collections.Counter()  # other SV types
    unparseable_ids = 0
    total_variants = 0

    # Size distributions for kept variants
    kept_sizes: Dict[str, list] = {"DEL": [], "DUP": []}

    # Per-sample record counts
    sample_record_counts: Dict[str, Dict[str, int]] = {}

    # Per-sample BED records: sample -> list of (chrom, start, end, svtype)
    sample_beds: Dict[str, list] = {}

    # ── Read VCF ──────────────────────────────────────────────────────────
    LOG.info("Parsing VCF: %s", vcf_path)

    # Use gzip to read the VCF line-by-line for maximum control
    open_fn = gzip.open if vcf_path.endswith(".gz") else open
    samples: List[str] = []

    with open_fn(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                samples = fields[9:]
                LOG.info("Found %d samples in VCF", len(samples))
                for s in samples:
                    sample_beds[s] = []
                    sample_record_counts[s] = collections.Counter()
                continue

            total_variants += 1
            fields = line.rstrip("\n").split("\t")
            chrom = fields[0]
            pos = int(fields[1])
            variant_id = fields[2]
            ref = fields[3]
            alt = fields[4]

            # ── Parse SV type from ID ─────────────────────────────────
            parsed = parse_sv_id(variant_id)
            if parsed is None:
                unparseable_ids += 1
                continue

            svtype, id_size = parsed
            type_total[svtype] += 1

            if svtype not in _KEEP_TYPES:
                type_excluded_type[svtype] += 1
                continue

            if id_size < min_size:
                type_excluded_size[svtype] += 1
                continue

            type_kept[svtype] += 1
            kept_sizes[svtype].append(id_size)

            # ── Compute genomic coordinates ───────────────────────────
            sv_start = pos  # 1-based start (VCF POS)
            sv_end = compute_sv_end(pos, len(ref), len(alt), svtype, id_size)

            # Convert to 0-based half-open BED coordinates
            bed_start = sv_start - 1
            bed_end = sv_end

            # ── Extract per-sample genotypes ──────────────────────────
            gt_fields = fields[9:]
            for i, gt_str in enumerate(gt_fields):
                # Genotypes are phased: "0|0", "0|1", "1|0", "1|1"
                # In this VCF they appear as "00", "01", "10", "11"
                # (pipe separator omitted in the shapeit5 output)
                gt = gt_str.strip()

                # Determine if the sample carries an alt allele
                if gt in ("00", "0|0", "0/0", ".", "./.", ".|."):
                    continue  # homozygous reference or missing

                sample_name = samples[i]
                sample_beds[sample_name].append(
                    (chrom, bed_start, bed_end, svtype)
                )
                sample_record_counts[sample_name][svtype] += 1

    if total_variants == 0:
        LOG.error("No variants found in VCF.")
        sys.exit(1)

    # ── Write per-sample BED files ────────────────────────────────────────
    samples_with_data = 0
    for sample_name, records in sample_beds.items():
        if not records:
            continue
        samples_with_data += 1
        bed_path = os.path.join(per_sample_dir, f"{sample_name}.bed")
        # Sort by chrom, start
        records.sort(key=lambda r: (r[0], r[1]))
        with open(bed_path, "w") as fh:
            for chrom, start, end, svtype in records:
                fh.write(f"{chrom}\t{start}\t{end}\t{svtype}\n")

    LOG.info(
        "Wrote BED files for %d / %d samples to %s",
        samples_with_data, len(samples), per_sample_dir,
    )

    # ── Compute and print summary ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("TRUTH SET SUMMARY")
    print("=" * 72)
    print(f"\nVCF: {vcf_path}")
    print(f"Total variant records: {total_variants:,}")
    print(f"Unparseable IDs (no regex match): {unparseable_ids:,}")
    print(f"Minimum SV size filter: {min_size:,} bp")
    print(f"Total samples: {len(samples):,}")
    print(f"Samples with ≥1 DEL/DUP call: {samples_with_data:,}")

    print(f"\n{'─' * 72}")
    print("SV TYPES IN VCF (all sizes)")
    print(f"{'─' * 72}")
    for svtype in sorted(type_total.keys()):
        print(f"  {svtype:>10s}:  {type_total[svtype]:>8,}")

    print(f"\n{'─' * 72}")
    print(f"INCLUDED (DEL/DUP ≥ {min_size:,} bp)")
    print(f"{'─' * 72}")
    total_included = 0
    for svtype in ("DEL", "DUP"):
        n = type_kept.get(svtype, 0)
        total_included += n
        print(f"  {svtype:>10s}:  {n:>8,}")
    print(f"  {'TOTAL':>10s}:  {total_included:>8,}")

    print(f"\n{'─' * 72}")
    print("EXCLUDED – by SV type (not DEL/DUP)")
    print(f"{'─' * 72}")
    for svtype in sorted(type_excluded_type.keys()):
        print(f"  {svtype:>10s}:  {type_excluded_type[svtype]:>8,}")
    print(f"  {'TOTAL':>10s}:  {sum(type_excluded_type.values()):>8,}")

    print(f"\n{'─' * 72}")
    print(f"EXCLUDED – DEL/DUP below {min_size:,} bp")
    print(f"{'─' * 72}")
    for svtype in ("DEL", "DUP"):
        n = type_excluded_size.get(svtype, 0)
        print(f"  {svtype:>10s}:  {n:>8,}")
    print(f"  {'TOTAL':>10s}:  {sum(type_excluded_size.values()):>8,}")

    print(f"\n{'─' * 72}")
    print("SIZE DISTRIBUTION OF INCLUDED VARIANTS (bp)")
    print(f"{'─' * 72}")
    import numpy as np

    for svtype in ("DEL", "DUP"):
        sizes = kept_sizes.get(svtype, [])
        if not sizes:
            print(f"  {svtype}: no variants")
            continue
        arr = np.array(sizes)
        print(f"  {svtype}:")
        print(f"    count  = {len(arr):>10,}")
        print(f"    min    = {int(arr.min()):>10,} bp")
        print(f"    median = {int(np.median(arr)):>10,} bp")
        print(f"    mean   = {int(arr.mean()):>10,} bp")
        print(f"    max    = {int(arr.max()):>10,} bp")
        print(f"    total  = {int(arr.sum()):>10,} bp")

    # Per-sample summary stats
    all_sample_counts = []
    for s in samples:
        n = sum(sample_record_counts[s].values())
        if n > 0:
            all_sample_counts.append(n)
    if all_sample_counts:
        arr = np.array(all_sample_counts)
        print(f"\n{'─' * 72}")
        print("PER-SAMPLE CALL COUNTS (samples with ≥1 call)")
        print(f"{'─' * 72}")
        print(f"  min    = {int(arr.min()):>10,}")
        print(f"  median = {int(np.median(arr)):>10,}")
        print(f"  mean   = {int(arr.mean()):>10,}")
        print(f"  max    = {int(arr.max()):>10,}")

    # Write a machine-readable summary TSV
    summary_path = os.path.join(output_dir, "truth_set_summary.tsv")
    with open(summary_path, "w") as fh:
        fh.write("metric\tvalue\n")
        fh.write(f"total_variants\t{total_variants}\n")
        fh.write(f"unparseable_ids\t{unparseable_ids}\n")
        fh.write(f"min_size_filter_bp\t{min_size}\n")
        fh.write(f"total_samples\t{len(samples)}\n")
        fh.write(f"samples_with_calls\t{samples_with_data}\n")
        for svtype in sorted(type_total.keys()):
            fh.write(f"type_total_{svtype}\t{type_total[svtype]}\n")
        for svtype in ("DEL", "DUP"):
            fh.write(f"type_kept_{svtype}\t{type_kept.get(svtype, 0)}\n")
            fh.write(f"type_excluded_size_{svtype}\t{type_excluded_size.get(svtype, 0)}\n")
        for svtype in sorted(type_excluded_type.keys()):
            fh.write(f"type_excluded_type_{svtype}\t{type_excluded_type[svtype]}\n")

    print(f"\n  Summary TSV written to: {summary_path}")
    print("=" * 72)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Parse phased SV VCF → per-sample BED truth sets.",
    )
    parser.add_argument(
        "--vcf",
        required=True,
        help="Path to the shapeit5-phased SV VCF (.vcf.gz).",
    )
    parser.add_argument(
        "--output-dir",
        default="truth_sets",
        help="Output directory (default: truth_sets).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=_DEFAULT_MIN_SIZE,
        help="Minimum SV size in bp to include (default: %(default)s).",
    )
    args = parser.parse_args()
    process_vcf(args.vcf, args.output_dir, min_size=args.min_size)


if __name__ == "__main__":
    main()

