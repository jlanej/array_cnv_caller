#!/usr/bin/env python3
"""
litmus_test.py – LRR / BAF probe-level quality assessment across SV states.

Reads a multi-sample BCF (with FORMAT/LRR and FORMAT/BAF) together with
per-sample truth-set BED files, labels every probe as DEL / NORMAL / DUP,
and produces:

1. **probe_stats.tsv.gz**  – probe-level table (sample, chrom, pos, lrr,
   baf, state, region_size, blacklist_region) for downstream analyses.
2. **summary_stats.tsv**   – per-state aggregate statistics (mean, median,
   std, IQR, skewness, kurtosis, N) for LRR and BAF.
3. **litmus_report.html**  – self-contained interactive Plotly dashboard
   with histograms, violin/box-plots, 2-D scatter, and per-chromosome
   break-downs.  Drop-down menus and sliders allow real-time filtering by
   chromosome, sample, SV-region size, LRR/BAF ranges, and blacklist region
   type (centromere / telomere / ENCODE blacklist / custom).

CLI
~~~
    python scripts/litmus_test.py \\
        --bcf multi.bcf \\
        --truth-dir truth_sets/per_sample/ \\
        --output-dir litmus_output/ \\
        --blacklist-dir resources/blacklists/

Blacklist BED files in ``resources/blacklists/`` are packaged with the repo
and document known problematic regions (centromeres, telomeres, ENCODE
blacklist v2).  See ``resources/blacklists/SOURCES.md`` for full citations.

The script reuses the BCF-reading and labelling conventions established
in ``ml_cnv_calling.py`` (half-open BED intervals, DEL=0 / NORMAL=1 /
DUP=2 encoding).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pysam

# ---------------------------------------------------------------------------
# Constants  (mirrors ml_cnv_calling.py)
# ---------------------------------------------------------------------------
CLASS_DEL = 0
CLASS_NORMAL = 1
CLASS_DUP = 2
CLASS_NAMES = {CLASS_DEL: "DEL", CLASS_NORMAL: "NORMAL", CLASS_DUP: "DUP"}
SVTYPE_TO_CLASS = {"DEL": CLASS_DEL, "DUP": CLASS_DUP}

LOG = logging.getLogger(__name__)


def _chrom_sort_key(c: str) -> Tuple[int, int]:
    """Sort key that orders numeric chromosomes first, then alpha."""
    suffix = c.replace("chr", "")
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, ord(suffix[0]) if suffix else 0)


# ===================================================================
# BCF / BED helpers
# ===================================================================
def get_bcf_samples(bcf_path: str) -> List[str]:
    """Return the list of sample names from a BCF/VCF header."""
    vcf = pysam.VariantFile(bcf_path)
    samples = list(vcf.header.samples)
    vcf.close()
    return samples


def match_samples(
    bcf_path: str, truth_dir: str
) -> Tuple[List[str], List[str], List[str]]:
    """Match BCF sample names to per-sample truth BED files."""
    bcf_samples = set(get_bcf_samples(bcf_path))
    truth_samples: set[str] = set()
    for fname in os.listdir(truth_dir):
        if fname.endswith(".bed"):
            truth_samples.add(fname[:-4])
    matched = sorted(bcf_samples & truth_samples)
    bcf_only = sorted(bcf_samples - truth_samples)
    truth_only = sorted(truth_samples - bcf_samples)
    return matched, bcf_only, truth_only


def load_truth_intervals(
    truth_dir: str, samples: List[str]
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Pre-load all per-sample truth BED files into sorted numpy arrays.

    Returns a nested dict:  ``intervals[sample][chrom]`` →
    ``(starts, ends, class_labels, region_sizes)`` with arrays sorted by
    start position for efficient binary search.
    """
    all_intervals: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    for sample in samples:
        bed_path = os.path.join(truth_dir, f"{sample}.bed")
        truth = pd.read_csv(
            bed_path,
            sep="\t",
            header=None,
            names=["chrom", "start", "end", "svtype"],
            dtype={"chrom": str, "start": int, "end": int, "svtype": str},
        )
        # Keep only DEL/DUP rows.
        truth = truth[truth["svtype"].isin(SVTYPE_TO_CLASS)]
        sample_intervals: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for chrom, grp in truth.groupby("chrom"):
            grp = grp.sort_values("start")
            starts = grp["start"].values.astype(np.int64)
            ends = grp["end"].values.astype(np.int64)
            cls = np.array(
                [SVTYPE_TO_CLASS[s] for s in grp["svtype"]], dtype=np.int64
            )
            sizes = ends - starts
            sample_intervals[chrom] = (starts, ends, cls, sizes)
        all_intervals[sample] = sample_intervals
    return all_intervals


def classify_probe(
    intervals: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    chrom: str,
    pos: int,
) -> Tuple[int, int]:
    """Classify a single probe using binary search on sorted intervals.

    Parameters
    ----------
    intervals : dict
        Per-chromosome interval arrays for one sample
        (as returned by :func:`load_truth_intervals` for a single sample).
    chrom : str
        Chromosome of the probe.
    pos : int
        Genomic position of the probe (1-based).

    Returns
    -------
    (class_label, region_size) : tuple[int, int]
        Class label (0=DEL, 1=NORMAL, 2=DUP) and the size of the
        overlapping truth region (0 when NORMAL).
    """
    chrom_ivls = intervals.get(chrom)
    if chrom_ivls is None:
        return CLASS_NORMAL, 0
    starts, ends, cls, sizes = chrom_ivls
    # Binary search: find the rightmost interval whose start <= pos.
    idx = int(np.searchsorted(starts, pos, side="right")) - 1
    # Walk backwards (handles rare overlapping intervals).
    while idx >= 0:
        if starts[idx] > pos:
            idx -= 1
            continue
        if pos < ends[idx]:
            return int(cls[idx]), int(sizes[idx])
        # All earlier intervals have start <= starts[idx] <= pos,
        # so if this one doesn't contain pos, none further back will
        # unless there are overlapping regions with different starts.
        # Walk back only while start == starts[idx] for overlaps at
        # the same position; otherwise stop.
        if idx > 0 and starts[idx - 1] == starts[idx]:
            idx -= 1
            continue
        break
    return CLASS_NORMAL, 0


def label_probes(
    probes: pd.DataFrame, truth_bed: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Label probes using a truth-set BED and record region sizes.

    This is a convenience wrapper that loads one BED file and classifies
    all probes.  For multi-sample work use :func:`load_truth_intervals`
    + :func:`classify_probe` instead (avoids repeated I/O).

    Returns
    -------
    labels : np.ndarray  – per-probe class (0=DEL, 1=NORMAL, 2=DUP).
    region_sizes : np.ndarray – size (bp) of the overlapping truth region,
        or 0 for NORMAL probes.
    """
    sample_name = os.path.splitext(os.path.basename(truth_bed))[0]
    truth_dir = os.path.dirname(truth_bed)
    ivl_map = load_truth_intervals(truth_dir, [sample_name])
    ivls = ivl_map[sample_name]

    labels = np.full(len(probes), CLASS_NORMAL, dtype=np.int64)
    region_sizes = np.zeros(len(probes), dtype=np.int64)

    for i, (_, row) in enumerate(probes.iterrows()):
        cls, sz = classify_probe(ivls, row["chrom"], row["pos"])
        labels[i] = cls
        region_sizes[i] = sz

    return labels, region_sizes


# ===================================================================
# Blacklist region helpers
# ===================================================================

#: Sentinel value used in the probe table when a probe is not in any
#: known problematic region.
BLACKLIST_NONE = ""

#: Directory within the repository that ships with pre-packaged BED files.
_REPO_BLACKLIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "resources", "blacklists",
)


def load_blacklist_regions(
    blacklist_dir: str,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Load all ``.bed`` files in *blacklist_dir* into sorted interval arrays.

    Each BED file's stem (filename without the ``.bed`` suffix) is used as the
    region-type label (e.g. ``chm13v2_centromeres.bed`` →
    ``"chm13v2_centromeres"``).
    Lines beginning with ``#`` are treated as comments and ignored.

    Parameters
    ----------
    blacklist_dir : str
        Directory containing ``.bed`` files.  Files without a ``.bed`` suffix
        are silently skipped.

    Returns
    -------
    dict
        Mapping ``region_type → chrom → (starts, ends, indices)`` where
        *starts* and *ends* are sorted ``np.int64`` arrays and *indices* is
        an array of integer indices into a per-region-type label list (always
        all zeros here; kept for API symmetry with
        :func:`load_truth_intervals`).
    """
    blacklist: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    if not os.path.isdir(blacklist_dir):
        LOG.warning("Blacklist directory not found: %s — skipping.", blacklist_dir)
        return blacklist

    for fname in sorted(os.listdir(blacklist_dir)):
        if not fname.endswith(".bed"):
            continue
        region_type = fname[:-4]  # strip .bed
        bed_path = os.path.join(blacklist_dir, fname)
        rows: list[dict] = []
        with open(bed_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                try:
                    rows.append(
                        {"chrom": parts[0], "start": int(parts[1]), "end": int(parts[2])}
                    )
                except ValueError:
                    continue

        if not rows:
            LOG.warning("Blacklist file %s has no usable rows — skipping.", fname)
            continue

        chrom_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        import pandas as _pd
        bl_df = _pd.DataFrame(rows)
        for chrom, grp in bl_df.groupby("chrom"):
            grp = grp.sort_values("start")
            starts = grp["start"].values.astype(np.int64)
            ends = grp["end"].values.astype(np.int64)
            indices = np.zeros(len(starts), dtype=np.int64)
            chrom_map[chrom] = (starts, ends, indices)
        blacklist[region_type] = chrom_map
        LOG.debug(
            "Loaded blacklist '%s': %d chromosomes, %d total regions.",
            region_type,
            len(chrom_map),
            sum(len(v[0]) for v in chrom_map.values()),
        )

    LOG.info(
        "Loaded %d blacklist track(s) from %s: %s",
        len(blacklist),
        blacklist_dir,
        list(blacklist.keys()),
    )
    return blacklist


def classify_probe_blacklist(
    blacklist: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]],
    chrom: str,
    pos: int,
) -> str:
    """Return the first blacklist region type that contains *pos*, or ``""``.

    Region types are checked in sorted order (alphabetical by BED filename
    stem).  If a probe falls in multiple overlapping blacklist tracks the
    first match is returned.

    Parameters
    ----------
    blacklist : dict
        Nested mapping returned by :func:`load_blacklist_regions`.
    chrom : str
        Chromosome of the probe.
    pos : int
        Genomic position of the probe (1-based VCF convention).

    Returns
    -------
    str
        Region-type label (e.g. ``"chm13v2_centromeres"``), or ``""`` if the
        probe does not overlap any blacklist region.
    """
    for region_type, chrom_map in sorted(blacklist.items()):
        chrom_ivls = chrom_map.get(chrom)
        if chrom_ivls is None:
            continue
        starts, ends, _ = chrom_ivls
        idx = int(np.searchsorted(starts, pos, side="right")) - 1
        while idx >= 0:
            if starts[idx] > pos:
                idx -= 1
                continue
            if pos < ends[idx]:
                return region_type
            if idx > 0 and starts[idx - 1] == starts[idx]:
                idx -= 1
                continue
            break
    return BLACKLIST_NONE


def annotate_probes_with_blacklist(
    df: "pd.DataFrame",
    blacklist_dir: str,
) -> "pd.DataFrame":
    """Add a ``blacklist_region`` column to *df* in place.

    Each probe is classified against all BED files in *blacklist_dir*.  The
    resulting column contains the region-type label (BED filename stem) for
    the first matching region, or an empty string for probes outside all
    blacklist tracks.

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level table with at least ``chrom`` and ``pos`` columns.
    blacklist_dir : str
        Directory of ``.bed`` blacklist files.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with a new ``blacklist_region`` column added.
    """
    blacklist = load_blacklist_regions(blacklist_dir)
    if not blacklist:
        df["blacklist_region"] = BLACKLIST_NONE
        return df

    df["blacklist_region"] = df.apply(
        lambda row: classify_probe_blacklist(blacklist, row["chrom"], row["pos"]),
        axis=1,
    )
    return df


def build_blacklist_summary(df: "pd.DataFrame") -> "pd.DataFrame":
    """Summarise probe and SV-call counts per blacklist region type and state.

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level table with ``blacklist_region`` and ``state`` columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``blacklist_region``, ``state``, ``n_probes``,
        ``pct_of_state``.  One row per (region, state) combination.
    """
    if "blacklist_region" not in df.columns:
        return pd.DataFrame(
            columns=["blacklist_region", "state", "n_probes", "pct_of_state"]
        )

    state_totals = df["state"].value_counts()
    rows: list[dict] = []
    for region, grp in df.groupby("blacklist_region", sort=True):
        for state, sub in grp.groupby("state"):
            n = len(sub)
            total = state_totals.get(state, 1)
            rows.append(
                {
                    "blacklist_region": region if region else "(none)",
                    "state": state,
                    "n_probes": n,
                    "pct_of_state": round(100.0 * n / total, 2),
                }
            )
    return pd.DataFrame(rows)


# ===================================================================
def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-state aggregate statistics for LRR and BAF.

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level table with columns: lrr, baf, state.

    Returns
    -------
    pd.DataFrame
        One row per (state, metric) combination with descriptive stats.
    """
    rows: list[dict] = []
    for state in ["DEL", "NORMAL", "DUP"]:
        sub = df[df["state"] == state]
        if sub.empty:
            continue
        for metric in ("lrr", "baf"):
            vals = sub[metric].dropna()
            q1, median, q3 = np.nanpercentile(vals, [25, 50, 75])
            rows.append(
                {
                    "state": state,
                    "metric": metric,
                    "n": len(vals),
                    "mean": float(np.nanmean(vals)),
                    "median": float(median),
                    "std": float(np.nanstd(vals)),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                    "min": float(np.nanmin(vals)),
                    "max": float(np.nanmax(vals)),
                    "skew": float(vals.skew()) if len(vals) > 2 else float("nan"),
                    "kurtosis": float(vals.kurtosis()) if len(vals) > 2 else float("nan"),
                    "pct_1": float(np.nanpercentile(vals, 1)),
                    "pct_99": float(np.nanpercentile(vals, 99)),
                }
            )
    return pd.DataFrame(rows)


# ===================================================================
# Data collection  (single-pass BCF scan)
# ===================================================================
def collect_probe_data(
    bcf_path: str,
    truth_dir: str,
    max_samples: Optional[int] = None,
    blacklist_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Read probes across matched samples in a **single BCF pass**.

    The BCF file is iterated exactly once.  For every variant record
    the FORMAT/LRR and FORMAT/BAF values of *all* matched samples are
    extracted and classified against pre-loaded truth intervals using
    binary search, so the total I/O is O(records) regardless of the
    number of samples.

    Parameters
    ----------
    bcf_path : str
        Multi-sample BCF with FORMAT/LRR and FORMAT/BAF.
    truth_dir : str
        Directory containing ``<sample>.bed`` truth files.
    max_samples : int, optional
        Cap the number of samples processed (useful for quick checks).
    blacklist_dir : str, optional
        Directory of ``.bed`` blacklist files (centromeres, telomeres, etc.)
        used to annotate each probe with a ``blacklist_region`` label.
        When *None* the column is added with every row set to ``""``.

    Returns
    -------
    pd.DataFrame
        Columns: sample, chrom, pos, lrr, baf, state, region_size,
        blacklist_region.
    """
    matched, bcf_only, truth_only = match_samples(bcf_path, truth_dir)
    LOG.info(
        "Samples — matched: %d, bcf_only: %d, truth_only: %d",
        len(matched),
        len(bcf_only),
        len(truth_only),
    )
    if not matched:
        raise ValueError(
            "No overlapping samples between BCF and truth directory."
        )

    if max_samples is not None:
        matched = matched[:max_samples]
        LOG.info("Capping to %d samples: %s", max_samples, matched)

    # ── Pre-load all truth intervals into memory ──────────────────────
    LOG.info("Pre-loading truth intervals for %d samples …", len(matched))
    all_intervals = load_truth_intervals(truth_dir, matched)

    # ── Pre-load blacklist intervals (if provided) ────────────────────
    blacklist: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    if blacklist_dir:
        blacklist = load_blacklist_regions(blacklist_dir)

    # ── Single pass through the BCF ──────────────────────────────────
    LOG.info("Starting single-pass BCF scan …")
    vcf = pysam.VariantFile(bcf_path)

    # Pre-allocate columnar lists for efficiency.
    col_sample: list[str] = []
    col_chrom: list[str] = []
    col_pos: list[int] = []
    col_lrr: list[float] = []
    col_baf: list[float] = []
    col_state: list[str] = []
    col_region_size: list[int] = []
    col_blacklist: list[str] = []
    samples_seen: set[str] = set()

    n_records = 0
    for rec in vcf.fetch():
        chrom = rec.chrom
        pos = rec.pos
        n_records += 1
        if n_records % 100_000 == 0:
            LOG.info("  … %d BCF records scanned", n_records)

        # Classify blacklist region once per position (shared across samples)
        bl_region = (
            classify_probe_blacklist(blacklist, chrom, pos)
            if blacklist else BLACKLIST_NONE
        )

        for sample in matched:
            fmt = rec.samples[sample]
            lrr = fmt.get("LRR")
            baf = fmt.get("BAF")
            if lrr is None or baf is None:
                continue

            cls, region_size = classify_probe(
                all_intervals[sample], chrom, pos
            )

            col_sample.append(sample)
            col_chrom.append(chrom)
            col_pos.append(pos)
            col_lrr.append(float(lrr))
            col_baf.append(float(baf))
            col_state.append(CLASS_NAMES[cls])
            col_region_size.append(region_size)
            col_blacklist.append(bl_region)
            samples_seen.add(sample)

    vcf.close()
    LOG.info(
        "BCF scan complete: %d records × %d samples → %d probe entries.",
        n_records,
        len(matched),
        len(col_sample),
    )

    if not col_sample:
        raise ValueError("No probe data collected from any sample.")

    result = pd.DataFrame(
        {
            "sample": col_sample,
            "chrom": col_chrom,
            "pos": col_pos,
            "lrr": col_lrr,
            "baf": col_baf,
            "state": col_state,
            "region_size": col_region_size,
            "blacklist_region": col_blacklist,
        }
    )
    LOG.info(
        "Collected %d probes across %d samples.",
        len(result),
        len(samples_seen),
    )
    return result


# ===================================================================
# Interactive Plotly dashboard
# ===================================================================
def build_dashboard(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: str,
    *,
    bcf_path: Optional[str] = None,
    truth_dir: Optional[str] = None,
    blacklist_dir: Optional[str] = None,
) -> None:
    """Generate a self-contained interactive HTML dashboard.

    The dashboard includes:
    * An "About This Report" section with data sources, signal explanations,
      sample list, truth-set statistics, methods, and a plot guide
    * LRR and BAF probability-density histograms per state (normalised so that
      the dominant NORMAL class does not obscure DEL / DUP shapes) — primary
      diagnostic figure shown first
    * DEL vs DUP direct comparison histograms (NORMAL excluded)
    * 2-D LRR×BAF scatter coloured by state (sub-sampled for speed)
    * Per-chromosome LRR distributions
    * Summary statistics table
    * **Blacklist region summary** — probe and SV-call counts per problematic
      region type (centromere, telomere, ENCODE blacklist, etc.) with a
      stacked bar chart visualisation
    * Interactive filter panel that regenerates histograms **and** violin plots
      in real time (chromosome, sample, region-size, LRR-range, and blacklist
      region filters to exclude/isolate problematic regions)

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level table (sample, chrom, pos, lrr, baf, state, region_size,
        blacklist_region).
    summary : pd.DataFrame
        Aggregate statistics from :func:`compute_summary_stats`.
    output_path : str
        File path for the HTML report.
    bcf_path : str, optional
        Path to the source BCF file; when provided the filename is shown in
        the About section so readers know exactly which data were used.
    truth_dir : str, optional
        Path to the truth-set BED directory; when provided it is shown in the
        About section alongside the SV truth description.
    blacklist_dir : str, optional
        Path to the blacklist BED directory; when provided, source information
        is shown in the About section and the Blacklist Region Summary section
        is included in the dashboard.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        LOG.error(
            "plotly is required for dashboard generation.  "
            "Install with:  pip install plotly"
        )
        raise

    # Publication-quality colour palette (colourblind-friendly)
    STATE_COLOURS = {"DEL": "#E15759", "NORMAL": "#4E79A7", "DUP": "#F28E2B"}
    STATES = ["DEL", "NORMAL", "DUP"]
    TEMPLATE = "plotly_white"
    FONT = dict(family="Arial, Helvetica, sans-serif", size=13, color="#2c2c2c")
    TITLE_FONT = dict(family="Arial, Helvetica, sans-serif", size=15, color="#1a1a1a")
    AXIS_FONT = dict(family="Arial, Helvetica, sans-serif", size=12)

    def _layout_defaults(**extra) -> dict:
        """Return a dict of common publication-quality layout settings."""
        base = dict(
            template=TEMPLATE,
            font=FONT,
            title_font=TITLE_FONT,
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#cccccc",
                borderwidth=1,
                font=dict(size=12),
            ),
            margin=dict(l=70, r=30, t=70, b=60),
        )
        base.update(extra)
        return base

    def _axis_style(title: str) -> dict:
        return dict(
            title_text=title,
            title_font=AXIS_FONT,
            tickfont=AXIS_FONT,
            showgrid=True,
            gridcolor="#ebebeb",
            zeroline=True,
            zerolinecolor="#cccccc",
        )

    # -- Utility: subsample for large datasets -------------------------
    def _subsample(data: pd.DataFrame, n: int = 50_000) -> pd.DataFrame:
        if len(data) <= n:
            return data
        return data.sample(n=n, random_state=42)

    def _hist_toggle_menu(title_density: str, title_count: str) -> list:
        """Return a Plotly updatemenus list with Density / Count toggle buttons.

        The active button (index 0) defaults to Density. Clicking a button
        restyles all histogram traces and updates y-axis titles and the
        chart title via relayout.
        """
        return [
            dict(
                type="buttons",
                direction="left",
                active=0,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.0, xanchor="right",
                y=1.14, yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#aaaaaa",
                borderwidth=1,
                font=dict(size=11, family="Arial, Helvetica, sans-serif"),
                buttons=[
                    dict(
                        label="Density",
                        method="update",
                        args=[
                            {"histnorm": "probability density"},
                            {
                                "yaxis.title.text": "Probability Density",
                                "yaxis2.title.text": "Probability Density",
                                "title.text": title_density,
                            },
                        ],
                    ),
                    dict(
                        label="Count",
                        method="update",
                        args=[
                            {"histnorm": ""},
                            {
                                "yaxis.title.text": "Count",
                                "yaxis2.title.text": "Count",
                                "title.text": title_count,
                            },
                        ],
                    ),
                ],
            )
        ]

    # ── 1. Density histograms (LRR & BAF) ────────────────────────────
    # Using histnorm="probability density" means each state's curve
    # integrates to 1, so the rare DEL / DUP states are equally visible
    # alongside the dominant NORMAL class.
    # go.Histogram embeds raw x-values (Plotly.js bins client-side), so
    # we cap each state at 200 K points — the density shape is faithfully
    # preserved by subsampling, and the legend label always shows the
    # full population count.
    fig_hist = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("LRR Probability Density by State",
                        "BAF Probability Density by State"),
        horizontal_spacing=0.10,
    )
    for state in STATES:
        sub_full = df[df["state"] == state]
        colour = STATE_COLOURS[state]
        n_state = len(sub_full)          # full count for the legend label
        sub = _subsample(sub_full, 200_000)
        fig_hist.add_trace(
            go.Histogram(
                x=sub["lrr"],
                name=f"{state} (n={n_state:,})",
                marker_color=colour,
                opacity=0.65,
                nbinsx=150,
                histnorm="probability density",
                legendgroup=state,
                showlegend=True,
            ),
            row=1, col=1,
        )
        fig_hist.add_trace(
            go.Histogram(
                x=sub["baf"],
                name=f"{state} BAF",
                marker_color=colour,
                opacity=0.65,
                nbinsx=150,
                histnorm="probability density",
                legendgroup=state,
                showlegend=False,
            ),
            row=1, col=2,
        )
    fig_hist.update_layout(
        barmode="overlay",
        title_text="LRR &amp; BAF Probability Density by Copy-Number State",
        height=520,
        **_layout_defaults(),
    )
    fig_hist.update_xaxes(**_axis_style("LRR"), row=1, col=1)
    fig_hist.update_xaxes(**_axis_style("BAF"), row=1, col=2)
    fig_hist.update_yaxes(**_axis_style("Probability Density"), row=1, col=1)
    fig_hist.update_yaxes(**_axis_style("Probability Density"), row=1, col=2)
    # Density ↔ Count toggle (defaults to Density)
    fig_hist.update_layout(
        updatemenus=_hist_toggle_menu(
            title_density="LRR & BAF Probability Density by Copy-Number State",
            title_count="LRR & BAF Distributions by Copy-Number State",
        )
    )

    # ── 2. DEL vs DUP direct comparison (NORMAL excluded) ────────────
    # Removing the dominant NORMAL class reveals subtle shape differences
    # between deletions and duplications at the same density scale.
    fig_del_dup = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("LRR: DEL vs DUP (NORMAL excluded)",
                        "BAF: DEL vs DUP (NORMAL excluded)"),
        horizontal_spacing=0.10,
    )
    for state in ["DEL", "DUP"]:
        sub_full = df[df["state"] == state]
        colour = STATE_COLOURS[state]
        n_state = len(sub_full)          # full count for the legend label
        sub = _subsample(sub_full, 200_000)
        fig_del_dup.add_trace(
            go.Histogram(
                x=sub["lrr"],
                name=f"{state} (n={n_state:,})",
                marker_color=colour,
                opacity=0.70,
                nbinsx=120,
                histnorm="probability density",
                legendgroup=state,
                showlegend=True,
            ),
            row=1, col=1,
        )
        fig_del_dup.add_trace(
            go.Histogram(
                x=sub["baf"],
                name=f"{state} BAF",
                marker_color=colour,
                opacity=0.70,
                nbinsx=120,
                histnorm="probability density",
                legendgroup=state,
                showlegend=False,
            ),
            row=1, col=2,
        )
    fig_del_dup.update_layout(
        barmode="overlay",
        title_text="DEL vs DUP Distribution Comparison",
        height=520,
        **_layout_defaults(),
    )
    fig_del_dup.update_xaxes(**_axis_style("LRR"), row=1, col=1)
    fig_del_dup.update_xaxes(**_axis_style("BAF"), row=1, col=2)
    fig_del_dup.update_yaxes(**_axis_style("Probability Density"), row=1, col=1)
    fig_del_dup.update_yaxes(**_axis_style("Probability Density"), row=1, col=2)
    # Density ↔ Count toggle (defaults to Density)
    fig_del_dup.update_layout(
        updatemenus=_hist_toggle_menu(
            title_density="DEL vs DUP Distribution Comparison",
            title_count="DEL vs DUP Distribution Comparison (Counts)",
        )
    )

    # ── 3. 2-D scatter (LRR vs BAF) ─────────────────────────────────
    fig_scatter = go.Figure()
    # Plot NORMAL first (background) then DEL/DUP on top for visibility
    for state in ["NORMAL", "DEL", "DUP"]:
        n_pts = 15_000 if state == "NORMAL" else 20_000
        sub = _subsample(df[df["state"] == state], n=n_pts)
        fig_scatter.add_trace(
            go.Scattergl(
                x=sub["lrr"],
                y=sub["baf"],
                mode="markers",
                name=state,
                marker=dict(
                    color=STATE_COLOURS[state],
                    size=3 if state == "NORMAL" else 4,
                    opacity=0.25 if state == "NORMAL" else 0.55,
                ),
            )
        )
    fig_scatter.update_layout(
        title_text="LRR vs BAF (sub-sampled; DEL/DUP plotted over NORMAL)",
        xaxis=dict(**_axis_style("LRR")),
        yaxis=dict(**_axis_style("BAF")),
        height=620,
        **_layout_defaults(),
    )

    # ── 4. Per-chromosome LRR box-plots ─────────────────────────────
    # go.Box also embeds raw y-values, so cap each state to keep HTML size
    # manageable without affecting the per-chromosome quartile estimates.
    chroms = sorted(df["chrom"].unique(), key=_chrom_sort_key)
    fig_chrom = go.Figure()
    for state in STATES:
        sub = _subsample(df[df["state"] == state], 50_000)
        fig_chrom.add_trace(
            go.Box(
                x=sub["chrom"],
                y=sub["lrr"],
                name=state,
                marker_color=STATE_COLOURS[state],
                line_color=STATE_COLOURS[state],
                boxmean="sd",
                opacity=0.85,
            )
        )
    fig_chrom.update_layout(
        title_text="LRR Distribution per Chromosome by Copy-Number State",
        xaxis=dict(**_axis_style("Chromosome"),
                   categoryorder="array", categoryarray=chroms),
        yaxis=dict(**_axis_style("LRR")),
        boxmode="group",
        height=520,
        **_layout_defaults(),
    )

    # ── 5. Region size vs mean LRR (for DEL/DUP only) ───────────────
    fig_size = go.Figure()
    for state in ["DEL", "DUP"]:
        sub = df[(df["state"] == state) & (df["region_size"] > 0)]
        if sub.empty:
            continue
        agg = (
            sub.groupby("region_size")
            .agg(mean_lrr=("lrr", "mean"), mean_baf=("baf", "mean"),
                 n=("lrr", "size"))
            .reset_index()
        )
        fig_size.add_trace(
            go.Scattergl(
                x=agg["region_size"],
                y=agg["mean_lrr"],
                mode="markers",
                name=f"{state} mean LRR",
                marker=dict(
                    color=STATE_COLOURS[state],
                    size=np.clip(np.log2(agg["n"].values + 1) * 2, 4, 18),
                    opacity=0.65,
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                ),
                text=[f"n={n}" for n in agg["n"]],
                hovertemplate="%{text}<br>size=%{x:,} bp<br>mean LRR=%{y:.4f}",
            )
        )
    fig_size.update_layout(
        title_text="Truth Region Size vs Mean LRR (DEL / DUP)",
        xaxis=dict(**_axis_style("Region size (bp)"), type="log"),
        yaxis=dict(**_axis_style("Mean LRR")),
        height=520,
        **_layout_defaults(),
    )

    # ── 6. Per-sample state counts ───────────────────────────────────
    sample_counts = (
        df.groupby(["sample", "state"]).size().unstack(fill_value=0)
    )
    fig_samples = go.Figure()
    for state in STATES:
        if state in sample_counts.columns:
            fig_samples.add_trace(
                go.Bar(
                    x=sample_counts.index.tolist(),
                    y=sample_counts[state].tolist(),
                    name=state,
                    marker_color=STATE_COLOURS[state],
                    opacity=0.85,
                )
            )
    fig_samples.update_layout(
        barmode="stack",
        title_text="Probe Counts per Sample by Copy-Number State",
        xaxis=dict(**_axis_style("Sample")),
        yaxis=dict(**_axis_style("Probe count")),
        height=500,
        **_layout_defaults(),
    )

    # ── 7. Summary statistics table ──────────────────────────────────
    fmt_cols = [c for c in summary.columns if c not in ("state", "metric")]
    header_vals = ["State", "Metric"] + [c.upper() for c in fmt_cols]
    # Per-column fill colours:
    #   col 0 (State)  – each cell gets the colour of its copy-number state
    #   col 1 (Metric) – alternating light-grey / white
    #   remaining cols – same alternating pattern
    state_col_colors = [STATE_COLOURS.get(s, "#f9f9f9") for s in summary["state"]]
    n_rows = len(summary)
    alt_colors = ["#f9f9f9" if i % 2 == 0 else "#ffffff" for i in range(n_rows)]
    fill_colors = [state_col_colors] + [alt_colors] * (len(fmt_cols) + 1)
    cell_vals = [
        summary["state"].tolist(),
        summary["metric"].tolist(),
    ] + [
        [f"{v:.4f}" if isinstance(v, float) else str(v) for v in summary[c]]
        for c in fmt_cols
    ]
    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header_vals,
                    align="center",
                    fill_color="#4E79A7",
                    font=dict(color="white", size=12,
                              family="Arial, Helvetica, sans-serif"),
                    height=32,
                ),
                cells=dict(
                    values=cell_vals,
                    align="center",
                    fill_color=fill_colors,
                    font=dict(size=11,
                              family="Arial, Helvetica, sans-serif"),
                    height=26,
                ),
            )
        ]
    )
    fig_table.update_layout(
        title_text="Summary Statistics by Copy-Number State",
        height=380,
        **_layout_defaults(),
    )

    # ── 8. Blacklist region summary (if blacklist_region column present) ──
    fig_blacklist: Optional["go.Figure"] = None  # type: ignore[name-defined]
    bl_summary: Optional[pd.DataFrame] = None
    if "blacklist_region" in df.columns:
        bl_summary = build_blacklist_summary(df)
        has_bl = not bl_summary.empty and any(
            r != "(none)" for r in bl_summary["blacklist_region"]
        )
        if has_bl:
            # Stacked bar: blacklist region type × SV state
            bl_pivot = (
                bl_summary[bl_summary["blacklist_region"] != "(none)"]
                .pivot_table(
                    index="blacklist_region",
                    columns="state",
                    values="n_probes",
                    fill_value=0,
                )
                .reset_index()
            )
            fig_blacklist = go.Figure()
            for state in STATES:
                if state in bl_pivot.columns:
                    fig_blacklist.add_trace(
                        go.Bar(
                            x=bl_pivot["blacklist_region"].tolist(),
                            y=bl_pivot[state].tolist(),
                            name=state,
                            marker_color=STATE_COLOURS[state],
                            opacity=0.85,
                        )
                    )
            fig_blacklist.update_layout(
                barmode="stack",
                title_text="Probe Counts per Problematic Region Type by Copy-Number State",
                xaxis=dict(**_axis_style("Blacklist Region")),
                yaxis=dict(**_axis_style("Probe count")),
                height=460,
                **_layout_defaults(),
            )

    # ── Assemble HTML ─────────────────────────────────────────────────
    html_parts: list[str] = []
    html_parts.append(_html_header())
    html_parts.append('<div class="container">')

    # About / methods description (always first)
    html_parts.append(
        _about_section_html(
            df, summary,
            bcf_path=bcf_path,
            truth_dir=truth_dir,
            blacklist_dir=blacklist_dir,
        )
    )

    # Density distributions are the key starting figure
    sections = [
        (
            "LRR &amp; BAF Density Distributions",
            fig_hist,
            "Primary diagnostic figure: each copy-number state is normalised to "
            "unit area so rare DEL and DUP populations are directly comparable to "
            "the dominant NORMAL class. Toggle between Density and Count modes "
            "using the button in the top-right corner of the plot.",
        ),
        (
            "DEL vs DUP Comparison (NORMAL excluded)",
            fig_del_dup,
            "Direct comparison of deletion and duplication signal shapes with the "
            "dominant NORMAL class removed, revealing subtle distribution "
            "differences at the same density scale.",
        ),
        ("Summary Statistics", fig_table, None),
        ("LRR vs BAF Scatter", fig_scatter, None),
        ("Per-Chromosome LRR", fig_chrom, None),
        ("Region Size vs Mean LRR", fig_size, None),
        ("Per-Sample Probe Counts", fig_samples, None),
    ]

    for title, fig, note in sections:
        html_parts.append(f'<div class="section"><h2>{title}</h2>')
        if note:
            html_parts.append(f'<p class="section-note">{note}</p>')
        html_parts.append(
            fig.to_html(full_html=False, include_plotlyjs=False)
        )
        html_parts.append("</div>")

    # ── Blacklist region summary section ─────────────────────────────
    if fig_blacklist is not None and bl_summary is not None:
        html_parts.append(_blacklist_section_html(fig_blacklist, bl_summary, blacklist_dir))

    # ── Interactive filtering panel (JavaScript) ─────────────────────
    html_parts.append(_filter_panel_html(df))

    html_parts.append("</div>")  # container
    html_parts.append("</body></html>")

    with open(output_path, "w") as fh:
        fh.write("\n".join(html_parts))
    LOG.info("Dashboard written to %s", output_path)


def _about_section_html(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    bcf_path: Optional[str] = None,
    truth_dir: Optional[str] = None,
    blacklist_dir: Optional[str] = None,
) -> str:
    """Generate the 'About This Report' section with data sources and methods.

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level table; used to derive sample list and state counts.
    summary : pd.DataFrame
        Aggregate stats from :func:`compute_summary_stats`; used to show
        per-state LRR summary statistics inline.
    bcf_path : str, optional
        Source BCF path; basename is shown in the Data Sources block.
    truth_dir : str, optional
        Truth-set BED directory; shown in the Data Sources block.
    blacklist_dir : str, optional
        Blacklist BED directory; shown in the Data Sources block with a list
        of loaded region track names and their sources.
    """
    samples = sorted(df["sample"].unique().tolist())
    n_samples = len(samples)
    state_counts = df["state"].value_counts()
    n_total = len(df)

    sample_badges = " ".join(
        f'<span class="sample-badge">{s}</span>' for s in samples
    )

    # Per-state LRR stats from summary DataFrame
    lrr_rows: list[dict] = []
    for _, row in summary.iterrows():
        if row["metric"] == "lrr":
            lrr_rows.append(row.to_dict())

    def _stat_row(state: str) -> str:
        matches = [r for r in lrr_rows if r["state"] == state]
        if not matches:
            return ""
        s = matches[0]
        n_probes = int(state_counts.get(state, 0))
        return (
            f'<tr><td><span class="badge-{state}">{state}</span></td>'
            f'<td>{n_probes:,}</td>'
            f'<td>{s.get("mean", 0):.3f}</td>'
            f'<td>{s.get("median", 0):.3f}</td>'
            f'<td>{s.get("std", 0):.3f}</td>'
            f'<td>{s.get("iqr", 0):.3f}</td></tr>'
        )

    stat_rows = "".join(_stat_row(st) for st in ["DEL", "NORMAL", "DUP"])

    # Build dynamic data-source lines so the About block stays accurate
    # regardless of which BCF or truth directory the caller used.
    bcf_label = (
        f"<code>{os.path.basename(bcf_path)}</code>" if bcf_path else "a multi-sample BCF file"
    )
    truth_label = (
        f"<code>{os.path.abspath(truth_dir)}</code>" if truth_dir else "the supplied truth directory"
    )

    # Build blacklist source block
    bl_tracks: list[str] = []
    if blacklist_dir and os.path.isdir(blacklist_dir):
        for fname in sorted(os.listdir(blacklist_dir)):
            if fname.endswith(".bed"):
                bl_tracks.append(fname[:-4])
    if bl_tracks:
        bl_dir_label = f"<code>{os.path.abspath(blacklist_dir)}</code>"
        bl_track_list = "".join(
            f'<li><code>{t}</code></li>' for t in bl_tracks
        )
        blacklist_block = f"""
    <div class="about-block">
      <h3>Blacklist / Problematic Regions</h3>
      <p>Each probe is annotated against known problematic genomic regions for
      the <strong>T2T-CHM13 v2.0</strong> assembly. Probes in these regions
      can be excluded or highlighted using the
      <strong>Interactive Filter Panel</strong> below to assess their impact
      on signal distributions.</p>
      <p>Blacklist directory: {bl_dir_label}</p>
      <p style="margin:6px 0 3px;font-size:13px;font-weight:600;">Region tracks loaded:</p>
      <ul class="about-list">{bl_track_list}</ul>
      <p style="margin:8px 0 0;font-size:12px;color:#555;">
        Sources: Nurk <em>et al.</em> (T2T, <em>Science</em> 2022);
        Altemose <em>et al.</em> (centromeres, <em>Science</em> 2022);
        Amemiya <em>et al.</em> (ENCODE Blacklist, <em>Sci Rep</em> 2019).
        See <code>resources/blacklists/SOURCES.md</code> for full citations.
      </p>
    </div>"""
    else:
        blacklist_block = ""

    return f"""
<div class="section about-section">
  <h2>About This Report</h2>
  <div class="about-grid">

    <div class="about-block">
      <h3>What Are We Looking At?</h3>
      <p>This interactive dashboard is a quality-control (&ldquo;litmus test&rdquo;)
      assessment of Illumina microarray probe signals across three copy-number states:
      <span class="badge-DEL">DEL</span>&nbsp;(deletion),
      <span class="badge-NORMAL">NORMAL</span>&nbsp;(diploid), and
      <span class="badge-DUP">DUP</span>&nbsp;(duplication). It validates that the
      two key signal channels &mdash; <strong>LRR</strong> and <strong>BAF</strong>
      &mdash; are discriminative across copy-number states and that the truth-set
      labels used to train the <em>array_cnv_caller</em> ML model are
      well-calibrated.</p>
      <p>The <strong>LRR &amp; BAF Density Distributions</strong> immediately below
      are the primary diagnostic figure: clearly separated peaks across states
      indicate that the array signals are suitable for machine-learning-based
      CNV calling.</p>
    </div>

    <div class="about-block">
      <h3>Signal Types</h3>
      <dl class="signal-dl">
        <dt>LRR &mdash; Log R Ratio</dt>
        <dd>Normalised probe intensity relative to the expected diploid signal.
        <strong>Deletions</strong> shift LRR negative (&lt;&thinsp;0);
        <strong>duplications</strong> shift it positive (&gt;&thinsp;0);
        diploid probes cluster near&nbsp;0. LRR is the primary copy-number
        signal.</dd>
        <dt>BAF &mdash; B-Allele Frequency</dt>
        <dd>Allelic balance at heterozygous SNP probe positions (range
        0&ndash;1). Diploid heterozygous probes cluster at&nbsp;0.5.
        <strong>Deletions</strong> cause BAF to shift toward 0 or 1 (loss of
        heterozygosity); <strong>duplications</strong> create additional
        clusters near &asymp;&thinsp;0.33 and 0.67. BAF provides complementary
        phase information to LRR.</dd>
      </dl>
    </div>

    <div class="about-block">
      <h3>Data Sources</h3>
      <ul class="about-list">
        <li><strong>Reference assembly:</strong>
        <a href="https://doi.org/10.1126/science.abj6987" target="_blank"
           rel="noopener noreferrer">T2T-CHM13 v2.0 (GCA_009914755.4)</a>
        &mdash; Nurk <em>et al.</em> <em>Science</em> 2022. Complete, gapless
        human genome including all centromeres and telomeres.</li>
        <li><strong>Array probe data:</strong> {bcf_label} with
        per-probe <code>FORMAT/LRR</code> and <code>FORMAT/BAF</code> fields.
        Contains {n_total:,}&nbsp;labelled probe observations across
        {n_samples}&nbsp;sample{'s' if n_samples != 1 else ''}.</li>
        <li><strong>Structural variant truth set:</strong> Per-sample BED files
        from {truth_label}. DEL/DUP calls &ge;&thinsp;1,000&nbsp;bp are used
        as ground-truth copy-number intervals. Each probe overlapping a truth
        interval is labelled DEL or DUP; all remaining probes are labelled
        NORMAL.</li>
        <li><strong>Primary citation:</strong> Schloissnig&nbsp;S &amp;
        Pani&nbsp;S. <em>Long-read sequencing and structural variant
        characterization in 1,019 samples from the 1000 Genomes
        Project.</em> <em>bioRxiv</em> (2024).
        <a href="https://doi.org/10.1101/2024.04.18.590093"
           target="_blank" rel="noopener noreferrer">
           doi:10.1101/2024.04.18.590093</a></li>
      </ul>
    </div>
{blacklist_block}
    <div class="about-block">
      <h3>Samples ({n_samples})</h3>
      <div class="sample-list">{sample_badges}</div>
      <p style="margin:10px 0 6px;font-size:13px;color:#555;">
        LRR summary statistics per copy-number state:
      </p>
      <table class="stats-table">
        <thead><tr>
          <th>State</th><th>Probes</th>
          <th>LRR mean</th><th>LRR median</th><th>LRR std</th><th>LRR IQR</th>
        </tr></thead>
        <tbody>{stat_rows}</tbody>
      </table>
      <p class="section-note" style="margin-top:8px;">
        Full per-state statistics (BAF, skewness, kurtosis, percentiles) are
        available in the Summary Statistics section below.
      </p>
    </div>

    <div class="about-block">
      <h3>Methods</h3>
      <ol class="about-list">
        <li><strong>Truth preparation:</strong> Per-sample BED files are
        derived from the shapeit5-phased SV callset using
        <code>prepare_truth_set.py</code>. Only DEL and DUP variants
        &ge;&thinsp;1,000&nbsp;bp are retained. Probes outside any truth
        interval are labelled NORMAL.</li>
        <li><strong>Probe labelling:</strong> For each BCF record the probe
        position is classified against pre-loaded, sorted truth intervals via
        binary search (O(log&thinsp;N) per probe). A single BCF pass processes
        all samples simultaneously, keeping I/O at O(records).</li>
        <li><strong>Blacklist annotation:</strong> In the same BCF pass each
        probe position is classified against pre-loaded T2T-CHM13 blacklist
        tracks (centromeres, telomeres, ENCODE-style blacklist). The resulting
        <code>blacklist_region</code> column enables toggling problematic
        regions on/off in the filter panel without re-running the pipeline.</li>
        <li><strong>Density normalisation:</strong> Histograms use
        <em>probability density</em> normalisation so each state integrates to
        1.0, placing the rare DEL/DUP classes on an equal visual footing with
        the dominant NORMAL class regardless of imbalanced probe counts.</li>
        <li><strong>Subsampling:</strong> States with &gt;200,000 probes are
        randomly subsampled for rendering performance; full population counts
        are shown in legend labels and the statistics table.</li>
      </ol>
    </div>

    <div class="about-block">
      <h3>Plot Guide</h3>
      <ul class="about-list">
        <li><strong>LRR &amp; BAF Density Distributions</strong> &mdash;
        <em>Primary figure.</em> Probability density per state for both signal
        channels. Toggle between Density and Count with the button in the
        top-right corner.</li>
        <li><strong>DEL vs DUP Comparison</strong> &mdash; Distributions for
        deletions and duplications only; NORMAL excluded so subtle shape
        differences are visible at the same scale.</li>
        <li><strong>Summary Statistics</strong> &mdash; Mean, median, SD, IQR,
        skewness, kurtosis, and percentiles per state and signal channel.</li>
        <li><strong>LRR vs BAF Scatter</strong> &mdash; 2-D joint distribution
        coloured by copy-number state (sub-sampled for rendering speed).</li>
        <li><strong>Per-Chromosome LRR</strong> &mdash; Box plots per
        chromosome; useful for detecting chromosomal-scale technical
        artefacts.</li>
        <li><strong>Region Size vs Mean LRR</strong> &mdash; Signal strength
        as a function of SV size. Larger SVs typically produce stronger LRR
        deviation.</li>
        <li><strong>Per-Sample Probe Counts</strong> &mdash; Stacked bar chart
        of probe counts by copy-number state for each sample; highlights
        sample-level imbalances in truth-set coverage.</li>
        <li><strong>Blacklist Region Summary</strong> &mdash; Stacked bar
        chart of probe counts per problematic region type (centromere,
        telomere, ENCODE blacklist) broken down by copy-number state, with
        a detailed percentage table. Assesses whether blacklist regions are
        enriched for false-positive DEL/DUP calls.</li>
        <li><strong>Interactive Filter Panel</strong> &mdash; Filter by
        chromosome, sample, minimum region size, LRR range, and blacklist
        region type (include all / exclude blacklist / show only blacklist).
        Density histograms and violin plots regenerate in real time.</li>
      </ul>
    </div>

  </div>
</div>"""


def _blacklist_section_html(
    fig_blacklist: "go.Figure",  # type: ignore[name-defined]
    bl_summary: pd.DataFrame,
    blacklist_dir: Optional[str] = None,
) -> str:
    """Build the Blacklist Region Summary section HTML.

    Parameters
    ----------
    fig_blacklist :
        Plotly Figure with the stacked bar chart of probe counts per region.
    bl_summary : pd.DataFrame
        Output of :func:`build_blacklist_summary` (blacklist_region, state,
        n_probes, pct_of_state).
    blacklist_dir : str, optional
        Path to the blacklist directory; shown as a source reference.
    """
    import plotly.graph_objects as go  # noqa: F401  (ensure available)

    source_note = ""
    if blacklist_dir:
        src_md = os.path.join(blacklist_dir, "SOURCES.md")
        source_note = (
            f' See <code>{os.path.abspath(src_md)}</code>'
            " for full citations and download instructions."
            if os.path.isfile(src_md) else ""
        )

    # Build a per-region breakdown table
    non_none = bl_summary[bl_summary["blacklist_region"] != "(none)"].copy()
    if non_none.empty:
        table_html = "<p class='section-note'>No probes fall in any blacklist region.</p>"
    else:
        rows_html = ""
        for region, grp in non_none.groupby("blacklist_region"):
            for _, row in grp.iterrows():
                state = row["state"]
                rows_html += (
                    f'<tr>'
                    f'<td><code>{region}</code></td>'
                    f'<td><span class="badge-{state}">{state}</span></td>'
                    f'<td>{int(row["n_probes"]):,}</td>'
                    f'<td>{float(row["pct_of_state"]):.2f}%</td>'
                    f'</tr>'
                )
        table_html = f"""
<table class="stats-table" style="margin-top:14px;">
  <thead><tr>
    <th>Blacklist Region</th><th>State</th>
    <th>Probes in region</th><th>% of state total</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>"""

    chart_html = fig_blacklist.to_html(full_html=False, include_plotlyjs=False)

    return f"""
<div class="section">
  <h2>Blacklist Region Summary (T2T-CHM13 v2.0)</h2>
  <p class="section-note">
    Probe counts per known problematic region type, broken down by
    copy-number state. A high proportion of DEL or DUP calls in centromere or
    telomere regions suggests potential false positives driven by repetitive-element
    signal artefacts. Use the <strong>Interactive Filter Panel</strong> below to
    exclude these regions from the density distributions and assess the impact
    on your calls.{source_note}
  </p>
  <p style="font-size:12px;color:#666;margin-bottom:6px;">
    <strong>Sources:</strong>
    Centromeres &mdash; Altemose <em>et al.</em>
    <a href="https://doi.org/10.1126/science.abl4178" target="_blank" rel="noopener noreferrer">
      <em>Science</em> 2022;376:eabl4178</a>;
    Telomeres &amp; assembly &mdash; Nurk <em>et al.</em>
    <a href="https://doi.org/10.1126/science.abj6987" target="_blank" rel="noopener noreferrer">
      <em>Science</em> 2022;376:44&ndash;53</a>;
    ENCODE Blacklist &mdash; Amemiya <em>et al.</em>
    <a href="https://doi.org/10.1038/s41598-019-45839-z" target="_blank" rel="noopener noreferrer">
      <em>Sci Rep</em> 2019;9:9354</a>.
    All regions are for T2T-CHM13 v2.0 (GCA_009914755.4).
  </p>
  {chart_html}
  {table_html}
</div>"""


def _html_header() -> str:
    """Return the HTML head with Plotly CDN and publication-quality styling."""
    return """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Litmus Test – LRR / BAF Probe Assessment</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: Arial, Helvetica, sans-serif;
    margin: 0; padding: 0;
    background: #f2f4f7;
    color: #2c2c2c;
    font-size: 14px;
    line-height: 1.5;
  }
  .container { max-width: 1440px; margin: 0 auto; padding: 28px 24px; }
  .report-header {
    text-align: center;
    background: linear-gradient(135deg, #1a3a5c 0%, #4E79A7 100%);
    color: #fff;
    border-radius: 12px;
    padding: 32px 20px 24px;
    margin-bottom: 28px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }
  .report-header h1 {
    margin: 0 0 8px;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 0.3px;
  }
  .report-header p {
    margin: 0;
    font-size: 14px;
    opacity: 0.88;
  }
  .section {
    background: #ffffff;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 24px 28px;
    margin-bottom: 24px;
    border: 1px solid #e8eaed;
  }
  /* ── About section ── */
  .about-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 20px;
    margin-top: 16px;
  }
  .about-block {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 18px;
  }
  .about-block h3 {
    margin: 0 0 10px;
    font-size: 14px;
    font-weight: 700;
    color: #1a3a5c;
    text-transform: uppercase;
    letter-spacing: 0.4px;
  }
  .about-block p, .about-block li, .about-block dd {
    font-size: 13px;
    color: #3a3a3a;
    line-height: 1.6;
    margin: 0 0 6px;
  }
  .about-list { padding-left: 18px; margin: 0; }
  .about-list li { margin-bottom: 7px; }
  .signal-dl dt {
    font-weight: 700;
    font-size: 13px;
    color: #1a3a5c;
    margin-top: 10px;
  }
  .signal-dl dt:first-child { margin-top: 0; }
  .signal-dl dd { margin: 2px 0 0 0; padding-left: 12px; }
  .sample-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 4px;
  }
  .sample-badge {
    background: #e8f0fb;
    border: 1px solid #b0c4de;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    font-family: monospace;
    color: #1a3a5c;
    font-weight: 600;
  }
  h2 {
    margin-top: 0;
    margin-bottom: 4px;
    color: #1a3a5c;
    font-size: 17px;
    font-weight: 600;
    border-bottom: 2px solid #e8eaed;
    padding-bottom: 10px;
  }
  .section-note {
    color: #666;
    font-size: 13px;
    margin: 4px 0 14px;
    font-style: italic;
  }
  /* ── Filter panel ── */
  .filter-panel {
    background: #eef2f7;
    border: 1px solid #d0d9e6;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 24px;
  }
  .filter-panel h2 { color: #1a3a5c; border-bottom-color: #c5cfe3; }
  .filter-row {
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    align-items: flex-end;
    margin-top: 14px;
  }
  .filter-group { display: flex; flex-direction: column; gap: 4px; }
  .filter-group label {
    font-weight: 600;
    font-size: 12px;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .filter-group select,
  .filter-group input[type="number"] {
    padding: 6px 10px;
    border: 1px solid #b0bcc9;
    border-radius: 5px;
    font-size: 13px;
    background: #fff;
    color: #2c2c2c;
    outline: none;
    transition: border-color 0.2s;
    min-width: 120px;
  }
  .filter-group select:focus,
  .filter-group input[type="number"]:focus { border-color: #4E79A7; }
  .lrr-range { display: flex; gap: 6px; align-items: center; }
  .lrr-range input { min-width: 72px; }
  .lrr-range span { color: #888; font-size: 13px; }
  #apply-btn {
    padding: 8px 22px;
    background: #4E79A7;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
    align-self: flex-end;
  }
  #apply-btn:hover { background: #1a3a5c; }
  #filtered-stats { margin-top: 18px; }
  #filtered-plot  { margin-top: 18px; }
  #filtered-violin { margin-top: 8px; }
  /* Stats table in filter panel */
  .stats-table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 10px;
    font-size: 12px;
  }
  .stats-table th {
    background: #4E79A7;
    color: #fff;
    padding: 7px 10px;
    text-align: center;
    font-weight: 600;
    letter-spacing: 0.3px;
  }
  .stats-table td {
    padding: 5px 10px;
    text-align: center;
    border-bottom: 1px solid #e8eaed;
  }
  .stats-table tr:nth-child(even) td { background: #f5f7fa; }
  .stats-table tr:last-child td { border-bottom: none; }
  .badge-DEL    { color: #fff; background:#E15759; border-radius:4px; padding:1px 6px; }
  .badge-NORMAL { color: #fff; background:#4E79A7; border-radius:4px; padding:1px 6px; }
  .badge-DUP    { color: #fff; background:#F28E2B; border-radius:4px; padding:1px 6px; }
</style>
</head>
<body>
<div class="container">
<div class="report-header">
  <h1>Litmus Test &ndash; LRR / BAF Probe-Level Assessment</h1>
  <p>Quality-control dashboard for array probe signals across copy-number states
     (DEL / NORMAL / DUP) &mdash; generated by
     <a href="https://github.com/jlanej/array_cnv_caller"
        style="color:#aed6f1;" target="_blank" rel="noopener noreferrer">
        array_cnv_caller</a></p>
</div>
</div>
"""


def _filter_panel_html(df: pd.DataFrame) -> str:
    """Build an interactive filtering panel with embedded data + JS.

    The panel lets users choose a chromosome, sample, LRR range and minimum
    region size, then re-renders both density histograms **and violin plots**
    for the filtered data on the fly using Plotly.js.
    """
    import json

    chroms = sorted(df["chrom"].unique().tolist(), key=_chrom_sort_key)
    samples = sorted(df["sample"].unique().tolist())

    # Collect distinct blacklist region types present in the data
    bl_col = df.get("blacklist_region") if hasattr(df, "get") else df["blacklist_region"] if "blacklist_region" in df.columns else None
    bl_regions: list[str] = []
    if bl_col is not None:
        bl_regions = sorted(r for r in bl_col.unique().tolist() if r)

    # Sub-sample for the filter panel to keep HTML size manageable
    cap = min(len(df), 300_000)
    sub = df.sample(n=cap, random_state=42) if len(df) > cap else df

    bl_list = sub["blacklist_region"].tolist() if "blacklist_region" in sub.columns else [""] * len(sub)

    data_json = json.dumps(
        {
            "chrom": sub["chrom"].tolist(),
            "sample": sub["sample"].tolist(),
            "lrr": [round(v, 5) for v in sub["lrr"].tolist()],
            "baf": [round(v, 5) for v in sub["baf"].tolist()],
            "state": sub["state"].tolist(),
            "region_size": sub["region_size"].tolist(),
            "blacklist_region": bl_list,
        }
    )

    chrom_options = "".join(
        f'<option value="{c}">{c}</option>' for c in chroms
    )
    sample_options = "".join(
        f'<option value="{s}">{s}</option>' for s in samples
    )
    bl_region_options = "".join(
        f'<option value="{r}">{r}</option>' for r in bl_regions
    )
    has_blacklist = "true" if bl_regions else "false"

    return f"""
<div class="filter-panel section">
  <h2>&#9881; Interactive Filtering Panel</h2>
  <p class="section-note">
    Apply any combination of filters below and click
    <strong>Apply Filters</strong> to regenerate density histograms,
    violin plots and summary statistics for the matching probe subset.
    Use the <strong>Blacklist regions</strong> filter to exclude or isolate
    probes in problematic T2T-CHM13 regions (centromeres, telomeres, ENCODE
    blacklist) and assess their impact on the signal distributions.
    Violin plots can be further toggled between all states and
    DEL&nbsp;/&nbsp;DUP only.
  </p>
  <div class="filter-row">
    <div class="filter-group">
      <label>Chromosome</label>
      <select id="filt-chrom">
        <option value="ALL" selected>ALL</option>
        {chrom_options}
      </select>
    </div>
    <div class="filter-group">
      <label>Sample</label>
      <select id="filt-sample">
        <option value="ALL" selected>ALL</option>
        {sample_options}
      </select>
    </div>
    <div class="filter-group">
      <label>Min region size (bp)</label>
      <input id="filt-min-size" type="number" value="0" min="0" step="1000"/>
    </div>
    <div class="filter-group">
      <label>LRR range</label>
      <div class="lrr-range">
        <input id="filt-lrr-lo" type="number" value="-5" step="0.1"/>
        <span>&ndash;</span>
        <input id="filt-lrr-hi" type="number" value="3" step="0.1"/>
      </div>
    </div>
    <div class="filter-group">
      <label>Blacklist regions</label>
      <select id="filt-blacklist">
        <option value="all" selected>Include all</option>
        <option value="exclude">Exclude blacklist</option>
        <option value="only">Show only blacklist</option>
        {f'<optgroup label="Single track">{bl_region_options}</optgroup>' if bl_region_options else ''}
      </select>
    </div>
    <div class="filter-group">
      <label>Violin states</label>
      <select id="filt-violin-states">
        <option value="all" selected>DEL + NORMAL + DUP</option>
        <option value="sv">DEL + DUP only</option>
      </select>
    </div>
    <div class="filter-group">
      <label>Histogram mode</label>
      <select id="filt-histnorm">
        <option value="probability density" selected>Density</option>
        <option value="">Count</option>
      </select>
    </div>
    <button id="apply-btn" onclick="applyFilters()">Apply Filters</button>
  </div>
  <div id="filtered-stats"></div>
  <div id="filtered-plot" style="margin-top:20px;"></div>
  <div id="filtered-violin" style="margin-top:6px;"></div>
</div>

<script>
var _DATA = {data_json};
var _COLOURS = {{'DEL':'#E15759','NORMAL':'#4E79A7','DUP':'#F28E2B'}};
var _FONT    = {{family:'Arial, Helvetica, sans-serif', size:13, color:'#2c2c2c'}};
var _AXFONT  = {{family:'Arial, Helvetica, sans-serif', size:12}};
// Allowlist of valid state labels to prevent unintended HTML injection
var _VALID_STATES = {{'DEL':true,'NORMAL':true,'DUP':true}};
var _HAS_BLACKLIST = {has_blacklist};

function _axis(title) {{
  return {{
    title: {{text: title, font: _AXFONT}},
    tickfont: _AXFONT,
    showgrid: true, gridcolor: '#ebebeb',
    zeroline: true, zerolinecolor: '#cccccc'
  }};
}}

function applyFilters() {{
  var chrom     = document.getElementById('filt-chrom').value;
  var sample    = document.getElementById('filt-sample').value;
  var minSz     = parseFloat(document.getElementById('filt-min-size').value) || 0;
  var lrrLo     = parseFloat(document.getElementById('filt-lrr-lo').value);
  var lrrHi     = parseFloat(document.getElementById('filt-lrr-hi').value);
  var blMode    = document.getElementById('filt-blacklist').value;
  var violinSts = document.getElementById('filt-violin-states').value;
  var histNorm  = document.getElementById('filt-histnorm').value;
  var yAxisLabel = histNorm === 'probability density' ? 'Probability Density' : 'Count';
  var blLabel;
  if (blMode === 'exclude')   blLabel = ' (blacklist excluded)';
  else if (blMode === 'only') blLabel = ' (blacklist only)';
  else if (blMode !== 'all')  blLabel = ' (' + blMode + ' only)';
  else                        blLabel = '';
  var histTitle  = histNorm === 'probability density'
    ? 'Filtered LRR &amp; BAF Probability Density by Copy-Number State' + blLabel
    : 'Filtered LRR &amp; BAF Distributions by Copy-Number State' + blLabel;

  var filtered = {{}};
  var n = _DATA.chrom.length;
  for (var i = 0; i < n; i++) {{
    if (chrom  !== 'ALL' && _DATA.chrom[i]  !== chrom)  continue;
    if (sample !== 'ALL' && _DATA.sample[i] !== sample) continue;
    if (_DATA.region_size[i] < minSz && _DATA.state[i] !== 'NORMAL') continue;
    if (_DATA.lrr[i] < lrrLo || _DATA.lrr[i] > lrrHi) continue;
    // Blacklist filter
    if (_HAS_BLACKLIST) {{
      var bl = _DATA.blacklist_region[i] || '';
      if (blMode === 'exclude' && bl !== '') continue;
      if (blMode === 'only'    && bl === '') continue;
      if (blMode !== 'all' && blMode !== 'exclude' && blMode !== 'only' && bl !== blMode) continue;
    }}
    var st = _DATA.state[i];
    if (!_VALID_STATES[st]) continue;  // skip any unexpected state label
    if (!filtered[st]) filtered[st] = {{lrr:[], baf:[]}};
    filtered[st].lrr.push(_DATA.lrr[i]);
    filtered[st].baf.push(_DATA.baf[i]);
  }}

  /* ── Summary table ──────────────────────────────────── */
  var rows = '';
  ['DEL','NORMAL','DUP'].forEach(function(st) {{
    if (!filtered[st]) return;
    var lrr = filtered[st].lrr.slice().sort(function(a,b){{return a-b;}});
    var baf = filtered[st].baf.slice().sort(function(a,b){{return a-b;}});
    rows += '<tr>'
      + '<td><span class="badge-'+st+'">'+st+'</span></td>'
      + '<td>'+lrr.length.toLocaleString()+'</td>'
      + '<td>'+mean(lrr).toFixed(4)+'</td>'
      + '<td>'+median(lrr).toFixed(4)+'</td>'
      + '<td>'+std(lrr).toFixed(4)+'</td>'
      + '<td>'+mean(baf).toFixed(4)+'</td>'
      + '<td>'+median(baf).toFixed(4)+'</td>'
      + '<td>'+std(baf).toFixed(4)+'</td>'
      + '</tr>';
  }});
  document.getElementById('filtered-stats').innerHTML =
    '<table class="stats-table"><thead><tr>'
    + '<th>State</th><th>N</th>'
    + '<th>LRR mean</th><th>LRR median</th><th>LRR std</th>'
    + '<th>BAF mean</th><th>BAF median</th><th>BAF std</th>'
    + '</tr></thead><tbody>' + rows + '</tbody></table>';

  /* ── Density histograms ─────────────────────────────── */
  var histTraces = [];
  ['DEL','NORMAL','DUP'].forEach(function(st) {{
    if (!filtered[st]) return;
    histTraces.push({{
      x: filtered[st].lrr, type:'histogram',
      name: st + ' (n=' + filtered[st].lrr.length.toLocaleString() + ')',
      marker:{{color:_COLOURS[st]}}, opacity:0.65, nbinsx:120,
      histnorm: histNorm, xaxis:'x', yaxis:'y',
      legendgroup: st
    }});
    histTraces.push({{
      x: filtered[st].baf, type:'histogram',
      name: st + ' BAF',
      marker:{{color:_COLOURS[st]}}, opacity:0.65, nbinsx:120,
      histnorm: histNorm, xaxis:'x2', yaxis:'y2',
      legendgroup: st, showlegend: false
    }});
  }});
  var histLayout = {{
    template: 'plotly_white',
    font: _FONT,
    grid: {{rows:1, columns:2, pattern:'independent'}},
    barmode:'overlay', height:480,
    xaxis:  _axis('LRR'),  yaxis:  _axis(yAxisLabel),
    xaxis2: _axis('BAF'),  yaxis2: _axis(yAxisLabel),
    title: {{text: histTitle, font:{{size:15}}}},
    legend: {{bgcolor:'rgba(255,255,255,0.8)', bordercolor:'#ccc', borderwidth:1}},
    margin: {{l:70, r:30, t:60, b:55}}
  }};
  Plotly.react('filtered-plot', histTraces, histLayout);

  /* ── Violin plots ───────────────────────────────────── */
  var stateList = (violinSts === 'sv') ? ['DEL','DUP'] : ['DEL','NORMAL','DUP'];
  var violinTraces = [];
  stateList.forEach(function(st) {{
    if (!filtered[st] || !filtered[st].lrr.length) return;
    violinTraces.push({{
      y: filtered[st].lrr, type:'violin',
      name: st, legendgroup: st,
      side: 'both',
      box: {{visible: true}},
      meanline: {{visible: true}},
      fillcolor: _COLOURS[st],
      line: {{color: _COLOURS[st]}},
      opacity: 0.75,
      points: false,
      xaxis: 'x3', yaxis: 'y3',
      showlegend: true
    }});
    violinTraces.push({{
      y: filtered[st].baf, type:'violin',
      name: st, legendgroup: st,
      side: 'both',
      box: {{visible: true}},
      meanline: {{visible: true}},
      fillcolor: _COLOURS[st],
      line: {{color: _COLOURS[st]}},
      opacity: 0.75,
      points: false,
      xaxis: 'x4', yaxis: 'y4',
      showlegend: false
    }});
  }});
  var violinLayout = {{
    template: 'plotly_white',
    font: _FONT,
    grid: {{rows:1, columns:2, pattern:'independent'}},
    violinmode: 'group',
    height: 520,
    xaxis3: {{title: {{text:'State', font:_AXFONT}}, tickfont:_AXFONT, showgrid:false}},
    yaxis3: _axis('LRR'),
    xaxis4: {{title: {{text:'State', font:_AXFONT}}, tickfont:_AXFONT, showgrid:false}},
    yaxis4: _axis('BAF'),
    title: {{text:'Filtered Violin + Box Plots', font:{{size:15}}}},
    legend: {{bgcolor:'rgba(255,255,255,0.8)', bordercolor:'#ccc', borderwidth:1}},
    margin: {{l:70, r:30, t:60, b:55}},
    annotations: [
      {{text:'LRR', xref:'paper', yref:'paper', x:0.23, y:1.05,
        showarrow:false, font:{{size:13, color:'#555'}}}},
      {{text:'BAF', xref:'paper', yref:'paper', x:0.77, y:1.05,
        showarrow:false, font:{{size:13, color:'#555'}}}}
    ]
  }};
  Plotly.react('filtered-violin', violinTraces, violinLayout);
}}

function mean(arr) {{
  if (!arr.length) return 0;
  var s = 0; for (var i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}}
function median(arr) {{
  if (!arr.length) return 0;
  var m = Math.floor(arr.length / 2);
  return arr.length % 2 ? arr[m] : (arr[m - 1] + arr[m]) / 2;
}}
function std(arr) {{
  if (arr.length < 2) return 0;
  var m = mean(arr), s = 0;
  for (var i = 0; i < arr.length; i++) s += (arr[i] - m) * (arr[i] - m);
  return Math.sqrt(s / (arr.length - 1));
}}
</script>
"""


# ===================================================================
# CLI
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Litmus test: LRR/BAF probe-level assessment across SV states."
    )
    p.add_argument(
        "--bcf",
        required=True,
        help="Multi-sample BCF/VCF with FORMAT/LRR and FORMAT/BAF.",
    )
    p.add_argument(
        "--truth-dir",
        required=True,
        help="Directory of per-sample truth BED files (<sample>.bed).",
    )
    p.add_argument(
        "--output-dir",
        default="litmus_output",
        help="Output directory (default: litmus_output).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap the number of samples processed (default: all matched).",
    )
    p.add_argument(
        "--blacklist-dir",
        default=_REPO_BLACKLIST_DIR,
        help=(
            "Directory of .bed blacklist files for T2T-CHM13 v2.0 problematic "
            "regions (centromeres, telomeres, ENCODE blacklist). "
            "Defaults to resources/blacklists/ within the repository. "
            "Set to '' to disable blacklist annotation."
        ),
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    blacklist_dir: Optional[str] = args.blacklist_dir if args.blacklist_dir else None

    # ── Collect probe data ────────────────────────────────────────────
    LOG.info("Collecting probe data …")
    df = collect_probe_data(
        args.bcf, args.truth_dir,
        max_samples=args.max_samples,
        blacklist_dir=blacklist_dir,
    )

    # ── Write probe-level TSV ─────────────────────────────────────────
    probe_path = os.path.join(args.output_dir, "probe_stats.tsv.gz")
    df.to_csv(probe_path, sep="\t", index=False, compression="gzip")
    LOG.info("Probe-level data written to %s", probe_path)

    # ── Summary statistics ────────────────────────────────────────────
    summary = compute_summary_stats(df)
    summary_path = os.path.join(args.output_dir, "summary_stats.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    LOG.info("Summary statistics written to %s", summary_path)
    print("\n" + summary.to_string(index=False) + "\n")

    # ── Interactive dashboard ─────────────────────────────────────────
    dash_path = os.path.join(args.output_dir, "litmus_report.html")
    build_dashboard(
        df, summary, dash_path,
        bcf_path=args.bcf,
        truth_dir=args.truth_dir,
        blacklist_dir=blacklist_dir,
    )
    LOG.info("All outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
