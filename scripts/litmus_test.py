#!/usr/bin/env python3
"""
litmus_test.py – LRR / BAF probe-level quality assessment across SV states.

Reads a multi-sample BCF (with FORMAT/LRR and FORMAT/BAF) together with
per-sample truth-set BED files, labels every probe as DEL / NORMAL / DUP,
and produces:

1. **probe_stats.tsv.gz**  – probe-level table (sample, chrom, pos, lrr,
   baf, state, region_size) for downstream analyses.
2. **summary_stats.tsv**   – per-state aggregate statistics (mean, median,
   std, IQR, skewness, kurtosis, N) for LRR and BAF.
3. **litmus_report.html**  – self-contained interactive Plotly dashboard
   with histograms, violin/box-plots, 2-D scatter, and per-chromosome
   break-downs.  Drop-down menus and sliders allow real-time filtering by
   chromosome, sample, SV-region size, and LRR/BAF ranges.

CLI
~~~
    python scripts/litmus_test.py \\
        --bcf multi.bcf \\
        --truth-dir truth_sets/per_sample/ \\
        --output-dir litmus_output/

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
# Statistics
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

    Returns
    -------
    pd.DataFrame
        Columns: sample, chrom, pos, lrr, baf, state, region_size.
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
    samples_seen: set[str] = set()

    n_records = 0
    for rec in vcf.fetch():
        chrom = rec.chrom
        pos = rec.pos
        n_records += 1
        if n_records % 100_000 == 0:
            LOG.info("  … %d BCF records scanned", n_records)

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
) -> None:
    """Generate a self-contained interactive HTML dashboard.

    The dashboard includes:
    * LRR and BAF histograms per state with overlaid densities
    * Violin + box-plots for LRR and BAF
    * 2-D LRR×BAF scatter coloured by state (sub-sampled for speed)
    * Per-chromosome LRR distributions
    * Summary statistics table
    * Drop-down menus for chromosome and sample filtering
    * Range sliders for region-size filtering

    Parameters
    ----------
    df : pd.DataFrame
        Probe-level table (sample, chrom, pos, lrr, baf, state, region_size).
    summary : pd.DataFrame
        Aggregate statistics from :func:`compute_summary_stats`.
    output_path : str
        File path for the HTML report.
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

    STATE_COLOURS = {"DEL": "#d62728", "NORMAL": "#2ca02c", "DUP": "#1f77b4"}
    STATES = ["DEL", "NORMAL", "DUP"]

    # -- Utility: subsample for large datasets -------------------------
    def _subsample(data: pd.DataFrame, n: int = 50_000) -> pd.DataFrame:
        if len(data) <= n:
            return data
        return data.sample(n=n, random_state=42)

    # ── 1. Histograms (LRR & BAF) ────────────────────────────────────
    fig_hist = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("LRR by State", "BAF by State"),
        horizontal_spacing=0.08,
    )
    for state in STATES:
        sub = df[df["state"] == state]
        fig_hist.add_trace(
            go.Histogram(
                x=sub["lrr"],
                name=f"{state} LRR",
                marker_color=STATE_COLOURS[state],
                opacity=0.6,
                nbinsx=200,
                legendgroup=state,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig_hist.add_trace(
            go.Histogram(
                x=sub["baf"],
                name=f"{state} BAF",
                marker_color=STATE_COLOURS[state],
                opacity=0.6,
                nbinsx=200,
                legendgroup=state,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig_hist.update_layout(
        barmode="overlay",
        title_text="LRR & BAF Distributions by Copy-Number State",
        height=500,
    )
    fig_hist.update_xaxes(title_text="LRR", row=1, col=1)
    fig_hist.update_xaxes(title_text="BAF", row=1, col=2)
    fig_hist.update_yaxes(title_text="Count", row=1, col=1)
    fig_hist.update_yaxes(title_text="Count", row=1, col=2)

    # ── 2. Violin / box-plots ────────────────────────────────────────
    fig_violin = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("LRR by State", "BAF by State"),
        horizontal_spacing=0.08,
    )
    for state in STATES:
        sub = _subsample(df[df["state"] == state])
        fig_violin.add_trace(
            go.Violin(
                y=sub["lrr"],
                name=state,
                box_visible=True,
                meanline_visible=True,
                fillcolor=STATE_COLOURS[state],
                line_color=STATE_COLOURS[state],
                opacity=0.7,
                legendgroup=state,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig_violin.add_trace(
            go.Violin(
                y=sub["baf"],
                name=state,
                box_visible=True,
                meanline_visible=True,
                fillcolor=STATE_COLOURS[state],
                line_color=STATE_COLOURS[state],
                opacity=0.7,
                legendgroup=state,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig_violin.update_layout(
        title_text="LRR & BAF Violin + Box Plots by State",
        height=500,
    )
    fig_violin.update_yaxes(title_text="LRR", row=1, col=1)
    fig_violin.update_yaxes(title_text="BAF", row=1, col=2)

    # ── 3. 2-D scatter (LRR vs BAF) ──────────────────────────────────
    fig_scatter = go.Figure()
    for state in STATES:
        sub = _subsample(df[df["state"] == state], n=20_000)
        fig_scatter.add_trace(
            go.Scattergl(
                x=sub["lrr"],
                y=sub["baf"],
                mode="markers",
                name=state,
                marker=dict(
                    color=STATE_COLOURS[state],
                    size=3,
                    opacity=0.4,
                ),
            )
        )
    fig_scatter.update_layout(
        title_text="LRR vs BAF (sub-sampled, coloured by state)",
        xaxis_title="LRR",
        yaxis_title="BAF",
        height=600,
    )

    # ── 4. Per-chromosome LRR box-plots ──────────────────────────────
    chroms = sorted(df["chrom"].unique(), key=_chrom_sort_key)
    fig_chrom = go.Figure()
    for state in STATES:
        sub = df[df["state"] == state]
        fig_chrom.add_trace(
            go.Box(
                x=sub["chrom"],
                y=sub["lrr"],
                name=state,
                marker_color=STATE_COLOURS[state],
                boxmean="sd",
            )
        )
    fig_chrom.update_layout(
        title_text="LRR Distribution per Chromosome by State",
        xaxis_title="Chromosome",
        yaxis_title="LRR",
        boxmode="group",
        height=500,
        xaxis=dict(categoryorder="array", categoryarray=chroms),
    )

    # ── 5. Region size vs mean LRR (for DEL/DUP only) ────────────────
    fig_size = go.Figure()
    for state in ["DEL", "DUP"]:
        sub = df[(df["state"] == state) & (df["region_size"] > 0)]
        if sub.empty:
            continue
        agg = (
            sub.groupby("region_size")
            .agg(mean_lrr=("lrr", "mean"), mean_baf=("baf", "mean"), n=("lrr", "size"))
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
                    size=np.clip(np.log2(agg["n"].values + 1) * 2, 3, 15),
                    opacity=0.6,
                ),
                text=[f"n={n}" for n in agg["n"]],
            )
        )
    fig_size.update_layout(
        title_text="Truth Region Size vs Mean LRR (DEL / DUP)",
        xaxis_title="Region size (bp)",
        yaxis_title="Mean LRR",
        xaxis_type="log",
        height=500,
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
                )
            )
    fig_samples.update_layout(
        barmode="stack",
        title_text="Probe Counts per Sample by State",
        xaxis_title="Sample",
        yaxis_title="Probe count",
        height=500,
    )

    # ── 7. Summary statistics table ──────────────────────────────────
    fmt_cols = [c for c in summary.columns if c not in ("state", "metric")]
    header_vals = ["State", "Metric"] + [c.upper() for c in fmt_cols]
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
                header=dict(values=header_vals, align="center"),
                cells=dict(values=cell_vals, align="center"),
            )
        ]
    )
    fig_table.update_layout(
        title_text="Summary Statistics by State",
        height=350,
    )

    # ── Assemble HTML ─────────────────────────────────────────────────
    html_parts: list[str] = []
    html_parts.append(_html_header())
    html_parts.append('<div class="container">')

    sections = [
        ("Summary Statistics", fig_table),
        ("LRR &amp; BAF Histograms", fig_hist),
        ("Violin &amp; Box Plots", fig_violin),
        ("LRR vs BAF Scatter", fig_scatter),
        ("Per-Chromosome LRR", fig_chrom),
        ("Region Size vs Mean LRR", fig_size),
        ("Per-Sample Probe Counts", fig_samples),
    ]

    for title, fig in sections:
        html_parts.append(f'<div class="section"><h2>{title}</h2>')
        html_parts.append(
            fig.to_html(full_html=False, include_plotlyjs=False)
        )
        html_parts.append("</div>")

    # ── Interactive filtering panel (JavaScript) ─────────────────────
    html_parts.append(_filter_panel_html(df))

    html_parts.append("</div>")  # container
    html_parts.append("</body></html>")

    with open(output_path, "w") as fh:
        fh.write("\n".join(html_parts))
    LOG.info("Dashboard written to %s", output_path)


def _html_header() -> str:
    """Return the HTML head with Plotly CDN and basic styling."""
    return """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Litmus Test – LRR / BAF Probe Assessment</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: Arial, Helvetica, sans-serif; margin: 0; padding: 0;
         background: #fafafa; color: #333; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
  h1 { text-align: center; margin-bottom: 8px; }
  .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
  .section { background: #fff; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.12);
             padding: 20px; margin-bottom: 24px; }
  h2 { margin-top: 0; color: #444; border-bottom: 2px solid #eee; padding-bottom: 8px; }
  .filter-panel { background: #f0f4f8; border-radius: 8px; padding: 20px;
                  margin-bottom: 24px; }
  .filter-panel label { font-weight: bold; margin-right: 10px; }
  .filter-panel select, .filter-panel input { margin-right: 20px; padding: 4px 8px; }
  #filtered-stats { margin-top: 16px; }
  #filtered-plot { margin-top: 16px; }
</style>
</head>
<body>
<div class="container">
<h1>Litmus Test – LRR / BAF Probe-Level Assessment</h1>
<p class="subtitle">Interactive quality dashboard for array probe signals
   across copy-number states (DEL / NORMAL / DUP)</p>
</div>
"""


def _filter_panel_html(df: pd.DataFrame) -> str:
    """Build an interactive filtering panel with embedded data + JS.

    The panel lets users choose a chromosome and an LRR range, then
    re-renders overlaid histograms for the filtered data on the fly.
    """
    # Encode lightweight per-state arrays in JSON for JS consumption.
    import json

    chroms = sorted(df["chrom"].unique().tolist(), key=_chrom_sort_key)
    samples = sorted(df["sample"].unique().tolist())

    # Sub-sample for the filter panel to keep HTML size manageable
    cap = min(len(df), 300_000)
    sub = df.sample(n=cap, random_state=42) if len(df) > cap else df

    data_json = json.dumps(
        {
            "chrom": sub["chrom"].tolist(),
            "sample": sub["sample"].tolist(),
            "lrr": [round(v, 5) for v in sub["lrr"].tolist()],
            "baf": [round(v, 5) for v in sub["baf"].tolist()],
            "state": sub["state"].tolist(),
            "region_size": sub["region_size"].tolist(),
        }
    )

    chrom_options = "".join(
        f'<option value="{c}">{c}</option>' for c in chroms
    )
    sample_options = "".join(
        f'<option value="{s}">{s}</option>' for s in samples
    )

    return f"""
<div class="section filter-panel">
  <h2>Interactive Filtering</h2>
  <p>Select filters and click <strong>Apply</strong> to regenerate the
     histograms below using only the matching probes.</p>
  <div>
    <label>Chromosome:
      <select id="filt-chrom">
        <option value="ALL" selected>ALL</option>
        {chrom_options}
      </select>
    </label>
    <label>Sample:
      <select id="filt-sample">
        <option value="ALL" selected>ALL</option>
        {sample_options}
      </select>
    </label>
    <label>Min region size (bp):
      <input id="filt-min-size" type="number" value="0" min="0" step="1000"/>
    </label>
    <label>LRR range:
      <input id="filt-lrr-lo" type="number" value="-5" step="0.1" style="width:60px"/>
      –
      <input id="filt-lrr-hi" type="number" value="3" step="0.1" style="width:60px"/>
    </label>
    <button onclick="applyFilters()">Apply</button>
  </div>
  <div id="filtered-stats"></div>
  <div id="filtered-plot"></div>
</div>

<script>
var _DATA = {data_json};

function applyFilters() {{
  var chrom  = document.getElementById('filt-chrom').value;
  var sample = document.getElementById('filt-sample').value;
  var minSz  = parseFloat(document.getElementById('filt-min-size').value) || 0;
  var lrrLo  = parseFloat(document.getElementById('filt-lrr-lo').value);
  var lrrHi  = parseFloat(document.getElementById('filt-lrr-hi').value);

  var filtered = {{}};  // state -> {{lrr:[], baf:[]}}
  var counts   = {{}};
  var n = _DATA.chrom.length;
  for (var i = 0; i < n; i++) {{
    if (chrom !== 'ALL' && _DATA.chrom[i] !== chrom) continue;
    if (sample !== 'ALL' && _DATA.sample[i] !== sample) continue;
    if (_DATA.region_size[i] < minSz && _DATA.state[i] !== 'NORMAL') continue;
    if (_DATA.lrr[i] < lrrLo || _DATA.lrr[i] > lrrHi) continue;
    var st = _DATA.state[i];
    if (!filtered[st]) {{ filtered[st] = {{lrr:[], baf:[]}}; counts[st] = 0; }}
    filtered[st].lrr.push(_DATA.lrr[i]);
    filtered[st].baf.push(_DATA.baf[i]);
    counts[st]++;
  }}

  // Summary text
  var html = '<table border="1" cellpadding="4" style="border-collapse:collapse; margin-top:8px;">';
  html += '<tr><th>State</th><th>N</th><th>LRR mean</th><th>LRR median</th><th>LRR std</th>';
  html += '<th>BAF mean</th><th>BAF median</th><th>BAF std</th></tr>';
  ['DEL','NORMAL','DUP'].forEach(function(st) {{
    if (!filtered[st]) return;
    var lrr = filtered[st].lrr.slice().sort(function(a,b){{return a-b;}});
    var baf = filtered[st].baf.slice().sort(function(a,b){{return a-b;}});
    var lrrM = mean(lrr), lrrMed = median(lrr), lrrS = std(lrr);
    var bafM = mean(baf), bafMed = median(baf), bafS = std(baf);
    html += '<tr><td>'+st+'</td><td>'+lrr.length+'</td>';
    html += '<td>'+lrrM.toFixed(4)+'</td><td>'+lrrMed.toFixed(4)+'</td><td>'+lrrS.toFixed(4)+'</td>';
    html += '<td>'+bafM.toFixed(4)+'</td><td>'+bafMed.toFixed(4)+'</td><td>'+bafS.toFixed(4)+'</td></tr>';
  }});
  html += '</table>';
  document.getElementById('filtered-stats').innerHTML = html;

  // Plotly traces
  var traces = [];
  var colours = {{'DEL':'#d62728','NORMAL':'#2ca02c','DUP':'#1f77b4'}};
  ['DEL','NORMAL','DUP'].forEach(function(st) {{
    if (!filtered[st]) return;
    traces.push({{
      x: filtered[st].lrr, type:'histogram', name: st+' LRR',
      marker:{{color:colours[st]}}, opacity:0.6, nbinsx:150, xaxis:'x', yaxis:'y'
    }});
    traces.push({{
      x: filtered[st].baf, type:'histogram', name: st+' BAF',
      marker:{{color:colours[st]}}, opacity:0.6, nbinsx:150, xaxis:'x2', yaxis:'y2'
    }});
  }});
  var layout = {{
    grid: {{rows:1, columns:2, pattern:'independent'}},
    barmode:'overlay', height:450,
    xaxis: {{title:'LRR'}}, yaxis: {{title:'Count'}},
    xaxis2:{{title:'BAF'}}, yaxis2:{{title:'Count'}},
    title:'Filtered LRR & BAF Histograms'
  }};
  Plotly.newPlot('filtered-plot', traces, layout);
}}

function mean(arr) {{
  if (!arr.length) return 0;
  var s = 0; for (var i=0;i<arr.length;i++) s+=arr[i]; return s/arr.length;
}}
function median(arr) {{
  if (!arr.length) return 0;
  var m = Math.floor(arr.length/2);
  return arr.length%2 ? arr[m] : (arr[m-1]+arr[m])/2;
}}
function std(arr) {{
  if (arr.length<2) return 0;
  var m = mean(arr), s = 0;
  for (var i=0;i<arr.length;i++) s+=(arr[i]-m)*(arr[i]-m);
  return Math.sqrt(s/(arr.length-1));
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
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Collect probe data ────────────────────────────────────────────
    LOG.info("Collecting probe data …")
    df = collect_probe_data(
        args.bcf, args.truth_dir, max_samples=args.max_samples
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
    build_dashboard(df, summary, dash_path)
    LOG.info("All outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
