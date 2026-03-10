#!/usr/bin/env python3
"""
ml_cnv_calling.py – Deep-learning CNV caller for Illumina array data.

Architecture
~~~~~~~~~~~~
A sequence-to-sequence model (``CNVSegmenter``) that combines:

* **1-D CNN** – local smoothing / feature extraction over noisy LRR and BAF
  signals.
* **Bidirectional LSTM** – captures long-range context and identifies copy-
  number state transition boundaries.

The model accepts three input channels per probe:

1. **LRR** (Log R Ratio)
2. **BAF** (B-Allele Frequency)
3. **Distance** – log10-scaled genomic distance to the next probe, making the
   model robust to varying Illumina array probe densities.

It predicts one of three CNV classes for each probe:

    DEL    (class 0) – deletion
    NORMAL (class 1) – normal diploid
    DUP    (class 2) – duplication

Training uses **weighted cross-entropy loss** to counter the heavy class
imbalance toward the NORMAL class.

CLI
~~~
    # Single-sample training
    python scripts/ml_cnv_calling.py train   --bcf in.bcf --truth-bed truth.bed --output model.pt

    # Multi-sample training (auto-matches BCF samples to per-sample BEDs)
    python scripts/ml_cnv_calling.py train   --bcf multi.bcf --truth-dir truth_sets/per_sample/ \\
        --min-probes 5 --overlap-report overlap.tsv --output model.pt

    python scripts/ml_cnv_calling.py predict --bcf in.bcf --model model.pt     --output calls.bed

Truth-set BED files are produced by ``scripts/prepare_truth_set.py`` from the
1000 Genomes ONT Vienna shapeit5-phased SV callset.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pysam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchcrf import CRF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 3  # DEL=0, NORMAL=1, DUP=2
CLASS_DEL = 0
CLASS_NORMAL = 1
CLASS_DUP = 2
CLASS_NAMES = {CLASS_DEL: "DEL", CLASS_NORMAL: "NORMAL", CLASS_DUP: "DUP"}
SVTYPE_TO_CLASS = {"DEL": CLASS_DEL, "DUP": CLASS_DUP}
INPUT_CHANNELS = 3  # LRR, BAF, distance
DEFAULT_WINDOW = 512  # probes per training window
DEFAULT_STRIDE = 256
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 32
DEFAULT_MIN_PROBES = 1

LOG = logging.getLogger(__name__)


# ===================================================================
# Model
# ===================================================================
class CNVSegmenter(nn.Module):
    """1-D CNN + Bi-LSTM sequence-to-sequence model for CNV segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels (default 3: LRR, BAF, distance).
    cnn_channels : int
        Number of output channels for each CNN layer.
    cnn_layers : int
        Number of stacked 1-D convolution layers.
    lstm_hidden : int
        Hidden size of each LSTM direction.
    lstm_layers : int
        Number of stacked LSTM layers.
    num_classes : int
        Number of output classes (DEL, NORMAL, DUP).
    dropout : float
        Dropout probability applied between CNN and LSTM layers.
    use_crf : bool
        When *True* a CRF layer is appended after the fully-connected head.
        During training the loss is the negative log-likelihood computed by
        the CRF; during inference the Viterbi algorithm is used to decode
        the most likely sequence of states.
    """

    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        cnn_channels: int = 64,
        cnn_layers: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
        use_crf: bool = False,
    ) -> None:
        super().__init__()
        self.use_crf = use_crf

        # -- 1-D CNN stack (causal-ish, same-length padding) ----------------
        layers: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(cnn_layers):
            layers.extend(
                [
                    nn.Conv1d(ch_in, cnn_channels, kernel_size=5, padding=2),
                    nn.BatchNorm1d(cnn_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            ch_in = cnn_channels
        self.cnn = nn.Sequential(*layers)

        # -- Bi-LSTM --------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # -- Fully-connected head -------------------------------------------
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

        # -- Optional CRF layer --------------------------------------------
        if use_crf:
            self.crf = CRF(num_classes, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass – returns emission logits.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, channels, seq_len)``.

        Returns
        -------
        Tensor
            Shape ``(batch, seq_len, num_classes)`` – logits (emissions)
            per probe.
        """
        # CNN: (B, C, L) -> (B, cnn_ch, L)
        h = self.cnn(x)
        # Transpose for LSTM: (B, L, cnn_ch)
        h = h.permute(0, 2, 1)
        # LSTM: (B, L, 2*hidden)
        h, _ = self.lstm(h)
        # FC: (B, L, num_classes)
        return self.fc(h)

    # -- CRF helpers --------------------------------------------------------

    def crf_loss(
        self, emissions: torch.Tensor, tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the CRF negative log-likelihood loss.

        Parameters
        ----------
        emissions : Tensor
            Shape ``(batch, seq_len, num_classes)`` – output of ``forward()``.
        tags : Tensor
            Shape ``(batch, seq_len)`` – ground-truth class indices.
        mask : Tensor, optional
            Boolean mask of shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Scalar loss (negated log-likelihood).
        """
        # torchcrf.CRF.forward returns log-likelihood; negate for loss
        return -self.crf(emissions, tags, mask=mask)

    def crf_decode(
        self, emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Viterbi-decode the most likely tag sequence.

        Parameters
        ----------
        emissions : Tensor
            Shape ``(batch, seq_len, num_classes)``.
        mask : Tensor, optional
            Boolean mask of shape ``(batch, seq_len)``.

        Returns
        -------
        list[list[int]]
            Best tag sequence for each item in the batch.
        """
        return self.crf.decode(emissions, mask=mask)


# ===================================================================
# Data helpers
# ===================================================================
def read_bcf_probes(
    bcf_path: str, sample: Optional[str] = None
) -> pd.DataFrame:
    """Read per-probe LRR and BAF from a BCF/VCF file.

    Expects FORMAT fields ``LRR`` and ``BAF`` (as produced by the
    illumina_idat_processing pipeline).

    Parameters
    ----------
    bcf_path : str
        Path to the BCF/VCF file.
    sample : str, optional
        Sample name to extract.  When *None*, the first sample is used.

    Returns
    -------
    pd.DataFrame
        Columns: chrom, pos, lrr, baf – sorted by (chrom, pos).
    """
    vcf = pysam.VariantFile(bcf_path)
    samples = list(vcf.header.samples)
    if not samples:
        raise ValueError(f"No samples found in {bcf_path}")
    sample = sample or samples[0]
    LOG.info("Reading probes for sample %s from %s", sample, bcf_path)

    records: list[dict] = []
    for rec in vcf.fetch():
        fmt = rec.samples[sample]
        lrr = fmt.get("LRR")
        baf = fmt.get("BAF")
        if lrr is None or baf is None:
            continue
        records.append(
            {
                "chrom": rec.chrom,
                "pos": rec.pos,
                "lrr": float(lrr),
                "baf": float(baf),
            }
        )
    vcf.close()

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No probes with LRR/BAF found in the input file.")
    df.sort_values(["chrom", "pos"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_distance_channel(df: pd.DataFrame) -> np.ndarray:
    """Compute log10 inter-probe distance as a feature channel.

    Within each chromosome the distance to the *next* probe is used;
    the last probe of each chromosome receives the median distance for
    that chromosome.  The result is log10-transformed (with a floor of 1
    to avoid log(0)).

    Returns
    -------
    np.ndarray
        1-D array of log10 distances, same length as *df*.
    """
    dist = np.zeros(len(df), dtype=np.float32)
    for chrom, grp in df.groupby("chrom"):
        idx = grp.index.values
        positions = grp["pos"].values
        d = np.diff(positions).astype(np.float32)
        median_d = float(np.median(d)) if len(d) > 0 else 1.0
        d = np.append(d, median_d)
        d = np.clip(d, 1.0, None)
        dist[idx] = np.log10(d)
    return dist


def assign_cn_labels(
    probes: pd.DataFrame, truth_bed: str, min_probes: int = 0
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Label each probe with a class using a truth-set BED file.

    The BED file columns are: chrom  start  end  svtype
    where svtype is ``DEL`` or ``DUP``.

    Probes overlapping a truth region inherit its class (DEL=0 or DUP=2);
    all others are labelled NORMAL (class 1).  Truth regions overlapping
    fewer than *min_probes* array probes are excluded from labelling.

    Returns
    -------
    labels : np.ndarray
        Per-probe class labels (int64): 0=DEL, 1=NORMAL, 2=DUP.
    stats : dict
        ``total_regions``, ``used_regions``, ``skipped_regions``,
        ``probes_labeled`` (count of non-NORMAL probes).
    """
    truth = pd.read_csv(
        truth_bed,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "svtype"],
        dtype={"chrom": str, "start": int, "end": int, "svtype": str},
    )
    labels = np.full(len(probes), CLASS_NORMAL, dtype=np.int64)

    total_regions = 0
    used_regions = 0
    skipped_regions = 0

    for chrom in truth["chrom"].unique():
        t_chrom = truth[truth["chrom"] == chrom]
        p_mask = probes["chrom"] == chrom
        p_pos = probes.loc[p_mask, "pos"].values
        p_idx = probes.loc[p_mask].index.values

        for _, row in t_chrom.iterrows():
            total_regions += 1
            svtype = row["svtype"]
            if svtype not in SVTYPE_TO_CLASS:
                skipped_regions += 1
                continue
            class_label = SVTYPE_TO_CLASS[svtype]
            overlap = (p_pos >= row["start"]) & (p_pos < row["end"])
            n_overlap = int(overlap.sum())
            if n_overlap < min_probes:
                skipped_regions += 1
                continue
            used_regions += 1
            labels[p_idx[overlap]] = class_label

    stats = {
        "total_regions": total_regions,
        "used_regions": used_regions,
        "skipped_regions": skipped_regions,
        "probes_labeled": int((labels != CLASS_NORMAL).sum()),
    }
    return labels, stats


def get_bcf_samples(bcf_path: str) -> List[str]:
    """Return the list of sample names from a BCF/VCF header."""
    vcf = pysam.VariantFile(bcf_path)
    samples = list(vcf.header.samples)
    vcf.close()
    return samples


def match_samples(
    bcf_path: str, truth_dir: str
) -> Tuple[List[str], List[str], List[str]]:
    """Match BCF sample names to per-sample truth BED files.

    Truth BED files are expected as ``<truth_dir>/<sample>.bed``.

    Returns
    -------
    matched : list[str]
        Samples present in both the BCF and the truth directory.
    bcf_only : list[str]
        Samples only in the BCF (no truth data).
    truth_only : list[str]
        Samples only in the truth directory (not in BCF).
    """
    bcf_samples = set(get_bcf_samples(bcf_path))

    truth_samples: set[str] = set()
    for fname in os.listdir(truth_dir):
        if fname.endswith(".bed"):
            truth_samples.add(fname[:-4])

    matched = sorted(bcf_samples & truth_samples)
    bcf_only = sorted(bcf_samples - truth_samples)
    truth_only = sorted(truth_samples - bcf_samples)
    return matched, bcf_only, truth_only


def write_overlap_report(
    report_path: str,
    matched: List[str],
    bcf_only: List[str],
    truth_only: List[str],
) -> None:
    """Write a TSV report of sample overlap between BCF and truth set."""
    rows: list[dict] = []
    for s in matched:
        rows.append(
            {"sample": s, "in_bcf": True, "in_truth": True, "status": "matched"}
        )
    for s in bcf_only:
        rows.append(
            {"sample": s, "in_bcf": True, "in_truth": False, "status": "bcf_only"}
        )
    for s in truth_only:
        rows.append(
            {"sample": s, "in_bcf": False, "in_truth": True, "status": "truth_only"}
        )
    df = pd.DataFrame(rows)
    df.sort_values("sample", inplace=True)
    df.to_csv(report_path, sep="\t", index=False)
    LOG.info("Wrote overlap report (%d samples) to %s", len(df), report_path)


# ===================================================================
# Dataset
# ===================================================================
class ProbeDataset(Dataset):
    """Sliding-window dataset over per-chromosome probe arrays."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window: int = DEFAULT_WINDOW,
        stride: int = DEFAULT_STRIDE,
    ) -> None:
        self.features = features  # (N, 3)
        self.labels = labels  # (N,)
        self.window = window
        self.stride = stride

        self.starts: list[int] = []
        n = len(features)
        pos = 0
        while pos + window <= n:
            self.starts.append(pos)
            pos += stride
        # Include a final partial window if there are remaining probes
        if n > 0 and (not self.starts or self.starts[-1] + window < n):
            self.starts.append(max(0, n - window))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.starts[idx]
        e = s + self.window
        x = self.features[s:e].T  # (3, W)
        y = self.labels[s:e]
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor


# ===================================================================
# Training
# ===================================================================
def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights for weighted cross-entropy."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.clip(counts, 1, None)  # avoid division by zero
    weights = 1.0 / counts
    weights /= weights.sum()
    weights *= NUM_CLASSES
    return torch.tensor(weights, dtype=torch.float32)


def compute_per_class_prf(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """Compute per-class precision/recall/F1 without external deps."""
    metrics: Dict[int, Dict[str, float]] = {}
    for cls in range(NUM_CLASSES):
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        metrics[cls] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics


def train_model(
    bcf_path: str,
    output_path: str,
    truth_bed: Optional[str] = None,
    truth_dir: Optional[str] = None,
    sample: Optional[str] = None,
    min_probes: int = DEFAULT_MIN_PROBES,
    overlap_report: Optional[str] = None,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH,
    device_name: str = "auto",
    use_crf: bool = False,
) -> None:
    """End-to-end training loop.

    Supports two modes:

    * **Single-sample** (``truth_bed``): one sample from the BCF is matched
      to a single truth BED file.
    * **Multi-sample** (``truth_dir``): the BCF sample names are matched
      against per-sample BED files in *truth_dir* (``<sample>.bed``).
      An overlap report is optionally written to *overlap_report*.
    """

    if not truth_bed and not truth_dir:
        raise ValueError("Either --truth-bed or --truth-dir must be provided.")

    # -- Device -------------------------------------------------------------
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    LOG.info("Using device: %s", device)

    # -- Data ---------------------------------------------------------------
    if truth_dir:
        # ── Multi-sample mode ─────────────────────────────────────────────
        matched, bcf_only, truth_only = match_samples(bcf_path, truth_dir)
        LOG.info(
            "Sample overlap: %d matched, %d BCF-only, %d truth-only",
            len(matched),
            len(bcf_only),
            len(truth_only),
        )

        if overlap_report:
            write_overlap_report(overlap_report, matched, bcf_only, truth_only)

        if not matched:
            raise ValueError(
                f"No BCF samples match truth BED files in {truth_dir}"
            )

        all_features: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for i, s in enumerate(matched, 1):
            LOG.info("Loading sample %d/%d: %s", i, len(matched), s)
            bed_path = os.path.join(truth_dir, f"{s}.bed")
            probes = read_bcf_probes(bcf_path, sample=s)
            labels, stats = assign_cn_labels(
                probes, bed_path, min_probes=min_probes
            )

            if stats["probes_labeled"] == 0:
                LOG.warning(
                    "  %s: 0 labeled probes after filtering "
                    "(min_probes=%d), skipping.",
                    s,
                    min_probes,
                )
                continue

            distances = compute_distance_channel(probes)
            features = np.column_stack(
                [probes["lrr"].values, probes["baf"].values, distances]
            )
            all_features.append(features)
            all_labels.append(labels)
            LOG.info(
                "  %s: %d probes, %d/%d truth regions used, %d probes labeled",
                s,
                len(probes),
                stats["used_regions"],
                stats["total_regions"],
                stats["probes_labeled"],
            )

        if not all_features:
            raise ValueError(
                "No samples with usable data after filtering."
            )

        # Train / val split by sample
        n_samp = len(all_features)
        if n_samp == 1:
            train_idx, val_idx = [0], [0]
        else:
            rng = np.random.RandomState(42)
            perm = rng.permutation(n_samp).tolist()
            n_val = max(1, n_samp // 10)
            train_idx = perm[:-n_val]
            val_idx = perm[-n_val:]

        train_datasets = [
            ProbeDataset(all_features[i], all_labels[i], window, stride)
            for i in train_idx
        ]
        val_datasets = [
            ProbeDataset(all_features[i], all_labels[i], window, stride)
            for i in val_idx
        ]
        train_ds = ConcatDataset(train_datasets)
        val_ds = ConcatDataset(val_datasets)

        all_train_labels = np.concatenate(
            [all_labels[i] for i in train_idx]
        )
        LOG.info(
            "Multi-sample: %d train samples (%d windows), "
            "%d val samples (%d windows)",
            len(train_idx),
            len(train_ds),
            len(val_idx),
            len(val_ds),
        )
    else:
        # ── Single-sample mode ────────────────────────────────────────────
        probes = read_bcf_probes(bcf_path, sample=sample)
        labels, stats = assign_cn_labels(
            probes, truth_bed, min_probes=min_probes
        )
        distances = compute_distance_channel(probes)

        features = np.column_stack(
            [probes["lrr"].values, probes["baf"].values, distances]
        )

        cn_values, cn_counts = np.unique(labels, return_counts=True)
        class_dist = {CLASS_NAMES.get(v, v): c for v, c in zip(cn_values, cn_counts)}
        LOG.info(
            "Loaded %d probes – class distribution: %s", len(probes), class_dist
        )
        LOG.info(
            "Truth regions: %d used, %d skipped (min_probes=%d)",
            stats["used_regions"],
            stats["skipped_regions"],
            min_probes,
        )

        n = len(features)
        split = int(0.9 * n)
        train_ds = ProbeDataset(
            features[:split], labels[:split], window, stride
        )
        val_ds = ProbeDataset(
            features[split:], labels[split:], window, stride
        )
        all_train_labels = labels[:split]

    # -- DataLoaders --------------------------------------------------------
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # -- Model & optimiser --------------------------------------------------
    model = CNVSegmenter(use_crf=use_crf).to(device)
    class_weights = compute_class_weights(all_train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_state: dict = {}

    for epoch in range(1, epochs + 1):
        # -- Train ----------------------------------------------------------
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)  # (B, W, C)
            if use_crf:
                loss = model.crf_loss(logits, yb)
            else:
                loss = criterion(logits.reshape(-1, NUM_CLASSES), yb.reshape(-1))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        # -- Validate -------------------------------------------------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        y_true_parts: list[np.ndarray] = []
        y_pred_parts: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                if use_crf:
                    loss = model.crf_loss(logits, yb)
                    decoded = model.crf_decode(logits)
                    preds = torch.tensor(
                        decoded, dtype=torch.long, device=device,
                    )
                else:
                    loss = criterion(
                        logits.reshape(-1, NUM_CLASSES), yb.reshape(-1)
                    )
                    preds = logits.argmax(dim=-1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
                y_true_parts.append(yb.cpu().reshape(-1).numpy())
                y_pred_parts.append(preds.cpu().reshape(-1).numpy())
        val_loss /= max(len(val_ds), 1)
        val_acc = correct / max(total, 1)
        y_true = np.concatenate(y_true_parts) if y_true_parts else np.array([], dtype=np.int64)
        y_pred = np.concatenate(y_pred_parts) if y_pred_parts else np.array([], dtype=np.int64)
        per_class = compute_per_class_prf(y_true, y_pred)

        scheduler.step(val_loss)
        LOG.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f  "
            "DEL(P/R/F1)=%.3f/%.3f/%.3f  DUP(P/R/F1)=%.3f/%.3f/%.3f",
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_acc,
            per_class[CLASS_DEL]["precision"],
            per_class[CLASS_DEL]["recall"],
            per_class[CLASS_DEL]["f1"],
            per_class[CLASS_DUP]["precision"],
            per_class[CLASS_DUP]["recall"],
            per_class[CLASS_DUP]["f1"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    # -- Save ---------------------------------------------------------------
    torch.save(best_state, output_path)
    LOG.info("Saved best model to %s (val_loss=%.4f)", output_path, best_val_loss)


# ===================================================================
# Prediction
# ===================================================================
def predict_cnv(
    bcf_path: str,
    model_path: str,
    output_path: str,
    sample: Optional[str] = None,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    min_confidence: float = 0.5,
    device_name: str = "auto",
    use_crf: bool = False,
) -> None:
    """Run sliding-window inference and write a BED file of CNV calls."""

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    # -- Load model ---------------------------------------------------------
    model = CNVSegmenter(use_crf=use_crf).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # -- Load probes --------------------------------------------------------
    probes = read_bcf_probes(bcf_path, sample=sample)
    distances = compute_distance_channel(probes)
    features = np.column_stack(
        [probes["lrr"].values, probes["baf"].values, distances]
    )  # (N, 3)

    n = len(features)
    vote_sums = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    vote_observations = np.zeros(n, dtype=np.float64)

    # -- Sliding window inference -------------------------------------------
    with torch.no_grad():
        pos = 0
        while pos < n:
            end = min(pos + window, n)
            chunk = features[pos:end]
            valid = end - pos
            if len(chunk) < window:
                # Pad the last chunk
                pad = np.zeros((window - len(chunk), INPUT_CHANNELS), dtype=np.float32)
                chunk = np.vstack([chunk, pad])
            x = torch.tensor(chunk.T, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(x)  # (1, W, C)
            if use_crf:
                # Build a mask so the CRF only scores valid (non-padded)
                # positions, then Viterbi-decode.
                mask = torch.zeros(1, window, dtype=torch.bool, device=device)
                mask[0, :valid] = True
                decoded = model.crf_decode(logits, mask=mask)
                tags = decoded[0][:valid]
                # Use one-hot confidence for CRF decoded tags
                for k, tag in enumerate(tags):
                    vote_sums[pos + k, tag] += 1.0
                vote_observations[pos:end] += 1.0
            else:
                probs = F.softmax(logits[0], dim=-1).cpu().numpy()
                vote_sums[pos:end] += probs[:valid]
                vote_observations[pos:end] += 1.0
            pos += stride

    mean_probs = vote_sums / np.maximum(vote_observations[:, None], 1.0)
    predictions = mean_probs.argmax(axis=1)
    prediction_confidence = mean_probs.max(axis=1)

    # -- Collapse adjacent probes into CNV segments -------------------------
    segments: list[dict] = []
    i = 0
    while i < n:
        cls = int(predictions[i])
        if cls == CLASS_NORMAL or prediction_confidence[i] < min_confidence:
            i += 1
            continue
        chrom = probes.iloc[i]["chrom"]
        start = int(probes.iloc[i]["pos"])
        j = i + 1
        while (
            j < n
            and int(predictions[j]) == cls
            and probes.iloc[j]["chrom"] == chrom
            and prediction_confidence[j] >= min_confidence
        ):
            j += 1
        end = int(probes.iloc[j - 1]["pos"])
        num_probes = j - i
        segments.append(
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "svtype": CLASS_NAMES[cls],
                "num_probes": num_probes,
            }
        )
        i = j

    # -- Write BED ----------------------------------------------------------
    seg_df = pd.DataFrame(segments)
    if seg_df.empty:
        LOG.info("No CNV segments predicted.")
        seg_df = pd.DataFrame(
            columns=["chrom", "start", "end", "svtype", "num_probes"]
        )
    seg_df.to_csv(output_path, sep="\t", index=False, header=False)
    LOG.info(
        "Wrote %d CNV segments (%d non-diploid probes) to %s",
        len(seg_df),
        int(((predictions != CLASS_NORMAL) & (prediction_confidence >= min_confidence)).sum()),
        output_path,
    )


# ===================================================================
# CLI
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml_cnv_calling",
        description="Deep-learning CNV caller for Illumina array data "
        "(1-D CNN + Bi-LSTM).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train --------------------------------------------------------------
    p_train = sub.add_parser(
        "train",
        help="Train the CNVSegmenter model.",
    )
    p_train.add_argument(
        "--bcf",
        required=True,
        help="Input BCF/VCF with FORMAT/LRR and FORMAT/BAF.",
    )
    truth_grp = p_train.add_mutually_exclusive_group(required=True)
    truth_grp.add_argument(
        "--truth-bed",
        help="Single truth-set BED file (chrom start end svtype).  "
        "Use with --sample for single-sample training.",
    )
    truth_grp.add_argument(
        "--truth-dir",
        help="Directory of per-sample truth BED files (<sample>.bed).  "
        "Enables multi-sample training by auto-matching BCF samples.",
    )
    p_train.add_argument(
        "--output", "-o", default="cnv_model.pt", help="Output model path."
    )
    p_train.add_argument(
        "--sample",
        default=None,
        help="Sample name in BCF (single-sample mode only).",
    )
    p_train.add_argument(
        "--min-probes",
        type=int,
        default=DEFAULT_MIN_PROBES,
        help="Minimum number of array probes overlapping a truth region "
        "for it to be included in training (default: %(default)s).",
    )
    p_train.add_argument(
        "--overlap-report",
        default=None,
        help="Path to write a TSV sample-overlap report "
        "(multi-sample mode only).",
    )
    p_train.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW, help="Window size."
    )
    p_train.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE, help="Window stride."
    )
    p_train.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs."
    )
    p_train.add_argument(
        "--lr", type=float, default=DEFAULT_LR, help="Learning rate."
    )
    p_train.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH, help="Batch size."
    )
    p_train.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cpu, cuda, cuda:0, …).",
    )
    p_train.add_argument(
        "--use-crf",
        action="store_true",
        default=False,
        help="Append a CRF layer after the Bi-LSTM for transition-aware "
        "segmentation (uses Viterbi decoding at inference).",
    )

    # -- predict ------------------------------------------------------------
    p_pred = sub.add_parser(
        "predict",
        help="Predict CNVs using a trained model.",
    )
    p_pred.add_argument(
        "--bcf",
        required=True,
        help="Input BCF/VCF with FORMAT/LRR and FORMAT/BAF.",
    )
    p_pred.add_argument(
        "--model",
        required=True,
        help="Trained model weights (.pt file).",
    )
    p_pred.add_argument(
        "--output", "-o", default="cnv_calls.bed", help="Output BED path."
    )
    p_pred.add_argument("--sample", default=None, help="Sample name in BCF.")
    p_pred.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW, help="Window size."
    )
    p_pred.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE, help="Window stride."
    )
    p_pred.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum softmax confidence for non-normal probes to be included "
        "in CNV segments (default: %(default)s).",
    )
    p_pred.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cpu, cuda, cuda:0, …).",
    )
    p_pred.add_argument(
        "--use-crf",
        action="store_true",
        default=False,
        help="Load a model with CRF layer and use Viterbi decoding.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        train_model(
            bcf_path=args.bcf,
            output_path=args.output,
            truth_bed=args.truth_bed,
            truth_dir=args.truth_dir,
            sample=args.sample,
            min_probes=args.min_probes,
            overlap_report=args.overlap_report,
            window=args.window,
            stride=args.stride,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device_name=args.device,
            use_crf=args.use_crf,
        )
    elif args.command == "predict":
        predict_cnv(
            bcf_path=args.bcf,
            model_path=args.model,
            output_path=args.output,
            sample=args.sample,
            window=args.window,
            stride=args.stride,
            min_confidence=args.min_confidence,
            device_name=args.device,
            use_crf=args.use_crf,
        )


if __name__ == "__main__":
    main()
