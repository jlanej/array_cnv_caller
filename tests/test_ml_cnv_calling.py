"""Unit tests for scripts/ml_cnv_calling.py."""

from __future__ import annotations

import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import torch

# Ensure the scripts directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))

from ml_cnv_calling import (
    CLASS_DEL,
    CLASS_DUP,
    CLASS_NAMES,
    CLASS_NORMAL,
    CNVSegmenter,
    DEFAULT_STRIDE,
    DEFAULT_WINDOW,
    INPUT_CHANNELS,
    NUM_CLASSES,
    ProbeDataset,
    assign_cn_labels,
    build_parser,
    compute_class_weights,
    compute_per_class_prf,
    compute_distance_channel,
)


# ===================================================================
# CNVSegmenter model tests
# ===================================================================
class TestCNVSegmenter:
    """Tests for the CNN + Bi-LSTM model."""

    def test_output_shape(self):
        model = CNVSegmenter()
        batch, seq_len = 4, 128
        x = torch.randn(batch, INPUT_CHANNELS, seq_len)
        out = model(x)
        assert out.shape == (batch, seq_len, NUM_CLASSES)

    def test_single_sample(self):
        model = CNVSegmenter()
        x = torch.randn(1, INPUT_CHANNELS, 64)
        out = model(x)
        assert out.shape == (1, 64, NUM_CLASSES)

    def test_default_window_size(self):
        model = CNVSegmenter()
        x = torch.randn(2, INPUT_CHANNELS, DEFAULT_WINDOW)
        out = model(x)
        assert out.shape == (2, DEFAULT_WINDOW, NUM_CLASSES)

    def test_custom_architecture(self):
        model = CNVSegmenter(
            cnn_channels=32,
            cnn_layers=2,
            lstm_hidden=64,
            lstm_layers=1,
            dropout=0.1,
        )
        x = torch.randn(1, INPUT_CHANNELS, 100)
        out = model(x)
        assert out.shape == (1, 100, NUM_CLASSES)

    def test_gradient_flow(self):
        """Verify gradients flow through the entire model."""
        model = CNVSegmenter()
        x = torch.randn(2, INPUT_CHANNELS, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_eval_mode_deterministic(self):
        """Model in eval mode should produce deterministic outputs."""
        model = CNVSegmenter()
        model.eval()
        x = torch.randn(1, INPUT_CHANNELS, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_parameter_count(self):
        """Model should have a reasonable number of parameters."""
        model = CNVSegmenter()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 1000  # Not trivially small
        assert n_params < 10_000_000  # Not unreasonably large


# ===================================================================
# Data helper tests
# ===================================================================
class TestComputeDistanceChannel:
    """Tests for compute_distance_channel()."""

    def test_basic_distances(self):
        df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr1", "chr1"],
                "pos": [100, 200, 300, 500],
            }
        )
        dist = compute_distance_channel(df)
        assert len(dist) == 4
        # Distances: 100, 100, 200, median(100,100,200)=100
        expected = np.log10([100.0, 100.0, 200.0, 100.0])
        np.testing.assert_allclose(dist, expected, atol=1e-5)

    def test_multi_chromosome(self):
        df = pd.DataFrame(
            {
                "chrom": ["chr1", "chr1", "chr2", "chr2"],
                "pos": [100, 200, 1000, 2000],
            }
        )
        dist = compute_distance_channel(df)
        assert len(dist) == 4
        # chr1: d=[100], median=100, last gets median -> [100, 100]
        # chr2: d=[1000], median=1000, last gets median -> [1000, 1000]

    def test_single_probe(self):
        df = pd.DataFrame({"chrom": ["chr1"], "pos": [1000]})
        dist = compute_distance_channel(df)
        assert len(dist) == 1
        # Single probe: diff is empty, median defaults to 1.0
        assert dist[0] == pytest.approx(np.log10(1.0), abs=1e-5)

    def test_floor_at_one(self):
        """Distance should be floored at 1 before log10."""
        df = pd.DataFrame(
            {"chrom": ["chr1", "chr1"], "pos": [100, 100]}  # zero distance
        )
        dist = compute_distance_channel(df)
        # log10(1) = 0
        assert dist[0] == pytest.approx(0.0, abs=1e-5)


class TestAssignCnLabels:
    """Tests for assign_cn_labels()."""

    def test_basic_labeling(self, tmp_path):
        probes = pd.DataFrame(
            {
                "chrom": ["chr1"] * 10,
                "pos": list(range(1000, 11000, 1000)),
            }
        )
        bed_path = tmp_path / "truth.bed"
        bed_path.write_text("chr1\t2500\t5500\tDEL\n")

        labels, stats = assign_cn_labels(probes, str(bed_path))

        # Probes at 3000, 4000, 5000 should be DEL (within 2500-5500)
        assert labels[2] == CLASS_DEL  # pos=3000
        assert labels[3] == CLASS_DEL  # pos=4000
        assert labels[4] == CLASS_DEL  # pos=5000
        assert labels[0] == CLASS_NORMAL  # pos=1000
        assert labels[9] == CLASS_NORMAL  # pos=10000

    def test_dup_labeling(self, tmp_path):
        probes = pd.DataFrame(
            {
                "chrom": ["chr1"] * 5,
                "pos": [1000, 2000, 3000, 4000, 5000],
            }
        )
        bed_path = tmp_path / "truth.bed"
        bed_path.write_text("chr1\t1500\t3500\tDUP\n")

        labels, stats = assign_cn_labels(probes, str(bed_path))
        assert labels[1] == CLASS_DUP  # pos=2000
        assert labels[2] == CLASS_DUP  # pos=3000
        assert labels[0] == CLASS_NORMAL
        assert labels[3] == CLASS_NORMAL

    def test_min_probes_filter(self, tmp_path):
        probes = pd.DataFrame(
            {
                "chrom": ["chr1"] * 5,
                "pos": [1000, 2000, 3000, 4000, 5000],
            }
        )
        bed_path = tmp_path / "truth.bed"
        # Region only overlaps 1 probe (pos=2000)
        bed_path.write_text("chr1\t1500\t2500\tDEL\n")

        # With min_probes=2, this region should be skipped
        labels, stats = assign_cn_labels(probes, str(bed_path), min_probes=2)
        assert all(l == CLASS_NORMAL for l in labels)
        assert stats["skipped_regions"] == 1

    def test_stats_output(self, tmp_path):
        probes = pd.DataFrame(
            {
                "chrom": ["chr1"] * 5,
                "pos": [1000, 2000, 3000, 4000, 5000],
            }
        )
        bed_path = tmp_path / "truth.bed"
        bed_path.write_text("chr1\t999\t5001\tDEL\n")

        labels, stats = assign_cn_labels(probes, str(bed_path))
        assert stats["total_regions"] == 1
        assert stats["used_regions"] == 1
        assert stats["skipped_regions"] == 0
        assert stats["probes_labeled"] == 5

    def test_empty_bed(self, tmp_path):
        probes = pd.DataFrame(
            {
                "chrom": ["chr1"] * 3,
                "pos": [100, 200, 300],
            }
        )
        bed_path = tmp_path / "empty.bed"
        bed_path.write_text("")

        labels, stats = assign_cn_labels(probes, str(bed_path))
        assert all(l == CLASS_NORMAL for l in labels)

    def test_bed_interval_is_half_open(self, tmp_path):
        probes = pd.DataFrame(
            {
                "chrom": ["chr1"] * 5,
                "pos": [1000, 2000, 3000, 4000, 5000],
            }
        )
        bed_path = tmp_path / "truth_half_open.bed"
        bed_path.write_text("chr1\t2000\t4000\tDEL\n")

        labels, _ = assign_cn_labels(probes, str(bed_path))
        assert labels[1] == CLASS_DEL  # start is included
        assert labels[2] == CLASS_DEL
        assert labels[3] == CLASS_NORMAL  # end is excluded


class TestComputeClassWeights:
    """Tests for compute_class_weights()."""

    def test_balanced(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        weights = compute_class_weights(labels)
        assert weights.shape == (NUM_CLASSES,)
        # Balanced classes should produce equal weights
        assert torch.allclose(weights[0], weights[1], atol=1e-5)
        assert torch.allclose(weights[1], weights[2], atol=1e-5)

    def test_imbalanced(self):
        # Heavy class imbalance toward class 1 (NORMAL)
        labels = np.array([1] * 1000 + [0] * 5 + [2] * 5)
        weights = compute_class_weights(labels)
        # Rare classes should have higher weights
        assert weights[CLASS_DEL] > weights[CLASS_NORMAL]
        assert weights[CLASS_DUP] > weights[CLASS_NORMAL]

    def test_missing_class(self):
        labels = np.array([0, 0, 1, 1])  # No class 2
        weights = compute_class_weights(labels)
        assert weights.shape == (NUM_CLASSES,)
        # Class 2 has count=0 -> clipped to 1, gets highest weight
        assert weights[2] > weights[0]


class TestComputePerClassPrf:
    """Tests for compute_per_class_prf()."""

    def test_balanced_predictions(self):
        y_true = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        y_pred = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        metrics = compute_per_class_prf(y_true, y_pred)
        for cls in (CLASS_DEL, CLASS_NORMAL, CLASS_DUP):
            assert metrics[cls]["precision"] == pytest.approx(1.0)
            assert metrics[cls]["recall"] == pytest.approx(1.0)
            assert metrics[cls]["f1"] == pytest.approx(1.0)

    def test_handles_missing_predictions(self):
        y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        y_pred = np.array([1, 1, 1, 1, 1, 1], dtype=np.int64)
        metrics = compute_per_class_prf(y_true, y_pred)
        assert metrics[CLASS_DEL]["precision"] == pytest.approx(0.0)
        assert metrics[CLASS_DEL]["recall"] == pytest.approx(0.0)
        assert metrics[CLASS_DUP]["precision"] == pytest.approx(0.0)
        assert metrics[CLASS_DUP]["recall"] == pytest.approx(0.0)
        assert metrics[CLASS_NORMAL]["precision"] == pytest.approx(2 / 6)
        assert metrics[CLASS_NORMAL]["recall"] == pytest.approx(1.0)


# ===================================================================
# ProbeDataset tests
# ===================================================================
class TestProbeDataset:
    """Tests for the ProbeDataset sliding-window dataset."""

    def test_basic_windowing(self):
        n = 1024
        features = np.random.randn(n, INPUT_CHANNELS).astype(np.float32)
        labels = np.ones(n, dtype=np.int64)
        ds = ProbeDataset(features, labels, window=512, stride=256)

        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (INPUT_CHANNELS, 512)
        assert y.shape == (512,)

    def test_small_input(self):
        n = 100
        features = np.random.randn(n, INPUT_CHANNELS).astype(np.float32)
        labels = np.ones(n, dtype=np.int64)
        ds = ProbeDataset(features, labels, window=512, stride=256)

        # Input smaller than window should still produce at least one item
        assert len(ds) >= 1

    def test_window_stride(self):
        n = 2048
        window, stride = 512, 256
        features = np.random.randn(n, INPUT_CHANNELS).astype(np.float32)
        labels = np.ones(n, dtype=np.int64)
        ds = ProbeDataset(features, labels, window=window, stride=stride)

        # Number of windows: ceil((n - window) / stride) + 1
        expected_windows = ((n - window) // stride) + 1
        assert len(ds) >= expected_windows

    def test_tensor_dtypes(self):
        features = np.random.randn(512, INPUT_CHANNELS).astype(np.float32)
        labels = np.ones(512, dtype=np.int64)
        ds = ProbeDataset(features, labels, window=512, stride=256)

        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.long


# ===================================================================
# CLI parser tests
# ===================================================================
class TestCLIParser:
    """Tests for the argparse CLI setup."""

    def test_train_command(self):
        parser = build_parser()
        args = parser.parse_args(
            ["train", "--bcf", "test.bcf", "--truth-bed", "truth.bed"]
        )
        assert args.command == "train"
        assert args.bcf == "test.bcf"
        assert args.truth_bed == "truth.bed"

    def test_train_multisample(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "train",
                "--bcf",
                "test.bcf",
                "--truth-dir",
                "truth_dir/",
                "--min-probes",
                "5",
                "--overlap-report",
                "overlap.tsv",
            ]
        )
        assert args.truth_dir == "truth_dir/"
        assert args.min_probes == 5
        assert args.overlap_report == "overlap.tsv"

    def test_predict_command(self):
        parser = build_parser()
        args = parser.parse_args(
            ["predict", "--bcf", "test.bcf", "--model", "model.pt"]
        )
        assert args.command == "predict"
        assert args.bcf == "test.bcf"
        assert args.model == "model.pt"
        assert args.min_confidence == 0.5

    def test_predict_command_min_confidence(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "predict",
                "--bcf",
                "test.bcf",
                "--model",
                "model.pt",
                "--min-confidence",
                "0.7",
            ]
        )
        assert args.min_confidence == pytest.approx(0.7)

    def test_truth_bed_and_dir_mutually_exclusive(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "train",
                    "--bcf",
                    "test.bcf",
                    "--truth-bed",
                    "t.bed",
                    "--truth-dir",
                    "d/",
                ]
            )

    def test_default_values(self):
        parser = build_parser()
        args = parser.parse_args(
            ["train", "--bcf", "test.bcf", "--truth-bed", "truth.bed"]
        )
        assert args.epochs == 30
        assert args.lr == 1e-3
        assert args.batch_size == 32
        assert args.window == 512
        assert args.stride == 256
        assert args.device == "auto"
        assert args.output == "cnv_model.pt"


# ===================================================================
# Constants tests
# ===================================================================
class TestConstants:
    """Tests that constants are consistent."""

    def test_class_names_coverage(self):
        for cls_id in range(NUM_CLASSES):
            assert cls_id in CLASS_NAMES

    def test_class_ids(self):
        assert CLASS_DEL == 0
        assert CLASS_NORMAL == 1
        assert CLASS_DUP == 2

    def test_input_channels(self):
        assert INPUT_CHANNELS == 3  # LRR, BAF, distance
