#!/usr/bin/env bash
# run_pipeline.sh – Batteries-included training pipeline for array_cnv_caller.
#
# This script orchestrates the full workflow:
#   1. Prepare per-sample BED truth files from the shapeit5-phased SV VCF
#   2. Train the CNVSegmenter model on a multi-sample BCF
#   3. (Optional) Predict CNVs for every sample in the BCF
#
# All prerequisite scripts and resources are pre-packed in the container
# image published to ghcr.io/jlanej/array_cnv_caller.  On HPC clusters,
# pull the image once with Apptainer and then run the pipeline:
#
#   apptainer pull docker://ghcr.io/jlanej/array_cnv_caller:main
#
#   apptainer exec --nv --bind /scratch:/scratch array_cnv_caller_main.sif \
#       bash /app/scripts/run_pipeline.sh \
#       --bcf /scratch/1000g_multisample.bcf \
#       --outdir /scratch/cnv_output
#
# The script auto-detects whether it is running inside the container
# (/app layout) or from a local repository checkout.
#
# Required inputs:
#   --bcf       Multi-sample BCF with FORMAT/LRR and FORMAT/BAF
#               (e.g. the 2141-sample BCF from illumina_idat_processing)
#
# Optional inputs:
#   --truth-vcf Path to the shapeit5-phased SV VCF (default: bundled
#               resources/shapeit5-phased-callset_final-vcf.phased.vcf.gz)
#   --outdir    Output directory (default: pipeline_output)
#   --epochs    Training epochs (default: 30)
#   --batch-size Batch size (default: 32)
#   --lr        Learning rate (default: 0.001)
#   --min-probes  Minimum probes overlapping a truth region (default: 5)
#   --device    Device selection (default: auto)
#   --predict   Also run prediction on every BCF sample after training

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────
BCF=""
TRUTH_VCF=""
OUTDIR="pipeline_output"
EPOCHS=30
BATCH_SIZE=32
LR=0.001
MIN_PROBES=5
DEVICE="auto"
PREDICT=0
LITMUS=0
LITMUS_MAX_SAMPLES=""

# ── Parse arguments ──────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") --bcf <multisample.bcf> [OPTIONS]

Required:
  --bcf PATH          Multi-sample BCF with FORMAT/LRR and FORMAT/BAF

Optional:
  --truth-vcf PATH    Shapeit5-phased SV VCF (default: bundled in image/repo)
  --outdir DIR        Output directory (default: pipeline_output)
  --epochs N          Training epochs (default: 30)
  --batch-size N      Batch size (default: 32)
  --lr FLOAT          Learning rate (default: 0.001)
  --min-probes N      Min array probes per truth region (default: 5)
  --device DEV        Device: auto, cpu, cuda, cuda:0 (default: auto)
  --predict           Run prediction on every BCF sample after training
  --litmus            Run litmus test (LRR/BAF probe assessment dashboard)
  --litmus-max-samples N  Cap samples for litmus test (default: all matched)
  -h, --help          Show this help message
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bcf)        BCF="$2";        shift 2 ;;
        --truth-vcf)  TRUTH_VCF="$2";  shift 2 ;;
        --outdir)     OUTDIR="$2";     shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr)         LR="$2";         shift 2 ;;
        --min-probes) MIN_PROBES="$2"; shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --predict)    PREDICT=1;       shift ;;
        --litmus)     LITMUS=1;        shift ;;
        --litmus-max-samples) LITMUS_MAX_SAMPLES="$2"; shift 2 ;;
        -h|--help)    usage 0 ;;
        *)            echo "Error: unknown option: $1" >&2; usage 1 ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────────
if [[ -z "$BCF" ]]; then
    echo "Error: --bcf is required." >&2
    usage 1
fi

if [[ ! -f "$BCF" ]]; then
    echo "Error: BCF file not found: $BCF" >&2
    exit 1
fi

# ── Detect app root (/app inside container, repo root otherwise) ─────────
# When running inside the Docker/Apptainer image, scripts and resources
# live under /app.  Outside the container, we resolve relative to the
# repository checkout that contains this script.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -d /app/scripts && -d /app/resources ]]; then
    APP_ROOT="/app"
else
    APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# ── Resolve truth VCF ────────────────────────────────────────────────────
if [[ -z "$TRUTH_VCF" ]]; then
    TRUTH_VCF="$APP_ROOT/resources/shapeit5-phased-callset_final-vcf.phased.vcf.gz"
fi

if [[ ! -f "$TRUTH_VCF" ]]; then
    echo "Error: Truth VCF not found at default location: $TRUTH_VCF" >&2
    echo "       Use --truth-vcf to specify a custom path." >&2
    exit 1
fi

# ── Output paths ─────────────────────────────────────────────────────────
mkdir -p "$OUTDIR"
TRUTH_DIR="$OUTDIR/truth_sets"
MODEL_PATH="$OUTDIR/cnv_model.pt"
OVERLAP_REPORT="$OUTDIR/overlap_report.tsv"
PREDICTIONS_DIR="$OUTDIR/predictions"
LITMUS_DIR="$OUTDIR/litmus"

TOTAL_STEPS=$((2 + (PREDICT == 1) + (LITMUS == 1)))

echo "============================================================"
echo "  array_cnv_caller – Training Pipeline"
echo "============================================================"
echo ""
echo "  BCF:            $BCF"
echo "  Truth VCF:      $TRUTH_VCF"
echo "  Output dir:     $OUTDIR"
echo "  App root:       $APP_ROOT"
echo "  Device:         $DEVICE"
echo "  Epochs:         $EPOCHS"
echo "  Batch size:     $BATCH_SIZE"
echo "  Learning rate:  $LR"
echo "  Min probes:     $MIN_PROBES"
echo "  Predict:        $([ "$PREDICT" -eq 1 ] && echo 'yes' || echo 'no')"
echo "  Litmus test:    $([ "$LITMUS" -eq 1 ] && echo 'yes' || echo 'no')"
echo ""

# ── Step 1: Prepare truth-set BED files ──────────────────────────────────
echo "────────────────────────────────────────────────────────────"
echo "  Step 1/$TOTAL_STEPS: Prepare truth-set BED files"
echo "────────────────────────────────────────────────────────────"

if [[ -d "$TRUTH_DIR/per_sample" ]] && ls "$TRUTH_DIR/per_sample/"*.bed >/dev/null 2>&1; then
    N_BED=$(ls "$TRUTH_DIR/per_sample/"*.bed | wc -l)
    echo "  → Found $N_BED existing BED files in $TRUTH_DIR/per_sample/; skipping."
else
    python "$APP_ROOT/scripts/prepare_truth_set.py" \
        --vcf "$TRUTH_VCF" \
        --output-dir "$TRUTH_DIR" \
        --min-size 1000
fi

echo ""

# ── Step 2: Train model ─────────────────────────────────────────────────
echo "────────────────────────────────────────────────────────────"
echo "  Step 2/$TOTAL_STEPS: Train CNVSegmenter model"
echo "────────────────────────────────────────────────────────────"

python "$APP_ROOT/scripts/ml_cnv_calling.py" train \
    --bcf "$BCF" \
    --truth-dir "$TRUTH_DIR/per_sample/" \
    --min-probes "$MIN_PROBES" \
    --overlap-report "$OVERLAP_REPORT" \
    --output "$MODEL_PATH" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device "$DEVICE"

echo ""
echo "  → Model saved to: $MODEL_PATH"
echo "  → Overlap report: $OVERLAP_REPORT"
echo ""

# ── Step 3 (optional): Predict CNVs ─────────────────────────────────────
STEP_NUM=3
if [[ "$PREDICT" -eq 1 ]]; then
    echo "────────────────────────────────────────────────────────────"
    echo "  Step $STEP_NUM/$TOTAL_STEPS: Predict CNVs for BCF samples"
    echo "────────────────────────────────────────────────────────────"

    mkdir -p "$PREDICTIONS_DIR"

    python "$APP_ROOT/scripts/ml_cnv_calling.py" predict \
        --bcf "$BCF" \
        --model "$MODEL_PATH" \
        --output "$PREDICTIONS_DIR/cnv_calls.bed" \
        --device "$DEVICE"

    echo ""
    echo "  → Predictions saved to: $PREDICTIONS_DIR/"
    echo ""
    STEP_NUM=$((STEP_NUM + 1))
fi

# ── Step N (optional): Litmus test – LRR/BAF probe assessment ────────────
if [[ "$LITMUS" -eq 1 ]]; then
    echo "────────────────────────────────────────────────────────────"
    echo "  Step $STEP_NUM/$TOTAL_STEPS: Litmus test (LRR/BAF probe dashboard)"
    echo "────────────────────────────────────────────────────────────"

    LITMUS_ARGS=(
        --bcf "$BCF"
        --truth-dir "$TRUTH_DIR/per_sample/"
        --output-dir "$LITMUS_DIR"
    )
    if [[ -n "$LITMUS_MAX_SAMPLES" ]]; then
        LITMUS_ARGS+=(--max-samples "$LITMUS_MAX_SAMPLES")
    fi

    python "$APP_ROOT/scripts/litmus_test.py" "${LITMUS_ARGS[@]}"

    echo ""
    echo "  → Litmus report: $LITMUS_DIR/litmus_report.html"
    echo "  → Probe stats:   $LITMUS_DIR/probe_stats.tsv.gz"
    echo "  → Summary stats: $LITMUS_DIR/summary_stats.tsv"
    echo ""
fi

echo "============================================================"
echo "  Pipeline complete."
echo "============================================================"
