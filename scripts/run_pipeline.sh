#!/usr/bin/env bash
# run_pipeline.sh – Batteries-included training pipeline for array_cnv_caller.
#
# This script orchestrates the full workflow:
#   1. Prepare per-sample BED truth files from the shapeit5-phased SV VCF
#   2. Train the CNVSegmenter model on a multi-sample BCF
#   3. (Optional) Predict CNVs for every sample in the BCF
#
# Designed for HPC environments using Apptainer (formerly Singularity).
#
# Usage:
#   # Apptainer (recommended for HPC)
#   bash scripts/run_pipeline.sh \
#       --bcf /path/to/1000g_multisample.bcf \
#       --sif array_cnv_caller.sif
#
#   # Native (no container)
#   bash scripts/run_pipeline.sh \
#       --bcf /path/to/1000g_multisample.bcf \
#       --native
#
# Required inputs:
#   --bcf       Multi-sample BCF with FORMAT/LRR and FORMAT/BAF
#               (e.g. the 2141-sample BCF from illumina_idat_processing)
#
# Optional inputs:
#   --truth-vcf Path to the shapeit5-phased SV VCF (default: bundled in
#               resources/shapeit5-phased-callset_final-vcf.phased.vcf.gz)
#   --sif       Path to the Apptainer SIF image
#   --native    Run without a container (requires local Python environment)
#   --outdir    Output directory (default: pipeline_output)
#   --epochs    Training epochs (default: 30)
#   --batch-size Batch size (default: 32)
#   --lr        Learning rate (default: 0.001)
#   --min-probes  Minimum probes overlapping a truth region (default: 5)
#   --device    Device selection (default: auto)
#   --predict   Also run prediction on every BCF sample after training
#   --bind      Extra Apptainer bind mounts (e.g. /scratch:/scratch)

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────
BCF=""
TRUTH_VCF=""
SIF=""
NATIVE=0
OUTDIR="pipeline_output"
EPOCHS=30
BATCH_SIZE=32
LR=0.001
MIN_PROBES=5
DEVICE="auto"
PREDICT=0
BIND_MOUNTS=""

# ── Parse arguments ──────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") --bcf <multisample.bcf> [--sif <image.sif> | --native] [OPTIONS]

Required:
  --bcf PATH          Multi-sample BCF with FORMAT/LRR and FORMAT/BAF

Execution mode (one required):
  --sif PATH          Apptainer SIF image
  --native            Run without a container

Optional:
  --truth-vcf PATH    Shapeit5-phased SV VCF (default: bundled in container/repo)
  --outdir DIR        Output directory (default: pipeline_output)
  --epochs N          Training epochs (default: 30)
  --batch-size N      Batch size (default: 32)
  --lr FLOAT          Learning rate (default: 0.001)
  --min-probes N      Min array probes per truth region (default: 5)
  --device DEV        Device: auto, cpu, cuda, cuda:0 (default: auto)
  --predict           Run prediction on every BCF sample after training
  --bind MOUNTS       Extra Apptainer bind mounts (e.g. /scratch:/scratch)
  -h, --help          Show this help message
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bcf)        BCF="$2";        shift 2 ;;
        --truth-vcf)  TRUTH_VCF="$2";  shift 2 ;;
        --sif)        SIF="$2";        shift 2 ;;
        --native)     NATIVE=1;        shift ;;
        --outdir)     OUTDIR="$2";     shift 2 ;;
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --lr)         LR="$2";         shift 2 ;;
        --min-probes) MIN_PROBES="$2"; shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --predict)    PREDICT=1;       shift ;;
        --bind)       BIND_MOUNTS="$2"; shift 2 ;;
        -h|--help)    usage 0 ;;
        *)            echo "Error: unknown option: $1" >&2; usage 1 ;;
    esac
done

# ── Validate ─────────────────────────────────────────────────────────────
if [[ -z "$BCF" ]]; then
    echo "Error: --bcf is required." >&2
    usage 1
fi

if [[ -z "$SIF" && "$NATIVE" -eq 0 ]]; then
    echo "Error: specify either --sif <image.sif> or --native." >&2
    usage 1
fi

if [[ -n "$SIF" && ! -f "$SIF" ]]; then
    echo "Error: SIF image not found: $SIF" >&2
    exit 1
fi

if [[ ! -f "$BCF" ]]; then
    echo "Error: BCF file not found: $BCF" >&2
    exit 1
fi

# ── Helper: build the run command ────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

run_cmd() {
    # Executes a Python script, via Apptainer or natively.
    local script="$1"; shift

    if [[ "$NATIVE" -eq 1 ]]; then
        python "$REPO_ROOT/$script" "$@"
    else
        local bind_args=()
        # Always bind the BCF directory so the container can read it
        local bcf_dir
        bcf_dir="$(cd "$(dirname "$BCF")" && pwd)"
        bind_args+=("--bind" "${bcf_dir}:${bcf_dir}")

        # Bind the output directory
        local abs_outdir
        abs_outdir="$(cd "$OUTDIR" 2>/dev/null && pwd || mkdir -p "$OUTDIR" && cd "$OUTDIR" && pwd)"
        bind_args+=("--bind" "${abs_outdir}:${abs_outdir}")

        # User-requested extra bind mounts
        if [[ -n "$BIND_MOUNTS" ]]; then
            bind_args+=("--bind" "$BIND_MOUNTS")
        fi

        # Use --nv for GPU passthrough when device is not cpu
        local nv_flag=""
        if [[ "$DEVICE" != "cpu" ]]; then
            nv_flag="--nv"
        fi

        apptainer run "${bind_args[@]}" $nv_flag "$SIF" \
            "/app/$script" "$@"
    fi
}

# ── Resolve truth VCF ────────────────────────────────────────────────────
if [[ -z "$TRUTH_VCF" ]]; then
    if [[ "$NATIVE" -eq 1 ]]; then
        TRUTH_VCF="$REPO_ROOT/resources/shapeit5-phased-callset_final-vcf.phased.vcf.gz"
    else
        # The container bundles the truth VCF in /app/resources/
        TRUTH_VCF="/app/resources/shapeit5-phased-callset_final-vcf.phased.vcf.gz"
    fi
fi

# ── Output paths ─────────────────────────────────────────────────────────
mkdir -p "$OUTDIR"
TRUTH_DIR="$OUTDIR/truth_sets"
MODEL_PATH="$OUTDIR/cnv_model.pt"
OVERLAP_REPORT="$OUTDIR/overlap_report.tsv"
PREDICTIONS_DIR="$OUTDIR/predictions"

TOTAL_STEPS=$([ "$PREDICT" -eq 1 ] && echo '3' || echo '2')

echo "============================================================"
echo "  array_cnv_caller – Training Pipeline"
echo "============================================================"
echo ""
echo "  BCF:            $BCF"
echo "  Truth VCF:      $TRUTH_VCF"
echo "  Output dir:     $OUTDIR"
echo "  Mode:           $([ "$NATIVE" -eq 1 ] && echo 'native' || echo "apptainer ($SIF)")"
echo "  Device:         $DEVICE"
echo "  Epochs:         $EPOCHS"
echo "  Batch size:     $BATCH_SIZE"
echo "  Learning rate:  $LR"
echo "  Min probes:     $MIN_PROBES"
echo "  Predict:        $([ "$PREDICT" -eq 1 ] && echo 'yes' || echo 'no')"
echo ""

# ── Step 1: Prepare truth-set BED files ──────────────────────────────────
echo "────────────────────────────────────────────────────────────"
echo "  Step 1/$TOTAL_STEPS: Prepare truth-set BED files"
echo "────────────────────────────────────────────────────────────"

if [[ -d "$TRUTH_DIR/per_sample" ]] && ls "$TRUTH_DIR/per_sample/"*.bed >/dev/null 2>&1; then
    N_BED=$(ls "$TRUTH_DIR/per_sample/"*.bed | wc -l)
    echo "  → Found $N_BED existing BED files in $TRUTH_DIR/per_sample/; skipping."
else
    run_cmd scripts/prepare_truth_set.py \
        --vcf "$TRUTH_VCF" \
        --output-dir "$TRUTH_DIR" \
        --min-size 1000
fi

echo ""

# ── Step 2: Train model ─────────────────────────────────────────────────
echo "────────────────────────────────────────────────────────────"
echo "  Step 2/$TOTAL_STEPS: Train CNVSegmenter model"
echo "────────────────────────────────────────────────────────────"

run_cmd scripts/ml_cnv_calling.py train \
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
if [[ "$PREDICT" -eq 1 ]]; then
    echo "────────────────────────────────────────────────────────────"
    echo "  Step 3/$TOTAL_STEPS: Predict CNVs for BCF samples"
    echo "────────────────────────────────────────────────────────────"

    mkdir -p "$PREDICTIONS_DIR"

    run_cmd scripts/ml_cnv_calling.py predict \
        --bcf "$BCF" \
        --model "$MODEL_PATH" \
        --output "$PREDICTIONS_DIR/cnv_calls.bed" \
        --device "$DEVICE"

    echo ""
    echo "  → Predictions saved to: $PREDICTIONS_DIR/"
    echo ""
fi

echo "============================================================"
echo "  Pipeline complete."
echo "============================================================"
