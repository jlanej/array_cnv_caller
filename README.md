# array_cnv_caller

CNV caller for Illumina array data, trained on matched 1000 Genomes array
and long-read sequencing resources.

## Overview

This project implements a deep-learning CNV (Copy Number Variant) caller that
operates on BAF (B-Allele Frequency) and LRR (Log R Ratio) signals produced by
Illumina genotyping arrays.  Training labels are derived from the highly curated
1000 Genomes ONT Vienna structural variant truth set – a population-scale
catalogue of 167,000+ sequence-resolved SVs from Oxford Nanopore long-read
sequencing of 1,019 samples ([data collection][1kgp_ont]).

Array data are prepared with the
[illumina_idat_processing](https://github.com/jlanej/illumina_idat_processing)
pipeline (stage 2 reclustered VCF/BCF with `FORMAT/LRR` and `FORMAT/BAF`),
processed across all samples via
[`process_1000g.sh`](https://github.com/jlanej/illumina_idat_processing/blob/main/scripts/process_1000g.sh).

The curated training set is sourced from:

* [PMC12350158](https://pmc.ncbi.nlm.nih.gov/articles/PMC12350158/) – curated
  structural variant truth sets.
* [1KG ONT Vienna FTP][1kgp_ont] – population-scale long-read SV calls from
  the 1000 Genomes Project.
* [shapeit5-phased callset][phased_vcf] – the phased, sequence-resolved SV VCF
  used as the primary training truth set, included in `resources/`.

[1kgp_ont]: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1KG_ONT_VIENNA/
[phased_vcf]: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1KG_ONT_VIENNA/release/v1.1/shapeit5-phased-callset/shapeit5-phased-callset_final-vcf.phased.vcf.gz

---

## Experimental Machine Learning CNV Calling

This repository includes a **state-of-the-art PyTorch 1-D CNN + Bidirectional
LSTM** model designed to replace traditional Hidden Markov Model (HMM)
approaches for array-based CNV calling.

### Architecture

The `CNVSegmenter` model is a sequence-to-sequence architecture:

| Stage | Description |
|-------|-------------|
| **1-D CNN** | Stacked convolutional layers extract local features and smooth noisy LRR/BAF signals. |
| **Bi-LSTM** | Bidirectional LSTM layers capture long-range genomic context and sharpen copy-number transition boundaries. |
| **FC head** | A fully connected layer predicts one of 3 CNV classes (DEL, NORMAL, DUP) per probe. |

The model accepts **three input channels** per probe:

1. **LRR** – Log R Ratio
2. **BAF** – B-Allele Frequency
3. **Distance** – log₁₀-scaled distance to the next probe

Encoding inter-probe distance explicitly makes the model **robust to different
Illumina array probe densities** (e.g. Omni2.5 vs. GSA vs. Mega).

Training uses **weighted cross-entropy loss** to handle the extreme class
imbalance inherent in array data (the vast majority of probes are NORMAL).

### Truth Set Preparation

Training labels come from the 1000 Genomes ONT Vienna shapeit5-phased SV
callset.  The phased VCF is included in `resources/` and the Python helper
script parses it into per-sample BED files:

```bash
python scripts/prepare_truth_set.py \
    --vcf resources/shapeit5-phased-callset_final-vcf.phased.vcf.gz \
    --output-dir truth_sets \
    --min-size 1000
```

The script:

* Derives SV type (DEL/DUP) from the variant ID field (sequence-resolved
  variants have no INFO/SVTYPE).
* Filters to DEL and DUP variants ≥ 1 kb.
* Writes per-sample BED files and a comprehensive summary of SV types
  included/excluded plus global metrics for comparison with the publication.

Output BED format:  `chrom  start  end  svtype`

Class mapping:

| SV type | Class | Label |
|---------|-------|-------|
| DEL     | 0     | Deletion |
| —       | 1     | Normal diploid |
| DUP     | 2     | Duplication |

### Installation

```bash
pip install -r requirements_ml.txt
```

### Usage

#### Single-sample training

```bash
python scripts/ml_cnv_calling.py train \
    --bcf sample.bcf \
    --truth-bed truth_sets/per_sample/SAMPLE.bed \
    --output cnv_model.pt \
    --epochs 30 \
    --device auto
```

#### Multi-sample training

When training on a multi-sample BCF (e.g. the ~2,141-sample BCF produced by
[`process_1000g.sh`](https://github.com/jlanej/illumina_idat_processing/blob/main/scripts/process_1000g.sh)),
use `--truth-dir` to point at the directory of per-sample BED files created by
`prepare_truth_set.py`.  The script automatically matches BCF sample names to
`<sample>.bed` files and trains on all matched samples:

```bash
python scripts/ml_cnv_calling.py train \
    --bcf multisample.bcf \
    --truth-dir truth_sets/per_sample/ \
    --min-probes 5 \
    --overlap-report overlap.tsv \
    --output cnv_model.pt
```

| Option | Description |
|--------|-------------|
| `--truth-dir` | Directory of `<sample>.bed` truth files (mutually exclusive with `--truth-bed`). |
| `--min-probes N` | Exclude truth regions overlapping fewer than *N* array probes (default: 1). |
| `--overlap-report` | Write a TSV listing every sample as `matched`, `bcf_only`, or `truth_only`. |

The overlap report (`overlap.tsv`) has columns:

```
sample    in_bcf    in_truth    status
HG00096   True      True        matched
NA12878   True      False       bcf_only
HG00099   False     True        truth_only
```

In multi-sample mode the train/validation split is by *sample* (90/10) to
prevent data leakage.

#### Prediction

```bash
python scripts/ml_cnv_calling.py predict \
    --bcf sample.bcf \
    --model cnv_model.pt \
    --output cnv_calls.bed
```

The output is a tab-separated BED file of predicted CNV regions – adjacent
probes sharing the same non-normal class are collapsed into contiguous
segments:

```
chr1    1500000    2300000    DEL    47
chr3    45000000   45800000   DUP    22
```

Columns: `chrom  start  end  svtype  num_probes`

## Data Sources & Citations

This project relies on the following data sources and publications:

> **Schloissnig, S., Pani, S. et al.** Long-read sequencing and structural
> variant characterization in 1,019 samples from the 1000 Genomes Project.
> *bioRxiv* 2024.04.18.590093 (2024).
> doi:[10.1101/2024.04.18.590093](https://doi.org/10.1101/2024.04.18.590093)
> — [PubMed](https://pubmed.ncbi.nlm.nih.gov/38659906/)

| Resource | Description |
|----------|-------------|
| [1KG ONT Vienna data collection][1kgp_ont] | Population-scale long-read SV calls from Oxford Nanopore sequencing of 1,019 1000 Genomes samples. |
| [shapeit5-phased callset][phased_vcf] | The phased, sequence-resolved SV VCF used as the primary training truth set (included in `resources/`). |
| [PMC12350158](https://pmc.ncbi.nlm.nih.gov/articles/PMC12350158/) | Curated structural variant truth sets. |

### Samples for methods development and evaluation

A subset of 1KG_ONT_VIENNA genomes overlap with samples analysed by the Human
Genome Structural Variation Consortium and are designated for methods
development and evaluation prior to publication by the project team:

> HG00268, HG00513, HG00731, HG02554, HG02953, NA12878, NA19129, NA19238,
> NA19331, NA19347

A subset VCF containing only these 10 samples is provided in
`tests/fixtures/test_samples.vcf.gz` for use in integration testing.

If you use this software or its data resources, please cite the publication
above and see [`CITATION.cff`](CITATION.cff) for machine-readable citation
metadata.

## HPC Usage with Apptainer

Most HPC clusters do not permit Docker but support
[Apptainer](https://apptainer.org/) (the successor to Singularity).  An
`Apptainer.def` definition file is provided so you can build a portable SIF
image that bundles the model code, Python dependencies, and the bundled truth
set VCF.

### Building the SIF image

```bash
apptainer build array_cnv_caller.sif Apptainer.def
```

### Batteries-included training pipeline

`scripts/run_pipeline.sh` orchestrates the complete workflow – truth-set
preparation, multi-sample training, and (optionally) prediction – in a single
command.  It targets the ~2,141-sample 1000 Genomes BCF produced by
[`process_1000g.sh`](https://github.com/jlanej/illumina_idat_processing/blob/main/scripts/process_1000g.sh)
and the shapeit5-phased truth set bundled in the container.

```bash
# Full training pipeline via Apptainer
bash scripts/run_pipeline.sh \
    --bcf /path/to/1000g_multisample.bcf \
    --sif array_cnv_caller.sif \
    --outdir pipeline_output \
    --epochs 30 \
    --min-probes 5 \
    --device auto

# With prediction on every sample after training
bash scripts/run_pipeline.sh \
    --bcf /path/to/1000g_multisample.bcf \
    --sif array_cnv_caller.sif \
    --predict

# Native execution (no container)
bash scripts/run_pipeline.sh \
    --bcf /path/to/1000g_multisample.bcf \
    --native
```

The pipeline:

1. **Prepares truth-set BED files** from the shapeit5-phased SV VCF (skipped
   if the output directory already contains BED files from a previous run).
2. **Trains the CNVSegmenter** on every BCF sample that has a matching
   truth BED file.  Samples are split 90/10 by sample for train/validation to
   prevent data leakage.
3. **(Optional)** Runs prediction on BCF samples using the trained model.

Pipeline options:

| Option | Default | Description |
|--------|---------|-------------|
| `--bcf` | *(required)* | Multi-sample BCF with FORMAT/LRR and FORMAT/BAF |
| `--sif` | — | Apptainer SIF image (use `--native` instead for bare-metal) |
| `--native` | — | Run without a container |
| `--truth-vcf` | bundled | Override the default shapeit5-phased VCF |
| `--outdir` | `pipeline_output` | Output directory |
| `--epochs` | 30 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--min-probes` | 5 | Min array probes per truth region |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, … |
| `--predict` | off | Also run prediction after training |
| `--bind` | — | Extra Apptainer bind mounts (e.g. `/scratch:/scratch`) |

### Running individual steps with Apptainer

```bash
# Prepare truth sets
apptainer run array_cnv_caller.sif \
    scripts/prepare_truth_set.py \
    --vcf resources/shapeit5-phased-callset_final-vcf.phased.vcf.gz \
    --output-dir truth_sets

# Train (bind-mount your data directory)
apptainer run --nv \
    --bind /data:/data \
    array_cnv_caller.sif \
    scripts/ml_cnv_calling.py train \
    --bcf /data/1000g_multisample.bcf \
    --truth-dir truth_sets/per_sample/ \
    --min-probes 5 \
    --overlap-report overlap.tsv \
    --output cnv_model.pt

# Predict
apptainer run --nv \
    --bind /data:/data \
    array_cnv_caller.sif \
    scripts/ml_cnv_calling.py predict \
    --bcf /data/sample.bcf \
    --model cnv_model.pt \
    --output cnv_calls.bed
```

> **GPU support:** Pass `--nv` to `apptainer run` for NVIDIA GPU passthrough.
> The pipeline script does this automatically when `--device` is not `cpu`.

### Example SLURM job script

```bash
#!/bin/bash
#SBATCH --job-name=array_cnv_train
#SBATCH --output=train_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

module load apptainer   # or: module load singularity

bash scripts/run_pipeline.sh \
    --bcf /scratch/$USER/1000g_multisample.bcf \
    --sif array_cnv_caller.sif \
    --outdir /scratch/$USER/cnv_output \
    --epochs 30 \
    --min-probes 5 \
    --device auto \
    --predict \
    --bind /scratch:/scratch
```

## Docker

A Docker image containing all dependencies is published automatically via
GitHub Actions. To build locally:

```bash
docker build -t array_cnv_caller .
```

The Docker image can also be converted to an Apptainer SIF for HPC use:

```bash
apptainer build array_cnv_caller.sif docker-daemon://array_cnv_caller:latest
```

## Testing

```bash
pip install pytest
pytest tests/ -v              # all tests
pytest tests/ -v -m integration  # integration tests only
```

## License

See [LICENSE](LICENSE).
