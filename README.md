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

## License

See [LICENSE](LICENSE).
