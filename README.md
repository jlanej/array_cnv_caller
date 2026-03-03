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

[1kgp_ont]: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1KG_ONT_VIENNA/

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
| **FC head** | A fully connected layer predicts one of 5 copy-number states (CN 0–4) per probe. |

The model accepts **three input channels** per probe:

1. **LRR** – Log R Ratio
2. **BAF** – B-Allele Frequency
3. **Distance** – log₁₀-scaled distance to the next probe

Encoding inter-probe distance explicitly makes the model **robust to different
Illumina array probe densities** (e.g. Omni2.5 vs. GSA vs. Mega).

Training uses **weighted cross-entropy loss** to handle the extreme class
imbalance inherent in array data (the vast majority of probes are CN = 2).

### Truth Set Preparation

The training labels come from the 1000 Genomes ONT Vienna release.  The helper
script downloads the merged Sniffles2 SV VCF and converts DEL/DUP calls into
per-sample BED files with copy-number labels:

```bash
bash scripts/prepare_truth_set.sh [output_dir]
```

Output BED format:  `chrom  start  end  CN  svtype`

CN mapping:

| Genotype | SV type | CN |
|----------|---------|----|
| hom DEL  | DEL     | 0  |
| het DEL  | DEL     | 1  |
| normal   | —       | 2  |
| het DUP  | DUP     | 3  |
| hom DUP  | DUP     | 4  |

### Installation

```bash
pip install -r requirements_ml.txt
```

### Usage

#### Training

```bash
python scripts/ml_cnv_calling.py train \
    --bcf sample.bcf \
    --truth-bed truth_sets/per_sample/SAMPLE.bed \
    --output cnv_model.pt \
    --epochs 30 \
    --device auto
```

#### Prediction

```bash
python scripts/ml_cnv_calling.py predict \
    --bcf sample.bcf \
    --model cnv_model.pt \
    --output cnv_calls.bed
```

The output is a tab-separated BED file of predicted CNV regions – adjacent
probes sharing the same non-diploid state (CN ≠ 2) are collapsed into
contiguous segments:

```
chr1    1500000    2300000    1    47
chr3    45000000   45800000   3    22
```

Columns: `chrom  start  end  CN  num_probes`

## License

See [LICENSE](LICENSE).
