#!/usr/bin/env bash
# prepare_truth_set.sh
#
# Downloads and converts the 1000 Genomes ONT Vienna structural variant
# truth set into per-sample BED files suitable for training the ML CNV caller.
#
# The source data comes from:
#   https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1KG_ONT_VIENNA/
# and the companion analysis repository:
#   https://github.com/1kg-ont-vienna/sv-analysis
#
# The curated population-scale SV callset (167k+ sequence-resolved SVs from
# ONT long-read sequencing of 1,019 samples) serves as the ground truth for
# copy-number state labelling of Illumina array probes.
#
# Usage:
#   bash scripts/prepare_truth_set.sh [output_dir]
#
# Requirements: bcftools, bedtools, wget/curl

set -euo pipefail

OUTPUT_DIR="${1:-truth_sets}"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 1.  Download the merged SV VCF from the 1KG ONT Vienna release
# ---------------------------------------------------------------------------
SV_VCF_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1KG_ONT_VIENNA/release/2024-09-20_v2/sniffles2/multisample/1kGP.sniffles2.vcf.gz"
SV_VCF_TBI_URL="${SV_VCF_URL}.tbi"
SV_VCF="${OUTPUT_DIR}/1kGP.sniffles2.vcf.gz"

if [ ! -f "${SV_VCF}" ]; then
    echo "[INFO] Downloading 1KG ONT Vienna merged SV VCF ..."
    wget -q --show-progress -O "${SV_VCF}" "${SV_VCF_URL}"
    wget -q --show-progress -O "${SV_VCF}.tbi" "${SV_VCF_TBI_URL}"
fi

# ---------------------------------------------------------------------------
# 2.  Extract CNV-relevant SV types (DEL, DUP) and convert to BED
# ---------------------------------------------------------------------------
# We keep DEL and DUP calls that have PASS or high-confidence filters.
# For each variant we record:  chrom  start  end  svtype  sample  genotype
#
# The copy-number label mapping used downstream:
#   DEL heterozygous (0/1)  -> CN=1
#   DEL homozygous   (1/1)  -> CN=0
#   DUP heterozygous (0/1)  -> CN=3
#   DUP homozygous   (1/1)  -> CN=4
#   No SV overlap           -> CN=2  (assigned during training)
# ---------------------------------------------------------------------------

CNV_BED="${OUTPUT_DIR}/1kGP_cnv_truth.bed.gz"

if [ ! -f "${CNV_BED}" ]; then
    echo "[INFO] Extracting DEL/DUP calls to BED format ..."

    bcftools view -i 'INFO/SVTYPE="DEL" || INFO/SVTYPE="DUP"' "${SV_VCF}" \
    | bcftools query \
        -f '[%CHROM\t%POS0\t%END\t%INFO/SVTYPE\t%SAMPLE\t%GT\n]' \
    | awk -F'\t' '
        # Keep only samples carrying an alternate allele
        $6 != "0/0" && $6 != "./." && $6 != "0|0" {
            # Map genotype to copy-number state
            cn = 2
            if ($4 == "DEL") {
                if ($6 == "1/1" || $6 == "1|1") cn = 0
                else cn = 1
            } else if ($4 == "DUP") {
                if ($6 == "1/1" || $6 == "1|1") cn = 4
                else cn = 3
            }
            print $1"\t"$2"\t"$3"\t"$5"\t"cn"\t"$4
        }
    ' \
    | sort -k4,4 -k1,1V -k2,2n \
    | gzip > "${CNV_BED}"

    echo "[INFO] Wrote $(zcat "${CNV_BED}" | wc -l) CNV truth records."
fi

# ---------------------------------------------------------------------------
# 3.  Split into per-sample BED files (for sample-level training)
# ---------------------------------------------------------------------------
PER_SAMPLE_DIR="${OUTPUT_DIR}/per_sample"
mkdir -p "${PER_SAMPLE_DIR}"

if [ -z "$(ls -A "${PER_SAMPLE_DIR}" 2>/dev/null)" ]; then
    echo "[INFO] Splitting truth set by sample ..."

    zcat "${CNV_BED}" | awk -F'\t' '{
        sample = $4
        outfile = "'"${PER_SAMPLE_DIR}"'/" sample ".bed"
        print $1"\t"$2"\t"$3"\t"$5"\t"$6 > outfile
    }'

    echo "[INFO] Created $(ls "${PER_SAMPLE_DIR}" | wc -l) per-sample BED files."
fi

echo "[DONE] Truth set prepared in ${OUTPUT_DIR}/"
echo ""
echo "Per-sample BED format:  chrom  start  end  CN  svtype"
echo "CN states: 0=homDEL  1=hetDEL  2=normal  3=hetDUP  4=homDUP"
echo ""
echo "Next step: use these BED files with the ML CNV caller:"
echo "  python scripts/ml_cnv_calling.py train \\"
echo "    --bcf <sample>.bcf \\"
echo "    --truth-bed ${PER_SAMPLE_DIR}/<sample>.bed \\"
echo "    --output model.pt"
