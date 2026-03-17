# Blacklist Region Sources

This directory contains BED files defining genomic regions that are known to be
problematic for long-read sequencing alignment, copy-number calling, or
microarray signal interpretation, specifically for the **T2T-CHM13 v2.0**
assembly.  These files are used by `litmus_test.py` to annotate and optionally
exclude probes that fall in difficult regions, allowing users to assess whether
downstream results are driven by known artefacts.

> **All coordinates in all BED files in this directory are for T2T-CHM13 v2.0
> (GCA_009914755.4), NOT hg38.**  Coordinate accuracy is confirmed by the
> chromosome end positions matching T2T contig sizes extracted from the array
> BCF header (e.g. `chr1 = 248,387,328 bp`, `chr9 = 150,617,247 bp`).

All intervals are in **0-based, half-open** BED format.

---

## Assembly

> Nurk S, Koren S, Rhie A *et al.* The complete sequence of a human genome.
> *Science.* 2022;376(6588):44–53.
> doi:[10.1126/science.abj6987](https://doi.org/10.1126/science.abj6987)

T2T-CHM13 v2.0 (also referred to as CHM13) is the first gapless, complete
assembly of a human genome, resolving all centromeres, telomeres, and
previously-inaccessible repetitive regions.

NCBI assembly record: <https://www.ncbi.nlm.nih.gov/assembly/GCA_009914755.4>

---

## Files

### `chm13v2_centromeres.bed`

| Field | Details |
|-------|---------|
| **Assembly** | T2T-CHM13 v2.0 (GCA_009914755.4) |
| **Regions** | 24 (one per canonical chromosome) |
| **Provenance** | Coordinates derived from Table S2 of Altemose *et al.* 2022 (active HOR array boundaries in CHM13v2.0), extended by ±500 kb to capture pericentromeric satellite-dense flanks. |
| **Paper** | Altemose N *et al.* "Complete genomic and epigenetic maps of human centromeres." *Science.* 2022;376(6588):eabl4178. doi:[10.1126/science.abl4178](https://doi.org/10.1126/science.abl4178) |
| **Direct download (canonical T2T track)** | T2T / Human Pangenomics S3 bucket — cenSat v2.0 annotation:<br>`https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/annotation/chm13v2.0_censat_v2.0.bed` |
| **UCSC Browser** | <https://genome.ucsc.edu/cgi-bin/hgGateway?db=hub_3671779_GCA_009914755.4> (cenSat track) |
| **Rationale** | Although CHM13 fully sequences centromeres, they remain dominated by α-satellite Higher-Order Repeats (HORs) and satellite DNA.  Array probes in these regions show inflated LRR variance and erratic BAF due to probe cross-hybridisation across nearly-identical repeat units.  Long-read SV callers also produce more false positives here because reads map non-uniquely within the HOR arrays. |

### `chm13v2_telomeres.bed`

| Field | Details |
|-------|---------|
| **Assembly** | T2T-CHM13 v2.0 (GCA_009914755.4) |
| **Regions** | 48 (two per canonical chromosome: 5′ and 3′ ends) |
| **Provenance** | 10 kb windows computed directly from confirmed T2T CHM13 v2.0 contig sizes extracted from the array BCF header (matching GCA_009914755.4). Actual CHM13 telomere lengths range from ~4 kb (chrY) to ~15 kb (chr10) as reported in Nurk *et al.* 2022. |
| **Paper** | Nurk S *et al.* "The complete sequence of a human genome." *Science.* 2022;376(6588):44–53. doi:[10.1126/science.abj6987](https://doi.org/10.1126/science.abj6987) |
| **Direct download (canonical T2T track)** | T2T / Human Pangenomics S3 bucket — telomere annotation:<br>`https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/annotation/chm13v2.0_telomere.bed` |
| **NCBI assembly** | <https://www.ncbi.nlm.nih.gov/assembly/GCA_009914755.4> |
| **Rationale** | Telomeric (TTAGGG)n repeats have no unique alignment anchor.  While CHM13 resolves the sub-telomeric transition, the outermost repeat arrays still cause reads to mis-map and produce noisy array probe signals (near-zero unique mappability). |

### `chm13v2_encode_blacklist.bed`

| Field | Details |
|-------|---------|
| **Assembly** | T2T-CHM13 v2.0 (GCA_009914755.4) — all coordinates are T2T |
| **Regions** | 79 (pericentromeric flanks, acrocentric short arms, telomeres, chr9 heterochromatic block) |
| **Provenance** | **MANUALLY CURATED using T2T CHM13 v2.0 coordinates.** The [Boyle-Lab ENCODE Blacklist project](https://github.com/Boyle-Lab/Blacklist) does **not** have a published CHM13/T2T release (it provides hg38/hg19/mm10 only). The T2T-specific regions here were constructed from four canonical T2T sources: (1) centromere alpha-satellite arrays from the T2T cenSat v2.0 track; (2) telomere terminal windows from the T2T telomere annotation; (3) acrocentric short arms (chr13/14/15/21/22), which are entirely satellite-rich in CHM13 and produce cross-hybridisation artefacts in Illumina arrays; (4) chr9 pericentromeric heterochromatic block (q-arm). |
| **Conceptual framework** | Amemiya HM, Kundaje A, Boyle AP. "The ENCODE Blacklist: Identification of Problematic Regions of the Genome." *Sci Rep.* 2019;9(1):9354. doi:[10.1038/s41598-019-45839-z](https://doi.org/10.1038/s41598-019-45839-z) |
| **Coordinate sources** | cenSat v2.0 (`chm13v2_centromeres.bed`) + telomere track (`chm13v2_telomeres.bed`) — see rows above for direct T2T S3 URLs |
| **Rationale** | In the absence of an official CHM13 ENCODE Blacklist, this file provides a conservative set of T2T-coordinate regions known from the literature to produce spurious LRR/BAF signal in Illumina arrays and alignment pile-ups in long-read SV callers. |

---

## Using a Custom Blacklist Directory

Any directory of `.bed` files can be passed to `litmus_test.py` via
`--blacklist-dir`.  Each file's stem (filename without the `.bed` extension) is
used as the region-type label in the dashboard.  For example, a file named
`segmental_duplications.bed` will appear as `segmental_duplications` in the
HTML report.

```bash
python scripts/litmus_test.py \
    --bcf my_array.bcf \
    --truth-dir truth_beds/ \
    --blacklist-dir resources/blacklists/
```

### Replacing packaged files with canonical T2T upstream downloads

The centromere and telomere BED files can be replaced with the authoritative
canonical T2T annotations using the direct S3 URLs below.

```bash
# Centromere alpha-satellite HOR array boundaries (cenSat v2.0) — T2T CHM13 v2.0
# Direct URL: https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/annotation/chm13v2.0_censat_v2.0.bed
wget -O resources/blacklists/chm13v2_centromeres.bed \
    https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/annotation/chm13v2.0_censat_v2.0.bed

# Telomere annotations — T2T CHM13 v2.0
# Direct URL: https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/annotation/chm13v2.0_telomere.bed
wget -O resources/blacklists/chm13v2_telomeres.bed \
    https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/annotation/chm13v2.0_telomere.bed

# chm13v2_encode_blacklist.bed is manually curated from the above T2T tracks;
# there is no upstream download equivalent (no CHM13 ENCODE Blacklist exists yet).
# Watch https://github.com/Boyle-Lab/Blacklist for a future CHM13 release.
```

---

## References

1. **T2T Assembly** — Nurk S, Koren S, Rhie A *et al.* The complete sequence of a human
   genome. *Science.* 2022;376(6588):44–53.
   doi:[10.1126/science.abj6987](https://doi.org/10.1126/science.abj6987)

2. **T2T Centromeres** — Altemose N *et al.* Complete genomic and epigenetic maps of
   human centromeres. *Science.* 2022;376(6588):eabl4178.
   doi:[10.1126/science.abl4178](https://doi.org/10.1126/science.abl4178)

3. **ENCODE Blacklist concept** — Amemiya HM, Kundaje A, Boyle AP. The ENCODE Blacklist:
   Identification of Problematic Regions of the Genome. *Sci Rep.* 2019;9(1):9354.
   doi:[10.1038/s41598-019-45839-z](https://doi.org/10.1038/s41598-019-45839-z)
   *(Note: the Boyle-Lab implementation only covers hg38/hg19/mm10 — no CHM13 version
   exists. Our `chm13v2_encode_blacklist.bed` uses T2T CHM13 v2.0 coordinates throughout.)*

4. **T2T / Human Pangenomics S3 Annotations** — Direct T2T downloads:
   - Index: <https://s3-us-west-2.amazonaws.com/human-pangenomics/index.html?prefix=T2T/CHM13/assemblies/annotation/>
   - cenSat v2.0: `https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/annotation/chm13v2.0_censat_v2.0.bed`
   - Telomere: `https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/annotation/chm13v2.0_telomere.bed`

5. **NCBI Assembly GCA_009914755.4** — T2T-CHM13 v2.0 record:
   <https://www.ncbi.nlm.nih.gov/assembly/GCA_009914755.4>

6. **UCSC T2T Browser** — T2T hub at UCSC Genome Browser:
   <https://genome.ucsc.edu/cgi-bin/hgGateway?db=hub_3671779_GCA_009914755.4>
