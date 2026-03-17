# Blacklist Region Sources

This directory contains BED files defining genomic regions that are known to be
problematic for long-read sequencing alignment, copy-number calling, or
microarray signal interpretation, specifically for the **T2T-CHM13 v2.0**
assembly.  These files are used by `litmus_test.py` to annotate and optionally
exclude probes that fall in difficult regions, allowing users to assess whether
downstream results are driven by known artefacts.

All coordinates are for the **T2T-CHM13 v2.0 (GCA_009914755.4)** human
reference assembly, in **0-based, half-open** BED format.

---

## Assembly

> Nurk S, Koren S, Rhie A *et al.* The complete sequence of a human genome.
> *Science.* 2022;376(6588):44–53.
> doi:[10.1126/science.abj6987](https://doi.org/10.1126/science.abj6987)

T2T-CHM13 v2.0 (also referred to as CHM13) is the first gapless, complete
assembly of a human genome, resolving all centromeres, telomeres, and
previously-inaccessible repetitive regions.  Chromosome sizes used here match
the VCF contig lengths in the accompanying array data (e.g.
`chr1 = 248,387,328 bp`, `chr9 = 150,617,247 bp`).

---

## Files

### `chm13v2_centromeres.bed`

| Field | Details |
|-------|---------|
| **Regions** | 24 (one per canonical chromosome) |
| **Source** | Altemose N *et al.* "Complete genomic and epigenetic maps of human centromeres." *Science.* 2022;376(6588):eabl4178. doi:[10.1126/science.abl4178](https://doi.org/10.1126/science.abl4178) — Table S2: active HOR array boundaries in CHM13v2.0, extended by ±500 kb to capture pericentromeric satellite flanks. |
| **Track** | T2T Consortium `cenSat` annotation: <https://github.com/marbl/CHM13/blob/master/Supplemental/cenSat_v2.0_CHM13.bed> |
| **Rationale** | Although CHM13 fully sequences centromeres, they remain dominated by α-satellite Higher-Order Repeats (HORs) and satellite DNA.  Array probes in these regions show inflated LRR variance and erratic BAF due to probe cross-hybridisation across nearly-identical repeat units.  Long-read SV callers also produce more false positives here because reads map non-uniquely within the HOR arrays. |

### `chm13v2_telomeres.bed`

| Field | Details |
|-------|---------|
| **Regions** | 48 (two per canonical chromosome: 5′ and 3′ ends) |
| **Source** | Chromosome sizes from GCA_009914755.4 assembly report (confirmed from VCF contig lengths). Telomere sizes reported in Nurk *et al.* 2022; 10 kb windows used as a conservative proxy (actual CHM13 telomere lengths range from ~4 kb on chrY to ~15 kb on chr10). |
| **Track** | T2T Consortium telomere annotation: <https://github.com/marbl/CHM13/blob/master/Supplemental/telomere_v2.0.bed> |
| **Rationale** | Telomeric (TTAGGG)n repeats have no unique alignment anchor.  While CHM13 resolves the sub-telomeric transition, the outermost repeat arrays still cause reads to mis-map and produce noisy array probe signals (near-zero unique mappability). |

### `chm13v2_encode_blacklist.bed`

| Field | Details |
|-------|---------|
| **Regions** | Pericentromeric heterochromatin flanks, large satellite arrays (HSat1–3, alphoid), and known array signal artefact zones for CHM13. |
| **Source** | Adapted from Amemiya HM, Kundaje A, Boyle AP. "The ENCODE Blacklist: Identification of Problematic Regions of the Genome." *Sci Rep.* 2019;9(1):9354. doi:[10.1038/s41598-019-45839-z](https://doi.org/10.1038/s41598-019-45839-z). Regions transferred and extended using the T2T Consortium difficult-region annotations. |
| **Full blacklist** | <https://github.com/Boyle-Lab/Blacklist> — see the CHM13 release for T2T-specific coordinates. |
| **T2T difficult regions** | <https://github.com/marbl/CHM13/blob/master/Supplemental/> |
| **Rationale** | The ENCODE Blacklist flags regions with anomalously high signal in virtually all NGS assays, regardless of the experimental protocol.  For T2T CHM13, additional satellite-rich flanks (especially chr9 pericentromere, chr13/14/15/21/22 acrocentric short arms) were added because these regions are known to produce spurious LRR elevations in Illumina arrays and alignment pile-ups in long-read SV callers. |

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

To download precise T2T CHM13 v2.0 annotations directly from the consortium:

```bash
# Centromere alpha-satellite boundaries
wget -O resources/blacklists/chm13v2_centromeres.bed \
    https://github.com/marbl/CHM13/raw/master/Supplemental/cenSat_v2.0_CHM13.bed

# Telomere annotations
wget -O resources/blacklists/chm13v2_telomeres.bed \
    https://github.com/marbl/CHM13/raw/master/Supplemental/telomere_v2.0.bed

# Full ENCODE Blacklist (CHM13 edition, when available)
wget -O resources/blacklists/chm13v2_encode_blacklist.bed.gz \
    https://github.com/Boyle-Lab/Blacklist/raw/master/lists/chm13-blacklist.v2.bed.gz
gunzip resources/blacklists/chm13v2_encode_blacklist.bed.gz
```

---

## References

1. **T2T Assembly** — Nurk S, Koren S, Rhie A *et al.* The complete sequence of a human
   genome. *Science.* 2022;376(6588):44–53.
   doi:[10.1126/science.abj6987](https://doi.org/10.1126/science.abj6987)

2. **T2T Centromeres** — Altemose N *et al.* Complete genomic and epigenetic maps of
   human centromeres. *Science.* 2022;376(6588):eabl4178.
   doi:[10.1126/science.abl4178](https://doi.org/10.1126/science.abl4178)

3. **ENCODE Blacklist** — Amemiya HM, Kundaje A, Boyle AP. The ENCODE Blacklist:
   Identification of Problematic Regions of the Genome. *Sci Rep.* 2019;9(1):9354.
   doi:[10.1038/s41598-019-45839-z](https://doi.org/10.1038/s41598-019-45839-z)

4. **T2T Consortium GitHub** — marbl/CHM13:
   <https://github.com/marbl/CHM13>

5. **UCSC T2T Browser** — T2T hub at UCSC Genome Browser:
   <https://genome.ucsc.edu/cgi-bin/hgGateway?db=hub_3671779_GCA_009914755.4>
