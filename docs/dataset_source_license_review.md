# Dataset Source and Redistribution Review

This is a preliminary source and redistribution review for the `Final_datasets` archive. It is based on the manuscript PDFs in `papers/`, local CSV headers, and public source pages checked on 2026-07-08. It is not legal advice. Before publishing the Zenodo record openly, each upstream dataset license should be verified against the exact files used to build the processed CSVs.

## Summary recommendation

Do not publish the entire archive as a single `cc-by-4.0` open dataset until all upstream licenses are confirmed. Several sources are clearly open with attribution, but at least one major source, QM9, is licensed as CC BY-NC-SA 4.0 in the Scientific Data article, which is not equivalent to CC BY 4.0 and restricts commercial reuse. Other sources are public or open in the scientific sense, but their redistribution license was not fully identifiable from the manuscript and accessible public pages.

The safest current Zenodo mode is therefore:

- create/upload as a draft or restricted record;
- share a secret preview link with collaborators or reviewers;
- publish openly only after license confirmation, or split the archive into license-compatible subsets.

## Feature-generation notes

- Composition-only and composition-plus-structure materials tables were featurized mainly with Magpie-style composition descriptors.
- Datasets containing crystal structures or structure-derived information additionally include geometric or local-environment descriptors; the full `3DSC_TC.csv` includes SOAP descriptors, while `3DSC_TC_nosoap.csv` removes SOAP and keeps the non-SOAP descriptor set.
- Molecular datasets using SMILES were converted into fixed-length molecular fingerprint descriptors.
- The OMol file uses 128-dimensional latent outputs extracted from a universal machine-learning interatomic potential (universal MLIP) as features; the license of both the source OMol data and the model-generated latent representation should be checked before open redistribution.

## Per-dataset review

| File | Source basis | Feature representation in this archive | Copyright/license status | Current redistribution recommendation |
|---|---|---|---|---|
| `HEA_exp_target_Hardness.csv` | Curated experimental HEA hardness benchmark used in the study. The supplied PDFs do not identify a standalone upstream repository or license. | Composition field plus Magpie-style composition descriptors. | Not verified. Treat as author-curated/third-party experimental compilation until source license is documented. | Do not publish openly without confirming the original source and redistribution rights. |
| `curie_temperature.csv` | Long et al. (2021), ferromagnetic-material Curie-temperature work. | Composition field plus Magpie-style composition descriptors. | Article/source identified, but dataset redistribution license not verified from public pages. | Verify whether the exact tabular data are author-owned or third-party-derived before open release. |
| `AM.csv` | Curated additive-manufacturing process/property benchmark used in the study. | Direct process/material input variables; no obvious Magpie/SMILES descriptor expansion. | Upstream source and license not identified in supplied PDFs. | Do not publish openly until the original dataset source and permission are documented. |
| `steel_neatened.csv` | Matbench/Automatminer steel benchmark source from Dunn et al. (2020). | Composition and process/testing fields plus Magpie-style descriptors. | Matbench repository is MIT-licensed; matminer repository uses a BSD-style license. Confirm whether the exact steel task data are covered by those terms or by additional source terms. | Likely redistributable with proper attribution if sourced from Matbench, but preserve upstream notices and verify task-level metadata. |
| `melting_points.csv` | Jean-Claude Bradley curated melting-point dataset cited by the manuscript. | SMILES plus fixed-length molecular fingerprint descriptors. | Dataset source cited, but explicit redistribution license was not verified in accessible pages during this review. | Verify original dataset license before open release. |
| `exp_bandgap.csv` | Inorganic band-gap dataset from Zhuo, Mansouri Tehrani, and Brgoch (2018). | Composition plus Magpie-style descriptors. | Article/source identified, but dataset redistribution license not verified. | Verify original data license before open release. |
| `3DSC_TC_nosoap.csv` | 3DSC-A superconductors dataset from Sommer et al. (2023), Materials Project-derived public subset. | Composition/crystal-structure-derived descriptors and Magpie-style descriptors; SOAP removed. | Scientific Data article is CC BY 4.0. The paper states the Materials Project-based 3DSC_MP dataset is publicly provided, while ICSD-derived structures must not be republished. | Likely suitable for redistribution with attribution if this file is only based on 3DSC_MP and not ICSD structures. State modifications clearly. |
| `3DSC_TC.csv` | Same 3DSC-A source as above. | Composition/crystal-structure-derived descriptors, Magpie-style descriptors, and SOAP descriptors. | Same as above; confirm no ICSD-derived structures or restricted fields are included. | Likely suitable if Materials Project-based only; include 3DSC citation and descriptor-generation notice. |
| `dielectric.csv` | Dielectric/optical materials benchmark cited to Petousis et al. (2017). | Composition/structure-derived descriptors plus Magpie-style descriptors. | Source citation identified in manuscript; exact dataset license not verified. Some versions of related benchmark tasks may be distributed via matminer/Matbench, but the exact source must be confirmed. | Verify exact upstream source and license before open release. |
| `optimade_2dmatpedia.csv` | 2DMatPedia dataset from Zhou et al. (2019), accessed/processed through OPTIMADE-style metadata. | Composition/structure-derived descriptors plus Magpie-style descriptors. | Scientific Data article is CC BY 4.0 and article metadata are CC0. Confirm whether downloaded 2DMatPedia data carry the same or separate terms. | Probably publishable with attribution if consistent with 2DMatPedia terms; verify the database download license first. |
| `bgfEmag.csv` | Materials Project-derived benchmark based on Jain et al. (2013). | Composition/structure-derived descriptors plus Magpie-style descriptors. | Materials Project data are public/open-access, but current API/data terms should be checked for redistribution and attribution requirements. | Verify current Materials Project terms before redistributing derived bulk tables. |
| `powerfactor2.csv` | Thermoelectric power-factor benchmark from Lim et al. (2021). | Mixed materials descriptors and computed quantities; not a simple Magpie-only table. | Article/source identified; data redistribution license not verified. | Verify source data license before open release. |
| `QM9_dataset.csv` | QM9 molecular dataset from Ramakrishnan et al. (2014). | SMILES plus fixed-length molecular fingerprint descriptors. | Scientific Data page states CC BY-NC-SA 4.0 for the work, with metadata under CC0. This is not compatible with relicensing the data as CC BY 4.0. | Do not publish the whole archive as CC BY 4.0 if QM9 is included. Consider preserving CC BY-NC-SA terms for this file, removing QM9 from an open CC-BY release, or obtaining permission. |
| `alex_icams.csv` | Alexandria/ICAMS large materials dataset from Schmidt et al. (2024). | Composition/structure-derived descriptors plus Magpie-style descriptors; includes Alexandria target fields. | Public pages indicate Alexandria data/workflows are available under Creative Commons licenses, but the exact license for the downloaded subset was not verified. | Verify exact Alexandria/ICAMS license before open release. |
| `omol_4M_gnn_single.csv` | OMol25 molecular dataset from Levine et al. (2025). | 128-dimensional latent features from a universal MLIP; target is HOMO-LUMO gap. | OMol25 is publicly released, but exact dataset and model-output licensing were not verified here. OMol25 also includes structures obtained from existing datasets, which may impose inherited terms. | Do not publish openly until OMol25 data license and universal-MLIP/model-output terms are confirmed. |

## Practical publication options

1. **Restricted Zenodo draft now**: acceptable for collaborator/reviewer preview, with no public DOI registration yet.
2. **Open subset release**: split out only datasets whose licenses are verified as compatible with the chosen Zenodo license.
3. **Script-and-provenance release**: publish code, descriptors-generation scripts, row identifiers, and source instructions; ask users to download original data from upstream sources.
4. **Mixed-license archive**: publish all files only if Zenodo metadata and file-level documentation clearly preserve each upstream license. Avoid declaring a single permissive license if one file is under NC/SA or unknown terms.

## Sources checked

- 3DSC Scientific Data page: public 3DSC_MP availability, ICSD non-republication statement, and CC BY 4.0 article license.
- 2DMatPedia Scientific Data page: CC BY 4.0 article license and CC0 metadata statement.
- QM9 Scientific Data page: Figshare data citation and CC BY-NC-SA 4.0 rights statement.
- Matbench and matminer GitHub license files: Matbench MIT license and matminer BSD-style license.
- Associated manuscript `BO_main.pdf` and supplementary `BO_supp.pdf`: dataset list, file-level source citations, sample counts, and benchmark roles.
