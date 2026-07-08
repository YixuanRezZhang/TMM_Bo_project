# Final materials closed-pool benchmark datasets

This archive contains the final CSV datasets used for the closed-pool materials optimization benchmarks in the associated Bayesian optimization study and code repository:

https://github.com/YixuanRezZhang/TMM_Bo_project

The files are prepared as tabular candidate pools for benchmarking Bayesian optimization and active sampling methods. Each row is one candidate material, molecule, alloy, or process condition. Feature columns contain composition-derived descriptors, structure-derived descriptors, molecular fingerprints, SOAP descriptors, universal-MLIP latent features, or process descriptors depending on the dataset. Target columns are prefixed with `target_` where possible and define the objective values used by the benchmark scripts.

## Dataset construction summary

The benchmark suite was assembled to span small experimental datasets, medium-size simulation datasets, and large high-dimensional molecular/materials pools. The accompanying manuscript reports 15 closed-pool datasets with sizes from 635 to 3,986,754 samples and feature dimensionalities from 10 to 8,863. Dataset complexity was characterized using sample count, raw feature dimensionality, intrinsic dimensionality estimated from PCA variance coverage, effective resolution, objective conflict, graph distances, neighborhood sparsity, feature-target mutual information proxies, autocorrelation, correlation length, multimodality, channel capacity, basin contrast, and deceptive-peak metrics.

The datasets include single-objective and multi-objective tasks. Examples include experimental hardness, Curie temperature, additive-manufacturing relative density, steel mechanical properties, melting point, inorganic band gap, superconducting critical temperature with auxiliary targets, dielectric refractive index, 2D-material band gap, Materials Project formation energy/magnetization/band-gap targets, thermoelectric power factor, QM9 molecular properties, Alexandria/ICAMS formation-energy/magnetization/band-gap targets, and OMol HOMO-LUMO gap targets.

## Feature and descriptor generation

- Materials datasets with composition fields were converted mainly with Magpie-style composition descriptors.
- Materials datasets with crystal structures or structure-derived fields additionally contain local-environment and geometry descriptors. `3DSC_TC.csv` includes SOAP descriptors; `3DSC_TC_nosoap.csv` keeps the corresponding non-SOAP feature set.
- Molecular datasets with SMILES, such as QM9 and melting points, use fixed-length molecular fingerprint descriptors.
- OMol is represented by 128-dimensional latent outputs extracted from a universal machine-learning interatomic potential (universal MLIP), rather than by direct Magpie, SOAP, or SMILES fingerprints.
- Some files combine raw collected quantities with generated descriptors; see the column names and target prefixes in each CSV.


## Files

| File | Samples | Columns in archive | Benchmark type | Main target columns / objectives | Literature/source basis |
|---|---:|---:|---|---|---|
| `HEA_exp_target_Hardness.csv` | 635 | 147 | Exp-S | `target_Hardness` | Curated experimental HEA hardness benchmark used in the study. |
| `curie_temperature.csv` | 1,749 | 142 | Exp-S | `target_Tc_real` | Ferromagnetic-material Curie-temperature dataset from Long et al. (2021). |
| `AM.csv` | 2,167 | 11 | Exp-M | `target_relative_density` | Curated additive-manufacturing process/property benchmark used in the study. |
| `steel_neatened.csv` | 2,180 | 137 | Exp-M | `target_0.2% Proof stress`, `target_Ultimate tensile strength` | Matbench/Automatminer steel benchmark source from Dunn et al. (2020). |
| `melting_points.csv` | 3,025 | 1,026 | Exp-M | `target_melting_point` | Jean-Claude Bradley curated melting-point dataset. |
| `exp_bandgap.csv` | 6,030 | 134 | Exp-M | `target_Eg (eV)` | Inorganic band-gap dataset from Zhuo, Mansouri Tehrani, and Brgoch (2018). |
| `3DSC_TC_nosoap.csv` | 5,773 | 153 | ExpSim-M | `target_tc`, `target_e_above_hull_2`, `target_band_gap_2`, `target_true_total_magnetization_2` | 3DSC-A superconductors dataset from Sommer et al. (2023), without SOAP descriptors. |
| `3DSC_TC.csv` | 5,773 | 8,868 | ExpSim-M | `target_tc`, `target_e_above_hull_2`, `target_band_gap_2`, `target_true_total_magnetization_2` | 3DSC-A superconductors dataset from Sommer et al. (2023), with SOAP descriptors. |
| `dielectric.csv` | 4,762 | 273 | Sim-M | `target_n` | High-throughput dielectric/optical materials data from Petousis et al. (2017). |
| `optimade_2dmatpedia.csv` | 6,351 | 276 | Sim-M | `target__twodmatpedia_band_gap` | 2DMatPedia dataset from Zhou et al. (2019), accessed/processed through OPTIMADE-style metadata. |
| `bgfEmag.csv` | 52,613 | 276 | Sim-L | `target_band_gap`, `target_minus-formation_energy_per_atom`, `target_magnetization` | Materials Project-derived benchmark based on Jain et al. (2013). |
| `powerfactor2.csv` | 94,250 | 30 | Sim-L | `target_Power Factor` | Thermoelectric power-factor benchmark from Lim et al. (2021). |
| `QM9_dataset.csv` | 133,139 | 1,029 | Sim-L | `target_isotropic polarizability`, `target_energy gap`, `target_heat capacity` | QM9 molecular dataset from Ramakrishnan et al. (2014). |
| `alex_icams.csv` | 415,412 | 279 | Sim-L | `target__alexandria_scan_hull_distance`, `target__alexandria_scan_magnetization`, `target__alexandria_scan_band_gap` | Alexandria/ICAMS large materials dataset from Schmidt et al. (2024). |
| `omol_4M_gnn_single.csv` | 3,986,754 | 130 | Sim-L | `target_homo_lumo_gap` | OMol25 molecular dataset from Levine et al. (2025), represented with 128-dimensional descriptors. |

## Usage with the code repository

The repository provides command-line entry points for closed-pool optimization:

```bash
conda activate Bo_project
python main.py data-closedpool
python main.py data-target-window
```

The generic closed-pool runner expects CSV files with target columns named using the `target_` prefix. The code repository also contains the full Bayesian optimization implementation, acquisition functions, surrogate models, sampling routines, and documentation.

## Provenance and citation

The source literature and data resources used to assemble the benchmark suite are listed below. Please cite the corresponding original dataset papers/resources when using individual datasets, and cite the associated Bayesian optimization code/paper when using this archive as a benchmark suite.

- Long, T., Fortunato, N. M., Zhang, Y., Gutfleisch, O. & Zhang, H. An accelerating approach of designing ferromagnetic materials via machine learning modeling of magnetic ground state and Curie temperature. Materials Research Letters 9, 169-174 (2021).
- Dunn, A., Wang, Q., Ganose, A., Dopp, D. & Jain, A. Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm. npj Computational Materials 6, 138 (2020).
- Bradley, J., Lang, A. & Williams, A. Jean-Claude Bradley Double Plus Good highly curated and validated melting point dataset (2014).
- Zhuo, Y., Mansouri Tehrani, A. & Brgoch, J. Predicting the band gaps of inorganic solids by machine learning. Journal of Physical Chemistry Letters 9, 1668-1673 (2018).
- Sommer, T., Willa, R., Schmalian, J. & Friederich, P. 3DSC-A dataset of superconductors including crystal structures. Scientific Data 10, 816 (2023).
- Petousis, I. et al. High-throughput screening of inorganic compounds for the discovery of novel dielectric and optical materials. Scientific Data 4, 1-12 (2017).
- Zhou, J. et al. 2DMatPedia, an open computational database of two-dimensional materials from top-down and bottom-up approaches. Scientific Data 6, 86 (2019).
- Jain, A. et al. Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. APL Materials 1 (2013).
- Lim, Y.-F., Ng, C. K., Vaitesswar, U. & Hippalgaonkar, K. Extrapolative Bayesian optimization with Gaussian process and neural network ensemble surrogate models. Advanced Intelligent Systems 3, 2100101 (2021).
- Ramakrishnan, R., Dral, P. O., Rupp, M. & von Lilienfeld, O. A. Quantum chemistry structures and properties of 134 kilo molecules. Scientific Data 1, 1-7 (2014).
- Schmidt, J. et al. Improving machine-learning models in materials science through large datasets. Materials Today Physics 48, 101560 (2024).
- Levine, D. S. et al. The Open Molecules 2025 (OMol25) dataset, evaluations, and models. arXiv:2505.08762 (2025).

## License and reuse note

This archive aggregates and featurizes datasets from multiple literature and database sources. The current Zenodo metadata is intentionally prepared for a restricted draft, not for immediate public publication. Before any public release, the depositor should verify that redistribution of each processed CSV is consistent with the corresponding source dataset licenses and terms.
