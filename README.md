# Reference Implementation for Algorithm 3

This supplementary material accompanies the paper

> **Bivariate Survival Analysis for Truncated and Censored Astronomical Data**
>
> *Applications to Galaxy Luminosity Functions*

by Tsutomu T. Takeuchi.

The paper develops **Algorithm 3**, a unified nonparametric estimator for
bivariate luminosity functions under simultaneous truncation and one-sided
censoring.

This supplementary material provides two Python implementations of
Algorithm 3.

---

# Repository structure

```
README.md
LICENSE
requirements.txt

bivariate_survival_reference.py
bivariate_survival_data_io.py

example/
    example_input.csv
    README_example.md
```

---

# Implementations

## 1. bivariate_survival_reference.py

This is the **reference implementation of Algorithm 3** described in the
paper.

Its primary purpose is reproducibility.

The implementation follows the mathematical formulation presented in
Sections 3.6--3.8 equation by equation.

It includes

- internal validation,
- brute-force verification,
- numerical diagnostics,
- figure generation,

and preserves a one-to-one correspondence between the mathematical
definitions and the implementation.

This version should be regarded as the implementation corresponding
directly to the published algorithm.

---

## 2. bivariate_survival_data_io.py

This is the **practical implementation of Algorithm 3** for user-supplied
astronomical catalogs.

Compared with the reference implementation,

- mock-data generation has been removed,
- validation routines have been removed,
- only external data input/output is retained,

while the estimator itself is identical.

This version is intended for practical applications to observational data.

---

# Relationship between the paper and the code

The mathematical definitions introduced in the paper correspond directly to
the implementation.

| Paper | Python implementation |
|--------|-----------------------|
| Algorithm 3 | `fit_algorithm3()` |
| Eq. (weighted_bivariate_risk) | weighted risk set |
| Eqs. (weighted_N11)--(weighted_N01) | weighted counting processes |
| Eqs. (weighted_DeltaLambda11)--(weighted_DeltaLambda01) | weighted hazard increments |
| Eq. (weighted_gamma) | Dąbrowska correction |
| Eq. (weighted_dabrowska_estimator) | product integral |
| Eq. (tc_luminosity_cdf) | final luminosity-space CDF |

Both Python files implement the same estimator presented as
**Algorithm 3** in the manuscript.

The reference implementation additionally contains validation procedures,
whereas the data-I/O implementation is intended for routine scientific use.

---

# Input format

The input catalog must contain the following columns.

| Column | Description |
|---------|-------------|
| `x1_obs` | observed logarithmic luminosity or upper limit in band 1 |
| `x2_obs` | observed logarithmic luminosity or upper limit in band 2 |
| `y1` | logarithmic detection limit in band 1 |
| `y2` | logarithmic detection limit in band 2 |
| `delta1` | detection indicator (1 = detected) |
| `delta2` | detection indicator (1 = detected) |
| `region` | observational region (`A`, `B`, `C`, or `D`) |

Algorithm 3 uses Regions A, B, and C explicitly.
The missing Region D is treated statistically through inverse
observation-probability weighting.

---

# Example

```
python bivariate_survival_data_io.py \
    --input example/example_input.csv \
    --outdir output
```

---

# Design philosophy

The implementation follows five principles.

1. The manuscript serves as the formal specification.

2. Every mathematical quantity appearing in Algorithm 3 has a direct
   implementation.

3. The jump set is exactly

   \[
   J_n = U_n \times V_n,
   \]

   as defined in the manuscript.

4. Numerical validation is separated from practical data analysis.

5. Readability and reproducibility are preferred over computational
   optimization.

---

# Citation

If these programs are used in published work, please cite the accompanying
paper.

---

# License

See `LICENSE`.
