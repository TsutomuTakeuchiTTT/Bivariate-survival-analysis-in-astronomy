# Example input catalog

This file illustrates the input format required by
`bivariate_survival_data_io.py`.
The example catalog example_input.csv was generated using the same mock-data model employed in the numerical experiments presented in Section 4 of the accompanying paper. The file contains the first 20 observed objects (Regions A, B, and C) after applying the simulated observational selection. Its purpose is solely to illustrate the required input format for Algorithm 3.

Each row corresponds to one astronomical object.

Columns are

| Column | Description |
|---------|-------------|
| x1_obs | observed logarithmic luminosity or censoring limit in band 1 |
| x2_obs | observed logarithmic luminosity or censoring limit in band 2 |
| y1 | logarithmic detection limit in band 1 |
| y2 | logarithmic detection limit in band 2 |
| delta1 | detection indicator (1 = detected, 0 = nondetection) |
| delta2 | detection indicator (1 = detected, 0 = nondetection) |
| region | observational region (A, B, or C) |

For nondetections,

- `x_obs` stores the detection limit,
- `delta = 0`,
- `region` is either B or C.

Objects in Region D do not appear in the input catalog because they are not observed.
Algorithm 3 accounts for Region D statistically through inverse observation-probability weighting.
