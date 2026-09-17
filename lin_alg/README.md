# lin_alg

## Multiplication Convention

Matrix multiplication in this library is done left to right using row vectors rather than the typical mathematical convention of right to left using column vectors. This is done to better facilitate operators such as `*=`.

`A * T` means apply the transformation $T$ to $A$.

> Because vectors are rows, the basis vectors of a matrix are rows.

## Matrix Indexing

- `nr()` / `nc()`: number of rows / columns
- `at(i, j)`: row `i`, column `j` (0-based)
- Storage is row-major: element `(i, j)` is at raw index `i*nc() + j`
