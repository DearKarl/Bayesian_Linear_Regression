# Dataset Note

This repository uses the Boston Housing dataset as a legacy regression
benchmark because it is the dataset used in the original project report.

The benchmark should not be treated as a modern housing policy dataset. In
particular, the original `b` feature encodes a racial-composition transform and
is ethically problematic. The experiment runner therefore includes a sensitivity
check that compares the full legacy feature set with a screened feature set that
drops `b`.

For new applied work, prefer better documented and more modern datasets. This
repository keeps Boston Housing only for reproducibility, interpretability, and
comparison with the original Bayesian linear regression analysis.
