# Guide: Recover a missing term from partial physics

Use this workflow when the equation family is known but one or more terms are missing.

1. Write the trusted terms and coefficients explicitly.
2. Define a finite candidate library for the unexplained term. Do not include arbitrary high-order terms without physical reason.
3. Split trajectories by experiment/trajectory, not by random time rows.
4. Run `recover-missing`.
5. Check support stability across trajectories/seeds.
6. Reinsert the recovered term into the full mechanism and validate trajectories.
7. Certify coverage; if residuals stay systematically above the normal numerical/measurement floor, treat the result as incomplete.

A low validation error is not sufficient proof of a physical mechanism if sources are collinear or the prior is inconsistent.
