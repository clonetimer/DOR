# Physics-aware search reduction

Use this when an algebraic/symbolic problem has many raw variables or obvious engineering units.

The reducer can:

- enumerate dimensionally consistent low-order monomials;
- test pairwise `difference`, `sum`, `ratio`, or `product` coordinates;
- retain a coordinate only when held-out error plus expression-complexity improves.

Important: dimensional screening assumes the remaining multiplicative coefficient is dimensionless. If the true law contains an omitted dimensional constant, treat the list as advisory rather than a hard exclusion set.

A proposed coordinate is not a mechanism certificate. Feed the reduced variables to an independent recovery backend and validate the final equation.
