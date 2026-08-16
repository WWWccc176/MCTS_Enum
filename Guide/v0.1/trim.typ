= v0.1 Search-Space Tightening

The original search initializes the incumbent radius with the first basis
vector,

$
  R_0 = norm(b_1).
$

However, LLL does not guarantee that $b_1$ is the shortest vector already
present in the basis. Therefore the initial search radius can be unnecessarily
large.

The improved initialization scans the complete basis first:

$
  i^* = op("argmin")_i norm(b_i),
  quad
  R_0 = norm(b_(i^*)),
  quad
  z_0 = e_(i^*).
$

Thus MCTS starts from the best vector already known without spending search
nodes rediscovering an existing basis vector.

== Immediate Radius Tightening

For a partial coefficient assignment, legal actions satisfy

$
  rho_(k+1) + g_k (z_k - c_k)^2 < R^2.
$

Whenever an exact terminal vector improves the incumbent,

$
  norm(v) < R,
$

the global radius is updated immediately:

$
  R <- norm(v).
$

The legal intervals of all active nodes are then tightened using the new
radius. Branches that can no longer satisfy the exact GSO bound are disabled
immediately.

Therefore the search loop becomes

$
  "exact improvement"
  arrow
  "update R"
  arrow
  "tighten legal space"
  arrow
  "continue MCTS".
$

*Hint:* Facing with higher dimentions (such as $italic(dim) > 90$), one SV $->$ one R update might be too frequent (or no impact), the global update policy requires to be further derived and verified.
