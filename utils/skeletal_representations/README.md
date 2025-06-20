## 🔃 Skeletal Representations

Helper tools to **write skeletal poses with different formalisms** and **go from one to another** effortlessly.

---

### Classical absolute cartesian representation

$$ \textbf{SkelPose}^{cart} := ((x_k, y_k, z_k))_{1 \leq k \leq N} =: (X_k)_k \in \mathbb{R}^{N \times 3} $$

---

### Quaternion-based representation

$$ \textbf{SkelPose}^{quart} := 
((\cos(\frac{\theta}{2}), \sin(\frac{\theta}{2}) u_x^{(i)}, \sin(\frac{\theta}{2}) u_y^{(i)}, \sin(\frac{\theta}{2}) u_z^{(i)}, l_i))_{1 \leq i \leq M} $$
$$ =: (q^{(i)}_1, q^{(i)}_2, q^{(i)}_3, q^{(i)}_4, l_i)_i 
\in \mathbb{R}^{M \times 5} $$
$$ \text{with } \| u^{(i)} \| = 1 $$
$$ \text{and } X_{root} \in \mathbb{R}^3 $$
$$ \text{and } G_{Skel} := (V, E) $$
$$ \text{where } V := \{ 1, 2, ..., N \},~ E := \{ (V_{Parent_i}, V_{Child_i}), 1 \leq i \leq M \} $$

Given the graph structure $G_{Skel}$ of the skeleton as a tree, 
and a bind pose $(X_k^0)_k$ (e.g. T-pose), one can obtain the quaternions-based representation
of the pose from the cartesian 3D coordinates by computing, recursively, starting from the root node:

$$
q^{(i)} = (\cos(\frac{\theta^{(i)}}{2}), \sin(\frac{\theta^{(i)}}{2}) u^{(i)})~ \text{where}~\begin{cases}
v^{(i)} = \frac{X_{Child_i} - X_{Parent_i}}{\| X_{Child_i} - X_{Parent_i} \|}, ~  v^{(i)}_0 
= \frac{X_{Child_i}^0 - X_{Parent_i}^0}{\| X_{Child_i}^0 - X_{Parent_i}^0 \|} \\
u^{(i)} = \frac{ v^{(i)} \wedge v^{(i)}_0 }{\| v^{(i)} \wedge v^{(i)}_0 \|} \\
\theta^{(i)} = \arccos(v^{(i)} \cdot v^{(i)}_0)
\end{cases}
$$

Inversely, to obtain cartesian $(x, y, z)$ coordinates from quaternions, we will compute, recursively from the root node:

$$ (0, X_k) = (0, X_{\text{Parent of }k}) + l_{i_k} \left( q^{(i_k)} (0, v^{(i_k)}_0) {q^{(i_k)}}^{-1} \right)$$

#### *Distance between quaternions (the geodesic norm)*

The distance between two quaternions $q$ and $q'$ can be expressed through the geodesic distance as:

$$ d_g(q, q') := | ln(q^{-1} q') | $$

where $q^{-1} = \frac{\overline{q}}{\| q \| ^2}$ with $\overline{q} = (\cos(\frac{\theta}{2}), -\sin(\frac{\theta}{2}) u)$
the conjugate of $q$.

Or:

$$ d_g(q, q') := \arccos( 2 (q q')^2 - 1) $$

**Sources**: *geodesic norm* section in the [Quaternion Wiki page](https://en.wikipedia.org/wiki/Quaternion).