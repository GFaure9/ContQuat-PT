# Preparing quaternion-based pose data from 3D cartesian joints coordinates data

Once the `.skels` files have been computed for a given dataset, `.quat` files can be computed following the
steps of this Python code:

```python
# -- Imports
from . import load_skel_sequences, cart_to_quat, write_quat_file

# -- Global variable
SUBSET="name_of_the_subset"  # 'train', 'test', 'dev', etc.
DATASET_FOLDER="path/to/your/dataset/folder"  # folder containing 'train.skels', 'test.skels' and 'dev.skels' files

# -- 1) Loading cartesian coordinates of joints (skeletal pose sequences)
print(f"Starting loading {SUBSET}.skels file...")
skel_sequences = load_skel_sequences(DATASET_FOLDER, SUBSET)  # N.B: it removes the counter
print(f"Loaded {SUBSET}.skels file!")

# -- 2) Computing quaternions sequences from joints coordinates
print(f"Starting computing quaternions for each skeletal sequence...")
root_pts_sequences, quat_sequences = cart_to_quat(skel_sequences)  # skel_structure is `ORIGINAL_S2SL_SKEL` by default
print("Finished computing quaternions for all sequences!")

# -- 3) Writing computed quaternion-based encodings of skeletal pose sequences to DATASET_FOLDER/SUBSET.quat
print(f"Starting writing quaternions to: {DATASET_FOLDER}/{SUBSET}.quat")
write_quat_file(
    quat_sequences,
    DATASET_FOLDER,
    SUBSET,
    with_counter=True,  # last value of each skeleton quaternion-based pose is the corresponding counter value t/T
    root_points_sequences=root_pts_sequences
)
print(f"Finished writing quaternions to: {DATASET_FOLDER}/{SUBSET}.quat")
```

---

### Quaternions formalism and computation: some insights

While the classical representation of a skeletal pose using 3D cartesian coordinates
of joints can be written as follows:

$$ \textbf{SkelPose}^{cart} := ((x_k, y_k, z_k))_{1 \leq k \leq N} =: (X_k)_k \in \mathbb{R}^{N \times 3} $$

the quaternion-based representation is expressed as:

$$ \textbf{SkelPose}^{quart} := 
((\cos(\frac{\theta}{2}), \sin(\frac{\theta}{2}) u_x^{(i)}, \sin(\frac{\theta}{2}) u_y^{(i)}, \sin(\frac{\theta}{2}) u_z^{(i)}, l_i))_{1 \leq i \leq M} $$
$$ =: (q^{(i)}_1, q^{(i)}_2, q^{(i)}_3, q^{(i)}_4, l_i)_i 
\in \mathbb{R}^{M \times 5} $$
$$ \text{with } \| u^{(i)} \| = 1 $$
$$ \text{and } X_{root} \in \mathbb{R}^3 $$
$$ \text{and } G_{Skel} := (V, E) $$
$$ \text{where } V := \{ 1, 2, ..., N \},~ E := \{ (V_{Parent_i}, V_{Child_i}), 1 \leq i \leq M \} $$

Note that from the graph structure $G_{Skel}$ of the skeleton as a tree, 
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

> [!NOTE]
> In our code, the skeletal graph structure will typically be defined through a tuple of triplets 
> `(Node ID Parent, Node ID Child, Node's Depth)`.
> For example, `((1, 2, 1), (2, 3, 2), (2, 4, 2)` describes the following very simple skeletal graph structure:
```
                    (1)
                     |
                    {1}
                     |
     (3) ---{2}---- (2) ---{2}---- (4)
```

Inversely, to obtain cartesian $(x, y, z)$ coordinates from quaternions, we will compute, recursively from the root node:

$$ (0, X_k) = (0, X_{\text{Parent of }k}) + l_{i_k} \left( q^{(i_k)} (0, v^{(i_k)}_0) {q^{(i_k)}}^{-1} \right)$$
