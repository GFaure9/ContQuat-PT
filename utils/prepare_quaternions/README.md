# Preparing quaternion-based pose data from 3D cartesian joints coordinates data


[//]: # (## 🔃 Skeletal Representations)

[//]: # ()
[//]: # (Helper tools to **write skeletal poses with different formalisms** and **go from one to another** effortlessly.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (### Classical absolute cartesian representation)

[//]: # ()
[//]: # ($$ \textbf{SkelPose}^{cart} := &#40;&#40;x_k, y_k, z_k&#41;&#41;_{1 \leq k \leq N} =: &#40;X_k&#41;_k \in \mathbb{R}^{N \times 3} $$)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (### Quaternion-based representation)

[//]: # ()
[//]: # ($$ \textbf{SkelPose}^{quart} := )

[//]: # (&#40;&#40;\cos&#40;\frac{\theta}{2}&#41;, \sin&#40;\frac{\theta}{2}&#41; u_x^{&#40;i&#41;}, \sin&#40;\frac{\theta}{2}&#41; u_y^{&#40;i&#41;}, \sin&#40;\frac{\theta}{2}&#41; u_z^{&#40;i&#41;}, l_i&#41;&#41;_{1 \leq i \leq M} $$)

[//]: # ($$ =: &#40;q^{&#40;i&#41;}_1, q^{&#40;i&#41;}_2, q^{&#40;i&#41;}_3, q^{&#40;i&#41;}_4, l_i&#41;_i )

[//]: # (\in \mathbb{R}^{M \times 5} $$)

[//]: # ($$ \text{with } \| u^{&#40;i&#41;} \| = 1 $$)

[//]: # ($$ \text{and } X_{root} \in \mathbb{R}^3 $$)

[//]: # ($$ \text{and } G_{Skel} := &#40;V, E&#41; $$)

[//]: # ($$ \text{where } V := \{ 1, 2, ..., N \},~ E := \{ &#40;V_{Parent_i}, V_{Child_i}&#41;, 1 \leq i \leq M \} $$)

[//]: # ()
[//]: # (Given the graph structure $G_{Skel}$ of the skeleton as a tree, )

[//]: # (and a bind pose $&#40;X_k^0&#41;_k$ &#40;e.g. T-pose&#41;, one can obtain the quaternions-based representation)

[//]: # (of the pose from the cartesian 3D coordinates by computing, recursively, starting from the root node:)

[//]: # ()
[//]: # ($$)

[//]: # (q^{&#40;i&#41;} = &#40;\cos&#40;\frac{\theta^{&#40;i&#41;}}{2}&#41;, \sin&#40;\frac{\theta^{&#40;i&#41;}}{2}&#41; u^{&#40;i&#41;}&#41;~ \text{where}~\begin{cases})

[//]: # (v^{&#40;i&#41;} = \frac{X_{Child_i} - X_{Parent_i}}{\| X_{Child_i} - X_{Parent_i} \|}, ~  v^{&#40;i&#41;}_0 )

[//]: # (= \frac{X_{Child_i}^0 - X_{Parent_i}^0}{\| X_{Child_i}^0 - X_{Parent_i}^0 \|} \\)

[//]: # (u^{&#40;i&#41;} = \frac{ v^{&#40;i&#41;} \wedge v^{&#40;i&#41;}_0 }{\| v^{&#40;i&#41;} \wedge v^{&#40;i&#41;}_0 \|} \\)

[//]: # (\theta^{&#40;i&#41;} = \arccos&#40;v^{&#40;i&#41;} \cdot v^{&#40;i&#41;}_0&#41;)

[//]: # (\end{cases})

[//]: # ($$)

[//]: # ()
[//]: # (Inversely, to obtain cartesian $&#40;x, y, z&#41;$ coordinates from quaternions, we will compute, recursively from the root node:)

[//]: # ()
[//]: # ($$ &#40;0, X_k&#41; = &#40;0, X_{\text{Parent of }k}&#41; + l_{i_k} \left&#40; q^{&#40;i_k&#41;} &#40;0, v^{&#40;i_k&#41;}_0&#41; {q^{&#40;i_k&#41;}}^{-1} \right&#41;$$)

[//]: # ()
[//]: # (#### *Distance between quaternions &#40;the geodesic norm&#41;*)

[//]: # ()
[//]: # (The distance between two quaternions $q$ and $q'$ can be expressed through the geodesic distance as:)

[//]: # ()
[//]: # ($$ d_g&#40;q, q'&#41; := | ln&#40;q^{-1} q'&#41; | $$)

[//]: # ()
[//]: # (where $q^{-1} = \frac{\overline{q}}{\| q \| ^2}$ with $\overline{q} = &#40;\cos&#40;\frac{\theta}{2}&#41;, -\sin&#40;\frac{\theta}{2}&#41; u&#41;$)

[//]: # (the conjugate of $q$.)

[//]: # ()
[//]: # (Or:)

[//]: # ()
[//]: # ($$ d_g&#40;q, q'&#41; := \arccos&#40; 2 &#40;q q'&#41;^2 - 1&#41; $$)

[//]: # ()
[//]: # (**Sources**: *geodesic norm* section in the [Quaternion Wiki page]&#40;https://en.wikipedia.org/wiki/Quaternion&#41;.)

[//]: # ()
[//]: # (## 🚹 Skeleton Structures)

[//]: # ()
[//]: # (*Helpers for common skeleton structures in SLP.*)

[//]: # ()
[//]: # (Typically, these structures will be **written in a graph-like structure** as follows.)

[//]: # (If we want to represent the following structure:)

[//]: # ()
[//]: # (```)

[//]: # (                    &#40;1&#41;)

[//]: # (                     |)

[//]: # (                    {1})

[//]: # (                     |)

[//]: # (     &#40;3&#41; ---{2}---- &#40;2&#41; ---{2}---- &#40;4&#41;)

[//]: # (```)

[//]: # ()
[//]: # (where `{i}` is the bone level number and `&#40;j&#41;` is the node ID, we will write:)

[//]: # ()
[//]: # (```python)

[//]: # (SKEL = &#40;)

[//]: # (    &#40;1, 2, 1&#41;,  # &#40;ParentJointID, ChildJointID, BoneID&#41;)

[//]: # (    &#40;2, 3, 2&#41;,)

[//]: # (    &#40;2, 4, 2&#41;,)

[//]: # (&#41;)

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (### Usage)

[//]: # ()
[//]: # (Import a structure as follows:)

[//]: # ()
[//]: # (```python)

[//]: # (from utils.skeleletal_structures_helper import YOUR_SKEL_STRUCTURE_NAME)

[//]: # (```)

[//]: # ()
[//]: # (Possible values for `YOUR_SKEL_STRUCTURE_NAME` are listed below.)

[//]: # ()
[//]: # (---)