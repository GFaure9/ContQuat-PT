## 🚹 Skeleton Structures

*Helpers for common skeleton structures in SLP.*

Typically, these structures will be **written in a graph-like structure** as follows.
If we want to represent the following structure:

```
                    (1)
                     |
                    {1}
                     |
     (3) ---{2}---- (2) ---{3}---- (4)
```

where `{i}` is the bone ID number and `(j)` is the node ID, we will write:

```python
SKEL = (
    (1, 2, 1),  # (ParentJointID, ChildJointID, BoneID)
    (2, 3, 2),
    (2, 4, 3),
)
```

---

### Usage

Import a structure as follows:

```python
from utils.skeleletal_structures_helper import YOUR_SKEL_STRUCTURE_NAME
```

Possible values for `YOUR_SKEL_STRUCTURE_NAME` are listed below.

---

### Available Structures

- `ORIGINAL_S2SL_SKEL`: 8 body joints + 2x21 hands joints (COCO format). Cf `original_s2sl_skel_structure.py`
- `ORIGINAL_S2SL_SKEL_INVERTED_HANDS`: same as above but inverting right and left body->hands wrists connexions
- *More coming in the future!*