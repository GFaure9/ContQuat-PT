### 📊🔧 Visualization tools

A toolbox to help generating visuals related to the Sign Language Production task.

Implemented features are:

- **poses / videos**
  - `make_skel_video`: create a video from a sequence of skeletal poses with given skeletal structure
  (with options to add an image and a text)
  - `stack_videos`: make a video from two videos putting them side-by-side

- **spectrograms**
  - `make_mel_spec_image`: create a 800x800 image from a numpy array Mel spectrogram of shape $(Nframes, Nmels)$

- **statistics**
  - `make_box_plots`: create box-plot(s) from data (cf. implementation for details)
  - `make_histograms`: create histogram(s) from data (cf. implementation for details)

---

#### How to use it?

To use the tools, simply import them from `utils.visualizations` as needed. E.g.:

```python
from utils.visualization import (
make_skel_video,
make_mel_spec_image,
stack_videos,
make_histograms,
make_box_plots,
)
```

or to import all:

```python
from utils.visualization import *
```

Have a look at `testing_tools.py` file to see a usage example of these functions.