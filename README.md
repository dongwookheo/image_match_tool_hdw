# matcher_hdw

Reusable image matching backend for stitching and localization experiments.

## Current MVP

- YAML matcher presets in `config/matchers/*.yaml`
- `SIFT + nearest neighbor + ratio test` sparse matching
- OpenCV RANSAC homography/fundamental estimation
- 4-view visualization: keypoints, raw matches, RANSAC matches, homography
- repo-local cache path at `.cache/models`

## Python API

```python
from matcher_hdw import ImageMatcher, load_builtin_config

config = load_builtin_config("sift_nn")
result = ImageMatcher(config).match(image0_bgr, image1_bgr)

print(result.ok, result.num_matches, result.num_inliers, result.inlier_ratio)
print(result.H_0_to_1)
```

## Examples

```bash
python examples/match_pair.py data/0000_left.png data/0000_right.png --show
python examples/match_pair_gui.py
```

The GUI uses DearPyGui. Initialize submodules and install the GUI extra before running it:

```bash
git submodule update --init --recursive
pip install -e '.[gui]'
```

## Cache

The default cache directory is `.cache/models`. Override it with `MATCHER_HDW_CACHE_DIR` or `runtime.cache_dir` in YAML.
