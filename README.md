# Custom Useful Utils

A small collection of Python utilities and image/video processing helpers.

## Repository Structure

- `custom_utils.py`
  - Contains `imread_custom()` for robust OpenCV image loading from paths with non-ASCII characters and Windows long-path support.
- `GitHub_Utils/`
  - `followers_check.py`
    - Fetches GitHub followers using the GitHub API and saves usernames to `followers/followersYYYY-MM-DD.json`.
  - `followings_check.py`
    - Fetches GitHub users you are following and saves usernames to `following/followingYYYY-MM-DD.json`.
  - `compare_follow.py`
    - Compares two follower snapshot JSON files and prints added and removed usernames.
  - `followers/` and `following/`
    - Directories to store GitHub follower/following snapshot files.
- `Img_Vid_manipulations_obj_tracking/`
  - `cv_classes.py`
    - Provides OpenCV-based filters, blemish removal, document scanner utilities, mouse interaction helpers, and a YOLO-based ball tracker.

## Key Features

### GitHub utilities
- Save GitHub follower and following lists to dated JSON files.
- Compare follower snapshots between two dates.
- Supports authenticated API requests via environment variables.

### Image and video utilities
- `imread_custom()` for safe OpenCV image reads from complex file paths.
- Real-time or static filters:
  - Cartoon
  - Cartoon stylized
  - Pencil sketch
  - Skin smoothing
  - Sunglasses overlay with optional reflection
- Blemish removal using seamless cloning or inpainting.
- Document scanner with automatic contour detection and perspective transformation.
- Ball tracker using YOLO and OpenCV tracker fallbacks.

## Usage

### GitHub utilities
1. Create a `.env` file at the repository root.
2. Add your GitHub username and optional token:

```dotenv
GITHUB_USERNAME=your_username
GITHUB_TOKEN=your_token
```

3. Run the followers or following script:

```bash
python GitHub_Utils/followers_check.py
python GitHub_Utils/followings_check.py
```

4. Compare snapshots:

```bash
python GitHub_Utils/compare_follow.py
```

### Image/video utilities
- Import `imread_custom` from `custom_utils` for robust image loading.
- Use `Img_Vid_manipulations_obj_tracking/cv_classes.py` to instantiate and apply filters or tracking.

Example:

```python
from Img_Vid_manipulations_obj_tracking.cv_classes import Filters

filters = Filters(glasses_path=None, reflection_path=None, source='webcam')
filters.start_filters(filter='cartoon')
```

## Dependencies

Recommended dependencies:

- `opencv-python`
- `numpy`
- `requests`
- `python-dotenv`
- `ultralytics` (for YOLO object detection)

Install with:

```bash
pip install opencv-python numpy requests python-dotenv ultralytics
```

> Note: `ultralytics` is only required for the YOLO-based ball tracker in `cv_classes.py`.

## Notes

- The GitHub utilities create date-specific JSON snapshots in `GitHub_Utils/followers/` and `GitHub_Utils/following/`.
- The image/video utilities expect valid paths or webcam sources and require OpenCV windows for display.
- Adjust the script file names and source paths as needed for your environment.
