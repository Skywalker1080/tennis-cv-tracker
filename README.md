# Tennis CV Tracker

Tennis CV Tracker is a computer vision project for tracking tennis players and the ball in match videos. It detects players, tracks the ball, maps positions to a mini-court, and overlays stats and visualizations on the video.

## Features

- Player and ball detection and tracking
- Court line/keypoint detection
- Mini-court visualization
- Player and ball speed/statistics overlay
- Video input/output

## Requirements

- Python 3.8+
- [YOLOv8](https://github.com/ultralytics/ultralytics) (for player detection)
- OpenCV
- NumPy
- pandas
- PyTorch (for deep learning models)
- Other dependencies in `requirements.txt`

## Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/tennis-cv-tracker.git
    cd tennis-cv-tracker
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Download models:**
    - Place your YOLOv8 weights (e.g., `yolov8x.pt`) and ball/court models in the `models/` directory.

4. **Prepare input video:**
    - Place your input video in the `videos/` directory (e.g., `videos/input_video.mp4`).

## Running the Project

To process a video and generate the output with overlays:

```sh
python main.py
```

The output video will be saved in the `output_video/` directory.

## Project Structure

```
tennis-cv-tracker/
│
├── main.py                  # Main entry point
├── utils/                   # Utility functions (bbox, conversions, etc.)
├── trackers/                # Player and ball trackers
├── court_line_detector/     # Court line/keypoint detection
├── mini_court/              # Mini-court visualization logic
├── models/                  # Pretrained model files
├── videos/                  # Input videos
├── output_video/            # Output videos
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Notes

- Make sure your models are compatible with the code.
- Adjust paths in `main.py` if your directory structure is different.
- For best results, use high-quality, stable tennis match videos.

## License

MIT License

---

For questions or contributions, please open an issue or pull request!