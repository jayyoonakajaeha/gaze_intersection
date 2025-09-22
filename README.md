# 👀 Gaze Estimation using GazeFollowing Model

This project implements a deep learning-based gaze estimation system using the pre-trained [GazeFollowing](https://github.com/svip-lab/GazeFollowing) model.

The goal is to predict the direction of a person's gaze in a video and visualize the results as image files.

## ⚙️ How it Works

The gaze estimation pipeline consists of three main stages:

1.  **Skeleton Detection**

      * We use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to detect the body skeleton of the person in the video. This information is extracted and saved as `.json` files for each frame.

2.  **Frame Extraction**

      * The target video is broken down into individual image frames, which will be used as inputs for the model.

3.  **Gaze Inference**

      * The skeleton data and image frames are fed into the GazeFollowing model.
      * The model predicts the gaze direction based on the person's head position and posture in each frame and saves the output as new images with the gaze visualized.

## 🚀 Getting Started

### Prerequisites

1.  **OpenPose**: You must have OpenPose installed to extract the necessary skeleton data. For installation instructions, please refer to the [official OpenPose repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
2.  **Python and Libraries**:
      * A Python 3.x environment is required.
      * Install the necessary libraries, including PyTorch and OpenCV.
    <!-- end list -->
    ```bash
    pip install torch torchvision opencv-python numpy
    ```

### Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/jayyoonakajaeha/gaze_estimation.git
cd gaze_estimation
```

-----

## 📖 How to Use

Follow these steps in order to get the final gaze estimation results.

### **Step 1: Generate JSON Files with OpenPose**

First, you need to run OpenPose on your target video to generate `.json` files containing the skeleton data for each frame.

  * When running OpenPose, use the `--write_json` flag to specify an output directory.

<!-- end list -->

```bash
# Example OpenPose command (modify paths to match your environment)
./build/examples/openpose/openpose.bin --video path/to/your/video.mp4 --write_json path/to/output/json_files/ --display 0 --render_pose 0
```

### **Step 2: Extract Video Frames**

Next, use the `frame_extract.py` script to split your source video into image frames.

  * `--video_path`: Path to your source video file.
  * `--save_path`: Path to the directory where the extracted frames will be saved.

<!-- end list -->

```bash
python frame_extract.py --video_path path/to/your/video.mp4 --save_path path/to/output/frames/
```

### **Step 3: Configure Paths in `inference.py`**

Now, open the main script `code/inference.py` and set the paths to the data you generated in the previous steps.

  * Edit the following two variables with your actual directory paths:

<!-- end list -->

```python
# Inside code/inference.py

def main():
    # ...
    # Modify this path to your JSON output folder from Step 1
    json_folder = "path/to/output/json_files"

    # Modify this path to your frames output folder from Step 2
    img_folder = "path/to/output/frames"
    # ...
```

### **Step 4: Run Gaze Inference**

You are all set\! Run the `inference.py` script to start the gaze estimation process.

```bash
python code/inference.py
```

### **Check Your Results**

After the script finishes, the following outputs will be generated in the project directory:

  * **`log/log.txt`**: A log file containing detailed results, including the predicted gaze coordinates.
  * **`results/`**: A directory containing the output images, with the predicted gaze direction visualized on each frame.

## 📚 References

  * **OpenPose**: [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
  * **GazeFollowing Model**: [svip-lab/GazeFollowing](https://github.com/svip-lab/GazeFollowing)
