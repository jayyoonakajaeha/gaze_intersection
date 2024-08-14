# gaze_intersection

you need to make .json file using openpose

code/inference.py is the main file

Used GazeFollowing by https://github.com/svip-lab/GazeFollowing

# how to use
First, make skeleton detecting json file by using openpose

Second, make frame images of your target video by using frame_extractor.py

Third, set directories of skeleton detecting json file and frame images on inference.py

Lastly, run inference.py then you will get log file and result images
