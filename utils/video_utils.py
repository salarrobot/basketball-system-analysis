import os 
import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def save_video(frames, output_path):
    # make sure the output directory exists
    dir_name = os.path.dirname(output_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # set up the codec and VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, fourcc, 24, (w, h))

    # write all frames
    for frame in frames:
        out.write(frame)
    out.release()