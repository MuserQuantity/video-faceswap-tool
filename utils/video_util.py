import os
import subprocess
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, clips_array


def read_frame_from_video(video_path, frame_number):
    # load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: cannot open video file.")
        return None
    # set current frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    # close video
    cap.release()
    if ret:
        return frame
    else:
        print(f"Error: cannot read frame {frame_number}.")
        return None


# merge video and audio
def merge_video_audio(video_path, audio_path, output_path):
    # check video format
    if not (
            video_path.endswith(".mp4")
            or video_path.endswith(".avi")
            or video_path.endswith(".mkv")
    ):
        print("Invalid video format. Supported formats are mp4, avi, mkv.")
        return
    if not (audio_path.endswith(".mp3") or audio_path.endswith(".wav")):
        print("Invalid audio format. Supported formats are mp3, wav.")
        return

    # merge video and audio
    command = f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac -b:a 5000k "{output_path}"'
    try:
        subprocess.run(command, shell=True, check=True)
        print("success")
    except subprocess.CalledProcessError as e:
        print("failed:", e)


def extract_video_stream(video_path, bitrate="5000k", resize_width=None, resize_height=None):
    # get base name and extension
    base_name, extension = os.path.splitext(video_path)
    # rename old video
    old_video_path = f"{base_name}_old{extension}"
    os.rename(video_path, old_video_path)
    # set output path
    output_path = base_name + ".mp4"
    # create video clip without audio
    video = VideoFileClip(old_video_path).without_audio()
    # resize video
    if resize_width and resize_height:
        if video.size[0] > resize_width and video.size[1] > resize_width:
            video = video.resize(height=resize_height, width=resize_width)
    # set output params
    output_params = ["-vcodec", "h264_nvenc", "-preset", "slow", "-crf", "23"]
    video.write_videofile(output_path, codec="libx264", bitrate=bitrate, ffmpeg_params=output_params)

    print(f"Processed video saved as {output_path}")


def merge_video_with_alpha(video_path, alpha_path, output_path, bitrate="5000k"):
    # load videos
    inference_video = VideoFileClip(video_path)
    alpha_video = VideoFileClip(alpha_path)

    # check video size
    target_height = min(inference_video.h, alpha_video.h)
    inference_video = inference_video.resize(height=target_height)
    alpha_video = alpha_video.resize(height=target_height)

    # create final clip
    final_clip = clips_array([[inference_video, alpha_video]]).without_audio()
    # set output params
    output_params = ["-vcodec", "h264_nvenc", "-preset", "slow", "-crf", "23"]
    final_clip.write_videofile(output_path, codec="libx264", bitrate=bitrate, ffmpeg_params=output_params)

    # 关闭视频文件
    inference_video.close()
    alpha_video.close()


if __name__ == "__main__":
    # video_path1 = "videos/video1.mov"
    # video_path2 = "videos/video2.mov"
    # merge_video_with_alpha(video_path1, video_path2, "videos/output.mp4")
    video_path = "videos/output.mov"
    extract_video_stream(video_path, bitrate="2000k")
