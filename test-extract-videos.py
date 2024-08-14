from utils.video_util import extract_video_stream

if __name__ == "__main__":
    # video_path1 = "videos/video1.mov"
    # video_path2 = "videos/video2.mov"
    # merge_video_with_alpha(video_path1, video_path2, "videos/output.mp4")
    video_path = "inference.mov"
    extract_video_stream(video_path, bitrate="2000k")