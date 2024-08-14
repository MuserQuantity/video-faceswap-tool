from utils.face_swap_utils import read_video_to_frames, read_json_data, face_swap
import imageio.v2 as iio

if __name__ == '__main__':
    # inference
    inference_video_path = "inference.mp4"
    inference_video_landmarks_path = "inference.mp4.txt"
    inference_video_frames = read_video_to_frames(inference_video_path)
    inference_video_landmarks = read_json_data(inference_video_landmarks_path)
    # mouth
    mouth_video_path = "mouth.mp4"
    mouth_video_landmarks_path = "mouth.mp4.txt"
    mouth_video_frames = read_video_to_frames(mouth_video_path)
    mouth_video_landmarks = read_json_data(mouth_video_landmarks_path)
    # output
    output_video = "output.mp4"
    w = iio.get_writer(
        output_video,
        format='FFMPEG',
        fps=25,
        codec='libx264',
        quality=7,
        macro_block_size=None,
        ffmpeg_params=['-preset', 'medium', '-crf', '23']
    )
    min_frames = min(len(inference_video_frames), len(mouth_video_frames))
    for i in range(min_frames):
        inference_video_frame = inference_video_frames[i]
        mouth_video_frame = mouth_video_frames[i]
        inference_video_landmark = inference_video_landmarks[i]
        mouth_video_landmark = mouth_video_landmarks[i]
        result = face_swap(mouth_video_frame, inference_video_frame, mouth_video_landmark, inference_video_landmark)
        w.append_data(result)
    w.close()
