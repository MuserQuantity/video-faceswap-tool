from utils.face_landmarker import *


def save_image_landmarks(img_path, save_path):
    landmarker = MediapipeLandmarker()
    img_lm478 = landmarker.extract_lm478_from_img_name(img_path)
    img_lm68 = img_lm478[index_lm68_from_lm478]
    with open(save_path, "w") as f:
        for pts in img_lm68.astype(int):
            f.write(f"{pts[0]} {pts[1]}\n")


def save_video_landmarks(video_path, save_path):
    landmarker = MediapipeLandmarker()
    ret = landmarker.extract_lm478_from_video_name(video_path)
    imgs, vids = ret
    video_landmarks = landmarker.combine_vid_img_lm478_to_lm68(imgs, vids)
    # json格式
    json_data = []
    for pts in video_landmarks.astype(int):
        json_data.append(pts.tolist())
    with open(save_path, "w") as f:
        json.dump(json_data, f)


if __name__ == '__main__':
    # save_image_landmarks("mouth.jpg", "mouth.jpg.txt")
    # save_image_landmarks("inference.jpg", "inference.jpg.txt")
    save_video_landmarks("mouth.mp4", "mouth.mp4.txt")
    save_video_landmarks("inference.mp4", "inference.mp4.txt")
