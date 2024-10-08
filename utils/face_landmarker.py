import json

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os
import copy
from tqdm import tqdm

# simplified mediapipe ldm at https://github.com/k-m-irfan/simplified_mediapipe_face_landmarks
index_lm141_from_lm478 = ([70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
                          + [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
                          + [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
                          + [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
                          + [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
                          + [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
                          + [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                             152, 148, 176, 149, 150, 136,
                             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                          + [468, 469, 470, 471, 472]
                          + [473, 474, 475, 476, 477]
                          + [64, 4, 294])
# lm141 without iris
index_lm131_from_lm478 = ([70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
                          + [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
                          + [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
                          + [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
                          + [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
                          + [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
                          + [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                             152, 148, 176, 149, 150, 136,
                             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                          + [64, 4, 294])

# face alignment lm68
index_lm68_from_lm478 = [127, 234, 93, 132, 58, 136, 150, 176, 152, 400, 379, 365, 288, 361, 323, 454, 356, 70, 63, 105,
                         66, 107, 336, 296, 334, 293,
                         300, 168, 197, 5, 4, 75, 97, 2, 326, 305, 33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373,
                         380, 61, 40, 37, 0, 267, 270,
                         291, 321, 314, 17, 84, 91, 78, 81, 13, 311, 308, 402, 14, 178]
# used for weights for key parts
unmatch_mask_from_lm478 = [93, 127, 132, 234, 323, 356, 361, 454]
index_eye_from_lm478 = ([33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
                        + [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249])
index_innerlip_from_lm478 = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
index_outerlip_from_lm478 = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
index_withinmouth_from_lm478 = ([76, 62]
                                + [184, 183, 74, 72, 73, 41, 72, 38, 11, 12, 302, 268, 303, 271, 304, 272, 408, 407]
                                + [292, 306]
                                + [325, 307, 319, 320, 403, 404, 316, 315, 15, 16, 86, 85, 179, 180, 89, 90, 96, 77])
index_mouth_from_lm478 = index_innerlip_from_lm478 + index_outerlip_from_lm478 + index_withinmouth_from_lm478

index_yaw_from_lm68 = list(range(0, 17))
index_brow_from_lm68 = list(range(17, 27))
index_nose_from_lm68 = list(range(27, 36))
index_eye_from_lm68 = list(range(36, 48))
index_mouth_from_lm68 = list(range(48, 68))


def read_video_to_frames(video_name):
    frames = []
    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        raise IOError("Cannot open video file {}".format(video_name))
    cnt = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Reached the end of the video or failed to read the frame.")
            break
        if frame_bgr is None:
            break
        # too large may cause error, so make sure the face position is in range of 1300x900
        height = frame_bgr.shape[0]
        width = frame_bgr.shape[1]
        if height > 1300:
            frame_bgr = frame_bgr[:1300, :, :]
        if width > 900:
            frame_bgr = frame_bgr[:, :900, :]
        frames.append(frame_bgr)
        cnt += 1
    cap.release()
    print('read {} frames'.format(len(frames)))
    frames = np.stack(frames)
    frames = np.flip(frames, -1)  # BGR ==> RGB
    return frames


def mv_pts(pts, x=0, y=0):
    return [int(round(pts[0] + x)), int(round(pts[1] + y))]


def get_half_face_landmarks_list(landmark, padding=2):
    half_face_landmarks_list = []
    for i in range(0, 68):
        if 1 < i < 15:
            if i in [2, 3, 4, 5, 6]:
                half_face_landmarks_list.append([landmark[i][0] + padding, landmark[i][1]])
            elif i in [10, 11, 12, 13, 14]:
                half_face_landmarks_list.append([landmark[i][0] - padding, landmark[i][1]])
            else:
                half_face_landmarks_list.append(landmark[i])
    half_face_landmarks_list.append([landmark[35][0] + padding, landmark[14][1]])
    half_face_landmarks_list.append([landmark[35][0] + padding, landmark[33][1] + padding])
    half_face_landmarks_list.append([landmark[31][0] - padding, landmark[33][1] + padding])
    half_face_landmarks_list.append([landmark[31][0] - padding, landmark[2][1]])
    return half_face_landmarks_list


def convert68_to_homolm(landmarks):
    homo_landmarks = []
    for i in range(0, 68):
        if 1 < i < 15:
            if i in [2, 3, 4, 5, 6]:
                homo_landmarks.append([landmarks[i][0] + 2, landmarks[i][1]])
            elif i in [10, 11, 12, 13, 14]:
                homo_landmarks.append([landmarks[i][0] - 2, landmarks[i][1]])
            else:
                homo_landmarks.append(landmarks[i])
    homo_landmarks.append([landmarks[35][0], landmarks[35][1]])
    homo_landmarks.append([landmarks[29][0], landmarks[29][1]])
    homo_landmarks.append([landmarks[31][0], landmarks[31][1]])
    return homo_landmarks


def save_homolm(landmarks, save_path):
    data_list = []
    for i, pts in enumerate(landmarks):
        half_pts = convert68_to_homolm(pts)
        for j, p in enumerate(half_pts):
            half_pts[j] = [int(round(p[0])), int(round(p[1]))]
        data_list.append(half_pts)
    with open(save_path, 'w') as file_object:
        json.dump(data_list, file_object)


def save_half_face_landmarks(landmarks, save_path):
    json_data = []
    for i, pts in enumerate(landmarks):
        half_pts = get_half_face_landmarks_list(pts)
        for j, p in enumerate(half_pts):
            half_pts[j] = [int(round(p[0])), int(round(p[1]))]
        json_data.append(half_pts)
    print(json_data)
    with open(save_path, 'w') as file_object:
        json.dump(json_data, file_object)


def save_half_face_mask_landmarks(landmarks, save_path):
    json_data = []
    for i, pts in enumerate(landmarks):
        half_pts = get_half_face_landmarks_list(pts)
        for j, p in enumerate(half_pts):
            if 0 <= j <= 4:
                half_pts[j] = mv_pts(p, 5, 0)
            elif 5 <= j <= 7:
                if j == 5:
                    half_pts[j] = mv_pts(p, 3, -3)
                elif j == 6:
                    half_pts[j] = mv_pts(p, 0, -3)
                elif j == 7:
                    half_pts[j] = mv_pts(p, -3, -3)
            elif 8 <= j <= 12:
                half_pts[j] = mv_pts(p, -5, 0)
            else:
                if j == 13:
                    half_pts[j] = mv_pts(p, 3, 0)
                if j == 14:
                    half_pts[j] = mv_pts(p, 3, 3)
                if j == 15:
                    half_pts[j] = mv_pts(p, -3, 3)
                if j == 16:
                    half_pts[j] = mv_pts(p, -3, 0)
        json_data.append(half_pts)
    with open(save_path, 'w') as file_object:
        json.dump(json_data, file_object)


def save_full_face_landmarks(landmarks, save_path):
    json_data = []
    for i, pts in enumerate(landmarks):
        half_pts = pts.tolist()
        for j, p in enumerate(half_pts):
            half_pts[j] = [int(round(p[0])), int(round(p[1]))]
        json_data.append(half_pts)
    print(json_data)
    with open(save_path, 'w') as file_object:
        json.dump(json_data, file_object)


class MediapipeLandmarker:
    def __init__(self):
        model_path = 'utils/mp_feature_extractors/face_landmarker.task'
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            print("downloading face_landmarker model from mediapipe...")
            model_url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
            os.system(f"wget {model_url}")
            os.system(f"mv face_landmarker.task {model_path}")
            print("download success")
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.image_mode_options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                               running_mode=vision.RunningMode.IMAGE,
                                                               # IMAGE, VIDEO, LIVE_STREAM
                                                               num_faces=1)
        self.video_mode_options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                               running_mode=vision.RunningMode.VIDEO,
                                                               # IMAGE, VIDEO, LIVE_STREAM
                                                               num_faces=1)

    def extract_lm478_from_img_name(self, img_name):
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lm478 = self.extract_lm478_from_img(img)
        return img_lm478

    def extract_lm478_from_img(self, img):
        img_landmarker = vision.FaceLandmarker.create_from_options(self.image_mode_options)
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=img.astype(np.uint8))
        img_face_landmarker_result = img_landmarker.detect(image=frame)
        img_ldm_i = img_face_landmarker_result.face_landmarks[0]
        img_face_landmarks = np.array([[l.x, l.y, l.z] for l in img_ldm_i])
        H, W, _ = img.shape
        img_lm478 = np.array(img_face_landmarks)[:, :2] * np.array([W, H]).reshape([1, 2])  # [478, 2]
        return img_lm478

    def extract_lm478_from_video_name(self, video_name, fps=25, anti_smooth_factor=2):
        frames = read_video_to_frames(video_name)
        print(f"video length: {len(frames)}")
        img_lm478, vid_lm478 = self.extract_lm478_from_frames(frames, fps, anti_smooth_factor)
        return img_lm478, vid_lm478

    def extract_lm478_from_frames(self, frames, fps=25, anti_smooth_factor=20):
        """
        frames: RGB, uint8
        anti_smooth_factor: float, 对video模式的interval进行修改, 1代表无修改, 越大越接近image mode
        """
        img_mpldms = []
        vid_mpldms = []
        img_landmarker = vision.FaceLandmarker.create_from_options(self.image_mode_options)
        vid_landmarker = vision.FaceLandmarker.create_from_options(self.video_mode_options)

        for i in tqdm(range(len(frames))):
            frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[i].astype(np.uint8))
            img_face_landmarker_result = img_landmarker.detect(image=frame)
            vid_face_landmarker_result = vid_landmarker.detect_for_video(image=frame, timestamp_ms=int(
                (1000 / fps) * anti_smooth_factor * i))
            try:
                img_ldm_i = img_face_landmarker_result.face_landmarks[0]
                vid_ldm_i = vid_face_landmarker_result.face_landmarks[0]
            except:
                print(f"Warning: failed detect ldm in idx={i}, use previous frame results.")
            img_face_landmarks = np.array([[l.x, l.y, l.z] for l in img_ldm_i])
            vid_face_landmarks = np.array([[l.x, l.y, l.z] for l in vid_ldm_i])
            img_mpldms.append(img_face_landmarks)
            vid_mpldms.append(vid_face_landmarks)
        img_lm478 = np.stack(img_mpldms)[..., :2]
        vid_lm478 = np.stack(vid_mpldms)[..., :2]
        bs, H, W, _ = frames.shape
        img_lm478 = np.array(img_lm478)[..., :2] * np.array([W, H]).reshape([1, 1, 2])  # [T, 478, 2]
        vid_lm478 = np.array(vid_lm478)[..., :2] * np.array([W, H]).reshape([1, 1, 2])  # [T, 478, 2]
        return img_lm478, vid_lm478

    def combine_vid_img_lm478_to_lm68(self, img_lm478, vid_lm478):
        img_lm68 = img_lm478[:, index_lm68_from_lm478]
        vid_lm68 = vid_lm478[:, index_lm68_from_lm478]
        combined_lm68 = copy.deepcopy(img_lm68)
        combined_lm68[:, index_yaw_from_lm68] = vid_lm68[:, index_yaw_from_lm68]
        combined_lm68[:, index_brow_from_lm68] = vid_lm68[:, index_brow_from_lm68]
        combined_lm68[:, index_nose_from_lm68] = vid_lm68[:, index_nose_from_lm68]
        return combined_lm68

    def combine_vid_img_lm478_to_lm478(self, img_lm478, vid_lm478):
        combined_lm478 = copy.deepcopy(vid_lm478)
        combined_lm478[:, index_mouth_from_lm478] = img_lm478[:, index_mouth_from_lm478]
        combined_lm478[:, index_eye_from_lm478] = img_lm478[:, index_eye_from_lm478]
        return combined_lm478
