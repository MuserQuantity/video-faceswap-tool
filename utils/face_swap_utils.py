import numpy as np
import cv2
import imageio.v2 as iio
import json


def read_points(path):
    # Create an array of points.
    points = []

    # Read points
    with open(path) as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    return points


def read_json_data(file_path):
    try:
        # Open and read JSON file
        with open(file_path, 'r') as file:
            # Parse JSON data
            data = json.load(file)

        # Check if data is a list
        if not isinstance(data, list):
            raise ValueError("JSON data is not expected list format")

        # Return data
        return data

    except json.JSONDecodeError:
        print("JSON decoding error")
    except FileNotFoundError:
        print(f"File '{file_path}' does not exist")
    except ValueError as e:
        print(f"Data format error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # If any errors occur, return an empty list
    return []


def read_video_to_frames(video_name):
    frames = []
    cap = cv2.VideoCapture(video_name)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_name}")
    count = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Reached the end of the video or failed to read the frame.")
            break  # If no frame can be read, exit the loop
        if frame_bgr is None:
            break
        frames.append(frame_bgr)
        count += 1
    cap.release()
    print(f'read {len(frames)} frames')
    frames = np.stack(frames)
    frames = np.flip(frames, -1)  # BGR to RGB
    return frames


def remove_specific_elements(input_list):
    if len(input_list) != 68:
        raise ValueError("Input list must have exactly 68 elements")

    indices_to_remove = [0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    return [item for index, item in enumerate(input_list) if index not in indices_to_remove]


def apply_affine_transform(src, src_tri, dst_tri, size):
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


def calculate_delaunay_triangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangle_list = subdiv.getTriangleList()

    delaunay_tri = []

    point = []

    for t in triangle_list:
        point.append((t[0], t[1]))
        point.append((t[2], t[3]))
        point.append((t[4], t[5]))

        point1 = (t[0], t[1])
        point2 = (t[2], t[3])
        point3 = (t[4], t[5])

        if rect_contains(rect, point1) and rect_contains(rect, point2) and rect_contains(rect, point3):
            index = []
            # Get face points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(point[j][0] - points[k][0]) < 1.0 and abs(point[j][1] - points[k][1]) < 1.0):
                        index.append(k)
            if len(index) == 3:
                delaunay_tri.append((index[0], index[1], index[2]))

        point = []
    return delaunay_tri


def warp_triangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2_rect = np.zeros((r2[3], r2[2]), dtype = img1_rect.dtype)

    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (height, width) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        ratio = height / float(height)
        dim = (int(width * ratio), height)
    else:
        ratio = width / float(width)
        dim = (width, int(height * ratio))

    return cv2.resize(image, dim, interpolation=inter)


def face_swap(img1, img2, points1, points2):
    img1_warped = np.copy(img2)

    # Points
    points1 = remove_specific_elements(points1)
    points2 = remove_specific_elements(points2)

    # Find convex hull
    hull1 = []
    hull2 = []

    hull_index = cv2.convexHull(np.array(points2), returnPoints=False)

    for i in range(0, len(hull_index)):
        hull1.append(points1[int(hull_index[i])])
        hull2.append(points2[int(hull_index[i])])

    # Find Delaunay triangulation for convex hull points
    size_img2 = img2.shape
    rect = (0, 0, size_img2[1], size_img2[0])

    dt = calculate_delaunay_triangles(rect, hull2)

    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        # Get points for img1, img2 corresponding to the triangles

        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        warp_triangle(img1, img1_warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1_warped), img2, mask, center, cv2.NORMAL_CLONE)

    return output
