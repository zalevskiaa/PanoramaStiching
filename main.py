import cv2
import numpy as np

import os


def read_images(dirname):
    images = []

    for filename in sorted(os.listdir(dirname)):
        image = cv2.imread(os.path.join(dirname, filename), cv2.IMREAD_COLOR)
        images.append(image)

    return images


def compute_matches(dst_img, src_img, nfeatures=1000, topfeatures=300):
    orb = cv2.ORB_create(nfeatures=nfeatures)

    kp1, des1 = orb.detectAndCompute(dst_img, None)
    kp2, des2 = orb.detectAndCompute(src_img, None)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher()

    matches = bf.match(des1, des2)

    # matches = sorted(
    #     matches,
    #     key=lambda m: (kp1[m.queryIdx].pt[1]-kp2[m.trainIdx].pt[1])**2
    # )
    matches = sorted(matches, key=lambda match: match.distance)
    matches = matches[:topfeatures]

    return kp1, des1, kp2, des2, matches


def show_matches(dst_img, kp1, src_img, kp2, matches):
    matched_img = cv2.drawMatches(
        dst_img, kp1,
        src_img, kp2,
        matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
    cv2.imshow("Matches", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compute_homography(dst_img, src_img):
    kp1, des1, kp2, des2, matches = compute_matches(dst_img, src_img)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    params = dict(
        # ransacReprojThreshold=100.0,
        maxIters=10000,
        confidence=0.995,
    )
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC, **params)
    return h


def compute_homographies(images):
    hs = {}
    imid = len(images) // 2
    hs[imid] = np.eye(3)

    for i in range(imid + 1, len(images)):
        hii = compute_homography(images[i - 1], images[i])
        him = np.dot(hs[i - 1], hii)
        hs[i] = him

    for i in range(imid - 1, -1, -1):
        hii = compute_homography(images[i + 1], images[i])
        him = np.dot(hs[i + 1], hii)
        hs[i] = him

    return hs


def img_corners(img, H=None):
    h, w = img.shape[:2]
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]])

    if h is not None:
        corners = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2), H
        ).reshape(-1, 2)

    return corners


def compute_corners(images, hs):
    all_corners = [
        img_corners(img, hs[i])
        for i, img in enumerate(images)
    ]
    return all_corners


def compute_res_images(images, hs, all_corners):
    all_corners = np.vstack(all_corners)

    x_min, y_min = all_corners.min(axis=0).astype(np.int32)
    x_max, y_max = all_corners.max(axis=0).astype(np.int32)

    shift_dst = [-x_min, -y_min]
    shift = np.array([[1, 0, shift_dst[0]],
                      [0, 1, shift_dst[1]],
                      [0, 0, 1]], dtype=np.float32)

    for i in hs.keys():
        hs[i] = shift @ hs[i]

    res_images = [
        cv2.warpPerspective(img, hs[i], (x_max - x_min, y_max - y_min))
        for i, img in enumerate(images)
    ]
    return res_images


def add_image(where, what, mode='vacant', inplace=True):
    assert where.shape == what.shape, "The images must have the same shape"

    if mode == 'vacant':
        mask = np.all(where == 0, axis=-1)
    elif mode == 'content':
        mask = np.all(what != 0, axis=-1)

    if not inplace:
        where = where.copy()
    where[mask] = what[mask]

    return where


def combine_images(images):
    i, j = 0, len(images) - 1
    result = np.zeros_like(images[0])

    while True:
        if i > j:
            break
        add_image(result, images[i])
        i += 1

        if i > j:
            break
        add_image(result, images[j])
        j -= 1

    return result


def show_image(result, max_w=1600, max_h=900):
    window_name = 'result'

    window_shape = np.array(result.shape[:2][::-1])
    if window_shape[0] > max_w:
        window_shape = window_shape * max_w / window_shape[0]
    if window_shape[1] > max_h:
        window_shape = window_shape * max_h / window_shape[1]
    window_shape = window_shape.astype(np.int32)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *window_shape)
    cv2.imshow(window_name, result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    images = read_images(os.path.join('img', '1'))
    hs = compute_homographies(images)
    all_corners = compute_corners(images, hs)
    res_images = compute_res_images(images, hs, all_corners)
    result = combine_images(res_images)

    # for i, res_image in enumerate(res_images):
    #     cv2.imwrite(os.path.join('img', f'result{i :02d}.jpg'), res_image)

    cv2.imwrite(os.path.join('img', 'result.jpg'), result)

    show_image(result)


if __name__ == "__main__":
    main()
