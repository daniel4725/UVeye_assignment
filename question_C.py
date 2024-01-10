import cv2
import numpy as np
from skimage.exposure import match_histograms
from question_A import compute_sad
from Helper_Functions import bgr2gray, imshow
import matplotlib.pyplot as plt


def estimate_y_translation(right_image: np.ndarray, left_image: np.ndarray,
                           display_process=True) -> int:
    """
    estimates the y translation of the right image, assuming the left and right images are missing
    translation in y to be rectified. the estimation is based on feature matching.
    display_process flag will present the estimation process.

    Returns: dy, the estimated delta y for the rectification

    """
    # the matching points code inspired from opencv site:
    # https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(bgr2gray(left_image), None)
    keypoints2, descriptors2 = sift.detectAndCompute(bgr2gray(right_image), None)

    # Create a Brute Force Matcher
    bf = cv2.BFMatcher(crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # matches = matches[:100]  # take the best 100

    # the y coordinate of the matches
    left_matches_y = np.array([keypoints1[match.queryIdx].pt[1] for match in matches])  # left img matched points
    right_matches_y = np.array([keypoints2[match.trainIdx].pt[1] for match in matches])  # right img matched points

    # find the y gaps between all the matches and get the most common one
    matches_dy = np.round(right_matches_y - left_matches_y).astype(np.int)
    dy = max(set(matches_dy), key=matches_dy.tolist().count)

    if display_process:
        img_matches = cv2.drawMatches(left_image, keypoints1, right_image, keypoints2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        imshow(img_matches, title="Display the matches between the images")

        plt.hist(matches_dy, bins=100, density=True)
        plt.title(r"The distribution of dy in all the matches")
        plt.xlabel("dy values")
        plt.show()

    return dy


def translate_y(image: np.ndarray, dy) -> np.ndarray:
    """
    translates an image in dy pixels in the y axis
    """
    translated_image = np.zeros_like(image)
    if dy < 0:
        dy = -dy
        translated_image[:-dy, :] = image[dy:]
    else:
        translated_image[dy:, :] = image[:image.shape[0] - dy]
    return translated_image


def rectify_yshift_right_image(right_image: np.ndarray, left_image: np.ndarray,
                               display_process=True) -> np.ndarray:
    """
    estimates the y translation and rectifies the right image accordingly
    Returns: rectified image
    """
    dy = estimate_y_translation(right_image, left_image, display_process)
    rect_right = translate_y(right_image, -dy)
    return rect_right


if __name__ == '__main__':
    from Helper_Functions import *

    data_dir = os.path.join(os.getcwd(), 'data')
    outputs_dir = os.path.join(os.getcwd(), 'outputs')
    config = load_config(os.path.join(data_dir, 'config.yaml'))
    disparity_range = config['disparity_range']
    left_disparity_GT = load_gt_disparity_image(os.path.join(data_dir, 'left_disparity_GT.pkl'))
    left_image_segmentation = load_segmentation(os.path.join(data_dir, 'left_image_segmentation.png'))

    C_dir = os.path.join(data_dir, 'C')
    left_image = cv2.imread(os.path.join(C_dir, 'left_image.png'))
    right_image = cv2.imread(os.path.join(C_dir, 'right_image.png'))

    # rescaling
    scale_factor = 1
    new_w = left_image.shape[1] // scale_factor
    new_h = left_image.shape[0] // scale_factor
    left_image = cv2.resize(left_image, (new_w, new_h))
    right_image = cv2.resize(right_image, (new_w, new_h))
    disparity_range[0] //= scale_factor
    disparity_range[1] //= scale_factor
    left_disparity_GT = cv2.resize(left_disparity_GT, (new_w, new_h))
    left_disparity_GT //= scale_factor
    left_image_segmentation = cv2.resize(left_image_segmentation, (new_w, new_h))

    from question_B import histogram_equalization
    right_image = histogram_equalization(image=right_image)
    left_image = histogram_equalization(image=left_image)

    translated_right_image = translate_y(right_image, dy=-200)
    rect_right = rectify_yshift_right_image(translated_right_image, left_image)

    imshow(right_image)
    imshow(translated_right_image)
    imshow(rect_right)

