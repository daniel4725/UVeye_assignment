import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from question_A import compute_sad, get_disparity_map_from_sad
from Helper_Functions import *


def histogram_equalization(image):
    """ perform histogram equalization either on gray scaled image or 3 channels (per channel)"""
    if image.ndim == 1:
        return cv2.equalizeHist(image)
    else:
        equalized_channels = [cv2.equalizeHist(image[:, :, channel]) for channel in range(3)]
        return cv2.merge(equalized_channels)  # Merge the equalized channels back into an RGB image


def get_disparity_section_B(right_image, left_image, load_saved_disparities: bool, outputs_dir: str,
                            disparity_range: np.ndarray, optimal_win_size: int, file_name: str,) -> np.ndarray:
    """
    computes the disparity using sad or loading the pre-saved disparity image
    """
    disparity_map_path = os.path.join(outputs_dir, 'B', 'saved_maps', file_name)
    if not load_saved_disparities:
        sad = compute_sad(bgr2gray(right_image), bgr2gray(left_image), disparity_range,
                          window_size=optimal_win_size)
        disparity_map = get_disparity_map_from_sad(sad, disparity_range[0])
        save_disparity_map(disparity_map, disparity_map_path)
    else:
        disparity_map = load_disparity_image(disparity_map_path)
    return disparity_map


if __name__ == '__main__':

    data_dir = os.path.join(os.getcwd(), 'data')
    # outputs_dir = os.path.join(os.getcwd(), 'outputs')
    # config = load_config(os.path.join(data_dir, 'config.yaml'))
    # disparity_range = config['disparity_range']
    # left_disparity_GT = load_gt_disparity_image(os.path.join(data_dir, 'left_disparity_GT.pkl'))
    # left_image_segmentation = load_segmentation(os.path.join(data_dir, 'left_image_segmentation.png'))
    #
    # B_dir = os.path.join(data_dir, 'B')
    # left_image = cv2.imread(os.path.join(B_dir, 'left_image.png'))
    # right_image = cv2.imread(os.path.join(B_dir, 'right_image.png'))
    #
    # scale_factor = 6
    # new_w = left_image.shape[1] // scale_factor
    # new_h = left_image.shape[0] // scale_factor
    # left_img = cv2.resize(left_image, (new_w, new_h))
    # right_img = cv2.resize(right_image, (new_w, new_h))
    # disparity_range[0] //= scale_factor
    # disparity_range[1] //= scale_factor
    # left_disparity_GT = cv2.resize(left_disparity_GT, (new_w, new_h))
    # left_disparity_GT //= scale_factor
    # left_image_segmentation = cv2.resize(left_image_segmentation, (new_w, new_h))


    # # convert the images to grayscale
    # left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    # right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # plt.hist(left_image.flatten())
    # plt.show()
    # plt.hist(right_image.flatten())
    # plt.show()


    # # convert the images to grayscale
    # left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    # right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # new_right = match_histograms(image=right_image, reference=left_image, multichannel=True)
    # hist_show(left_image)
    # hist_show(new_right)
    # hist_show(right_image)
    #
    # imshow(left_image)
    # imshow(new_right)
    # imshow(right_image)




