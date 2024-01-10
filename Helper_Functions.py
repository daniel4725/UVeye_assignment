import pickle

import matplotlib.pyplot as plt
import yaml
import cv2
import os


def load_gt_disparity_image(disparity_pickle_file_path:str):
    disparity = pickle.load(open(disparity_pickle_file_path, "rb"))
    return disparity


def load_disparity_image(disparity_pickle_file_path:str):
    disparity = pickle.load(open(disparity_pickle_file_path, "rb"))
    return disparity


def load_config(file_path:str):
    with open(file_path) as file:
        calib_config = yaml.load(file, Loader=yaml.FullLoader)
    return calib_config


def load_segmentation(file_path, threhold=20):
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    mask[mask < threhold] = 0
    mask[mask >= threhold] = 1
    return mask


def save_disparity_map(disparity_map,file_path):
    pickle.dump(disparity_map, open(file_path, "wb"))


def imshow(img, title=None):
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
    if not (title is None):
        plt.title(title)
    plt.show()


def disparity_show(img, title=None):
    plt.imshow(img)
    if not (title is None):
        plt.title(title)
    plt.colorbar()
    plt.show()


def hist_show(img, title=None):
    if img.ndim != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.hist(img.flatten(), bins=256, range=(0, 256))
    if not (title is None):
        plt.title(title)
    plt.show()


def read_section_images(data_dir: str, section: str, scale_factor=1):
    """
    read the images for the relevant section and rescales them if necessary
    :param data_dir: the data directory
    :param section: the section to read from (A, B or C)
    :param scale_factor: the factor to scale the images - No scaling is the default
    :return: the images in the section
    """
    left_image = cv2.imread(os.path.join(data_dir, section, 'left_image.png'))
    right_image = cv2.imread(os.path.join(data_dir, section, 'right_image.png'))
    if scale_factor != 1:
        new_w = left_image.shape[1] // scale_factor
        new_h = left_image.shape[0] // scale_factor
        left_image = cv2.resize(left_image, (new_w, new_h))
        right_image = cv2.resize(right_image, (new_w, new_h))
    return left_image, right_image


def rescale_variables(scale_factor, disparity_range, left_disparity_GT, left_image_segmentation):
    """
    rescales all the relevant variables by the 1/scaling factor
    :return: rescaled variables
    """
    if scale_factor != 1:
        new_w = left_image_segmentation.shape[1] // scale_factor
        new_h = left_image_segmentation.shape[0] // scale_factor
        disparity_range[0] //= scale_factor
        disparity_range[1] //= scale_factor
        left_disparity_GT = cv2.resize(left_disparity_GT, (new_w, new_h))
        left_disparity_GT //= scale_factor
        left_image_segmentation = cv2.resize(left_image_segmentation, (new_w, new_h))
    return disparity_range, left_disparity_GT, left_image_segmentation


def bgr2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)