import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd


def compute_sad(right_image: np.ndarray, left_image: np.ndarray,
                disparity_range: np.ndarray, window_size=3) -> np.ndarray:
    """
    Compute the Sum Absolute Difference for each disparity in disparity_range between blocks
    of size window_size in two rectified images.
    :param right_image: the right rectified image, either grayscale or 3 channels
    :param left_image: the left rectified image, either grayscale or 3 channels
    :param disparity_range: the relevant ranges of disparities to calculate the SAD for
    :param window_size: block size for computing the SAD
    :return: SAD tensor of size [image_height, image_width, num_of_disparities]
    """
    # input check
    if window_size % 2 == 0:  # change it to odd for convenience
        window_size += 1
    assert right_image.shape == left_image.shape, "right_image and left_image must have the same shape"  # can also be resolved to allow such an input
    # assuming disparity_range legal...

    half_window_size = window_size // 2
    height, width = left_image.shape[:2]

    num_disparities = disparity_range[1] - disparity_range[0] + 1  # total number of disparities
    sad = np.full((height, width, num_disparities), np.inf)  # initialize the output SAD tensor
    kernel = np.ones(shape=(window_size, window_size), dtype=left_image.dtype)  # convolution kernel for fast summation

    # iterate over all the disparities
    for disp_idx, disp in tqdm(enumerate(range(disparity_range[0], disparity_range[1] + 1))):

        # calculate the absolute differance for the specific disparity
        current_disp_abs_diff = np.abs(right_image[:, :width - disp] - left_image[:, disp:])

        # sum the absolute differance for each block using convolution with the ones kernel
        if current_disp_abs_diff.ndim == 3:  # if we are using 3 channels over sum all the channels
            current_disp_SAD = my_convolve2d_fft(current_disp_abs_diff[:, :, 0], kernel)
            current_disp_SAD += my_convolve2d_fft(current_disp_abs_diff[:, :, 1], kernel)
            current_disp_SAD += my_convolve2d_fft(current_disp_abs_diff[:, :, 2], kernel)
        else:   # if we are using gray scale
            current_disp_SAD = my_convolve2d_fft(current_disp_abs_diff, kernel)

        # crop edges (the values there are invalid)
        current_disp_SAD = current_disp_SAD[half_window_size: - half_window_size, half_window_size: - half_window_size]

        # insert to the sad tensor that contains all the disparities' SAD
        sad[half_window_size: - half_window_size, half_window_size + disp: width - half_window_size, disp_idx] = current_disp_SAD

    return sad


def get_disparity_map_from_sad(sad: np.ndarray, lower_disparity_range: int) -> np.ndarray:
    """
    computes the disparity map from SAD tensor created by compute_sad function using
    minimal SAD for the disparity prediction.
    :param sad: SAD tensor created by compute_sad function using
    :param lower_disparity_range: the bottom disparity range used to compute the SAD tensor
    :return: disparity map
    """
    disparity = np.argmin(sad, axis=2).astype(np.float)
    # Replace indices with inf with nan
    min_values = np.min(sad, axis=2)
    disparity[min_values == np.inf] = np.nan
    return disparity + lower_disparity_range


def my_convolve2d_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    performs a 2D convolution of an image and a kernel using Fast Fourier Transform.
    :param image: the input image
    :param kernel: the convolution kernel
    :return: the convolved image
    """
    # assuming square kernel
    half_kernel_size = kernel.shape[0] // 2

    # Get the size for FFT which is image size + kernel size - 1
    fft_shape = [image.shape[0] + kernel.shape[0] - 1,
                 image.shape[1] + kernel.shape[1] - 1]

    # FFT for the image and the kernel and multiply in the frequency domain
    mult_in_freq_domain = np.fft.fft2(image, fft_shape) * np.fft.fft2(kernel, fft_shape)

    # inverse FFT to get back to the spatial domain and take the real part
    convolved_image = np.fft.ifft2(mult_in_freq_domain).real

    # round and crop to the original image size
    convolved_image = np.round(convolved_image)
    convolved_image = convolved_image[half_kernel_size: -half_kernel_size, half_kernel_size: -half_kernel_size]

    return convolved_image


def get_seg_boundary(segmentation: np.ndarray, thickness) -> np.ndarray:
    """
    extracts the contours of a given segmentation
    :param segmentation: binary segmentation image
    :param thickness: boundary line thikness
    :return: a boundary segmentation
    """
    seg_boundary = np.zeros_like(segmentation, dtype=np.uint8)
    seg_boundary = np.stack([seg_boundary, seg_boundary, seg_boundary], axis=2)

    # detect the contours on the binary image
    contours, _ = cv2.findContours(image=segmentation, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # draw contours on the boundary image
    cv2.drawContours(image=seg_boundary, contours=contours, contourIdx=-1, color=(1, 1, 1), thickness=thickness)

    return seg_boundary[:, :, 0]


def compare_disparities(prediction: np.ndarray, ground_truth: np.ndarray, segmentation: np.ndarray,
                        boundary_importance=0.75, chosen_metric="BPR5", display=True) -> dict:
    """
    function to compare and evaluate disparity map predictions.
    Args:
        prediction: predicted disparity
        ground_truth: GT disparity
        segmentation: segmentation map to fpcus on
        boundary_importance: the importance of the boundary in the metrics (0 to 1)
        chosen_metric: the chosen metric to compare all
        display: flag for displaying the diff map between the disparities in the relevant region

    Returns: dictionary of metrics
    """
    seg_boundary = get_seg_boundary(segmentation, thickness=10) == 1
    gt_valid_mask = 1 - (np.isnan(ground_truth) | np.isinf(ground_truth))
    pred_valid_mask = 1 - np.isnan(prediction)

    # take the only the valid and relevant parts and create a mask: one for the boundary and one for the interior
    relevant_interior = (gt_valid_mask & pred_valid_mask & segmentation) == 1
    relevant_boundary = (relevant_interior & seg_boundary) == 1
    relevant_interior = relevant_interior & (~ seg_boundary)  # avoid overlapping

    pred_in = prediction[relevant_interior]
    gt_in = ground_truth[relevant_interior]
    pred_boundary = prediction[relevant_boundary]
    gt_boundary = ground_truth[relevant_boundary]

    # absolut_errors
    AE_interior = abs(pred_in - gt_in)
    AE_boundary = abs(pred_boundary - gt_boundary)

    metrics = dict(
        # Bad Pixel Ratio
        BPR2_interior=(AE_interior > 2).sum() / len(AE_interior),
        BPR2_boundary=(AE_boundary > 2).sum() / len(AE_boundary),
        BPR5_interior=(AE_interior > 5).sum() / len(AE_interior),
        BPR5_boundary=(AE_boundary > 5).sum() / len(AE_boundary),

        # Mean Absolute Error
        MAE_interior=AE_interior.mean(),
        MAE_boundary=AE_boundary.mean(),

        # Root Mean Squared Error
        RMSE_interior=np.sqrt((AE_interior ** 2).mean()),
        RMSE_boundary=np.sqrt((AE_boundary ** 2).mean()),

    )
    # the combination between boundary and interior for the chosen metric
    a = boundary_importance
    boundary_score = metrics[f"{chosen_metric}_boundary"]
    interior_score = metrics[f"{chosen_metric}_interior"]
    metrics["combined_metric"] = a * boundary_score + (1 - a) * interior_score

    if display:
        # print(metrics)
        diff = abs(prediction - ground_truth)
        diff[~ (relevant_interior | relevant_boundary)] = np.nan
        plt.imshow(diff)
        plt.title("Absolute Error map of the relevant part")
        plt.colorbar()
        plt.show()

    return metrics


def plot_results_graphs(test_res_csv):
    """
    plots graph for presenting the comparisons between different window sizes
    and using gray scale or rgb in the SAD algorythm for disparity generation.
    """
    # merge the csv to have the RGB or grayscale information as part of each metric
    gray = test_res_csv[test_res_csv["channels"] == "gray"]
    gray.columns = [f"{col_name}_gray" if col_name not in ["channels", "win_size"] else col_name for col_name in gray.columns ]
    gray = gray.reset_index(drop=True)

    colors = test_res_csv[test_res_csv["channels"] == "RGB"]
    colors.columns = [f"{col_name}_RGB" if col_name not in ["channels", "win_size"] else col_name for col_name in colors.columns ]
    colors = colors.drop(columns=["channels", "win_size"])
    colors = colors.reset_index(drop=True)

    results = pd.concat([gray, colors], axis=1)

    # plot the results
    metrics = ["BPR2", "BPR5", "MAE", "RMSE"]
    for metric in metrics:
        y = [f'{metric}_interior_gray', f'{metric}_boundary_gray',
             f'{metric}_interior_RGB', f'{metric}_boundary_RGB']
        results.plot(x="win_size", y=y, kind='line', marker='o', color=['lightblue', 'blue', 'lightcoral', 'red'])

        # Add labels and title
        plt.xlabel('window size')
        plt.ylabel(metric)
        plt.title(f'{metric} as function of window size - RGB and gray scale')
        plt.show()


    y = [f'combined_metric_gray', f'combined_metric_RGB']
    results.plot(x="win_size", y=y, kind='line', marker='o')

    # Add labels and title
    plt.xlabel('window size')
    plt.ylabel("combined metric")
    plt.title(f'Combined metric as function of window size - RGB and gray scale')
    plt.show()

if __name__ == '__main__':
    from Helper_Functions import *
    import pandas as pd

    data_dir = os.path.join(os.getcwd(), 'data')
    outputs_dir = os.path.join(os.getcwd(), 'outputs')
    config = load_config(os.path.join(data_dir, 'config.yaml'))
    disparity_range = config['disparity_range']
    left_disparity_GT = load_gt_disparity_image(os.path.join(data_dir, 'left_disparity_GT.pkl'))
    left_image_segmentation = load_segmentation(os.path.join(data_dir, 'left_image_segmentation.png'))

    A_dir = os.path.join(data_dir, 'A')
    left_image = cv2.imread(os.path.join(A_dir, 'left_image.png'))
    right_image = cv2.imread(os.path.join(A_dir, 'right_image.png'))

    scale_factor = 1
    window_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    compute_all = True

    # computes the disparities and compares the metrics for different window sizes
    # and both 3-channeled and gray scaled images
    if compute_all:
        for window_size in window_sizes:
            # gray scale
            sad = compute_sad(bgr2gray(right_image), bgr2gray(left_image), disparity_range, window_size=window_size)
            disparity_map = get_disparity_map_from_sad(sad, disparity_range[0])
            save_disparity_map(disparity_map, os.path.join(outputs_dir, 'A', 'saved_maps', f'scale{scale_factor}_gray_win{window_size}.pkl'))

            # 3 channels
            sad = compute_sad(right_image, left_image, disparity_range, window_size=window_size)
            disparity_map = get_disparity_map_from_sad(sad, disparity_range[0])
            save_disparity_map(disparity_map, os.path.join(outputs_dir, 'A', 'saved_maps', f'scale{scale_factor}_colors_win{window_size}.pkl'))


    # Create an empty DataFrame for the metrics
    path_gray = os.path.join('outputs', 'A', "disparity_SAD_win3.pkl")
    d = load_gt_disparity_image(path_gray)
    metrics = compare_disparities(d, left_disparity_GT, left_image_segmentation, display=False)
    metrics["win_size"] = 3
    metrics["channels"] = 1
    df = pd.DataFrame(columns=list(metrics.keys()))

    for window_size in tqdm(window_sizes):
        path_gray = os.path.join('outputs', 'A', 'saved_maps', f"scale{scale_factor}_gray_win{window_size}.pkl")
        d = load_gt_disparity_image(path_gray)
        metrics = compare_disparities(d, left_disparity_GT, left_image_segmentation, display=False)
        metrics["win_size"] = window_size
        metrics["channels"] = "gray"
        df = df.append(metrics, ignore_index=True)

        path_colors = os.path.join('outputs', 'A', 'saved_maps', f"scale{scale_factor}_colors_win{window_size}.pkl")
        d = load_gt_disparity_image(path_colors)
        metrics = compare_disparities(d, left_disparity_GT, left_image_segmentation, display=False)
        metrics["win_size"] = window_size
        metrics["channels"] = "RGB"
        df = df.append(metrics, ignore_index=True)

    results_path = os.path.join(outputs_dir, 'A', 'tests_results.csv')
    df.to_csv(results_path, index=False)


    # scale_factor = 8
    # new_w = left_img.shape[1] // scale_factor
    # new_h = left_img.shape[0] // scale_factor
    # left_img = cv2.resize(left_img, (new_w, new_h))
    # right_img = cv2.resize(right_img, (new_w, new_h))
    # disparity_range[0] //= scale_factor
    # disparity_range[1] //= scale_factor
    # left_disparity_GT = cv2.resize(left_disparity_GT, (new_w, new_h))
    # left_disparity_GT //= scale_factor
    # left_image_segmentation = cv2.resize(left_image_segmentation, (new_w, new_h))
    #
    # # -------------------  1  -------------------
    # sad = compute_sad(right_img, left_img, disparity_range, window_size=3)
    #
    # # -------------------  2  -------------------
    # disparity_map = get_disparity_map_from_sad(sad, disparity_range[0])
    # save_disparity_map(disparity_map, os.path.join(outputs_dir, 'A', 'disparity_SAD_win3.pkl'))
    #
    # plt.imshow(disparity_map)
    # plt.show()
    # plt.imshow(left_disparity_GT)
    # plt.show()
    # compare_disparities(disparity_map, left_disparity_GT, left_image_segmentation)

    # # -------------------  3  -------------------
    # metrics = []
    # for window_size in [3, 5, 7, 9, 11]:
    #     metric = compare_disparities(disparity_map, left_disparity_GT, left_image_segmentation)
    #     metrics.append(metric)
    #     optimal_win_size = 3
    #
    # plt.imshow(disparity_map)
    # plt.show()
    # plt.imshow(left_disparity_GT)
    # plt.show()
    # # -------------------  4  -------------------
    # save_disparity_map(disparity_map, os.path.join(outputs_dir, 'A', f'disparity_SAD_win{optimal_win_size}.pkl'))





