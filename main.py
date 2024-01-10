from question_A import *
from question_B import *
from question_C import *
from Helper_Functions import *
import pandas as pd


def main(sections2run=('A', 'B', 'C'), scale_factor=1, load_saved_disparities=False):
    """
    Main function of the assignment. runs the sections of the sections2run input.
    (scale_factor > 1) for faster computations by scaling down images.
    when load_saved_disparities flag activated, all the SAD computations will not be done
    and the pre-computed disparity maps will be loaded
    """
    # relevant paths and variables for all the assignment
    data_dir = os.path.join(os.getcwd(), 'data')
    outputs_dir = os.path.join(os.getcwd(), 'outputs')
    config = load_config(os.path.join(data_dir, 'config.yaml'))
    disparity_range = config['disparity_range']
    left_disparity_GT = load_gt_disparity_image(os.path.join(data_dir, 'left_disparity_GT.pkl'))
    left_image_segmentation = load_segmentation(os.path.join(data_dir, 'left_image_segmentation.png'))

    disparity_range, left_disparity_GT, left_image_segmentation = rescale_variables(scale_factor, disparity_range,
                                                                                    left_disparity_GT, left_image_segmentation)
    default_optimal_win_size = 3  # in case running B or C without A
    # ------------------------------------------------------------
    # ------------------------ section A -------------------------
    # ------------------------------------------------------------
    if 'A' in sections2run:
        left_image, right_image = read_section_images(data_dir, section='A', scale_factor=scale_factor)

        # -------------------  1  -------------------
        if load_saved_disparities:
            disparity_map_path = os.path.join(outputs_dir, 'A', 'disparity_SAD_win3.pkl')
            disparity_map = load_disparity_image(disparity_map_path)
        else:
            # Write a function that computes the Sum of Absolute Differences
            sad = compute_sad(bgr2gray(right_image), bgr2gray(left_image), disparity_range, window_size=3)

            # Write a function that computes the disparity for each pixel by the minimum cost
            disparity_map = get_disparity_map_from_sad(sad, disparity_range[0])

        # Save the disparity map
        save_disparity_map(disparity_map, os.path.join(outputs_dir, 'A', 'disparity_SAD_win3.pkl'))

        disparity_show(disparity_map, title="A1: Disparity Map of SAD with window=3")
        disparity_show(left_disparity_GT, title="Ground True Disparity")

        # -------------------  2  -------------------
        # Write a function that compares your results to the ground truth
        metrics_A2 = compare_disparities(disparity_map, left_disparity_GT, left_image_segmentation, display=True)
        print("metrics question 2:", metrics_A2)

        # -------------------  3  -------------------
        # Find an optimal window size for SAD that yields the best score according to the metric you defined

        # the maps has been pre computed and evaluated using compare_disparities.
        # to regenerate them one can run question_A.py. they will be saved in [project_dir]/outputs/A/saved_maps
        # for more convenience and faster runtime, the metrics has
        # been saved in [project_dir]/outputs/A/tests_results.csv

        tests_results = pd.read_csv(os.path.join(outputs_dir, 'A', 'tests_results.csv'))
        best_score_row = tests_results.loc[tests_results['combined_metric'].idxmin()]
        optimal_win_size = best_score_row["win_size"]
        best_disparity_map = disparity_map  # because it is in the same window_size (3)
        plot_results_graphs(tests_results)

        # -------------------  4  -------------------
        # Save the disparity map
        best_disparity_map_path = os.path.join(outputs_dir, 'A', f'disparity_SAD_win{optimal_win_size}.pkl')
        save_disparity_map(best_disparity_map, best_disparity_map_path)

    # ------------------------------------------------------------
    # ------------------------ section B -------------------------
    # ------------------------------------------------------------
    if 'B' in sections2run:
        left_image, right_image = read_section_images(data_dir, section='B', scale_factor=scale_factor)

        # if running B without A
        if 'A' not in sections2run:
            optimal_win_size = default_optimal_win_size

        # -------------------  5  -------------------
        # Compute the disparity between these two new images, using the optimal window size
        if load_saved_disparities:
            disparity_map_path = os.path.join(outputs_dir, 'B', f'disparity_SAD_win{optimal_win_size}.pkl')
            disparity_map = load_disparity_image(disparity_map_path)
        else:
            sad = compute_sad(bgr2gray(right_image), bgr2gray(left_image), disparity_range, window_size=optimal_win_size)
            disparity_map = get_disparity_map_from_sad(sad, disparity_range[0])

        disparity_show(disparity_map, title="B5: Disparity Map of SAD - exposure problem")
        metrics = compare_disparities(disparity_map, left_disparity_GT, left_image_segmentation)
        print("metrics question 5:", metrics)

        # -------------------  6  -------------------
        # Save the disparity map
        disparity_map_path = os.path.join(outputs_dir, 'B', f'disparity_SAD_win{optimal_win_size}.pkl')
        save_disparity_map(disparity_map, disparity_map_path)

        # -------------------  7  -------------------
        # Suggest an improvement to the algorithm and test it on both sets (images sections A & B):
        # load the images from A
        left_image_A, right_image_A = read_section_images(data_dir, section='A', scale_factor=scale_factor)

        # histogram matching improvement:
        # test on images of B -
        right_image_hist_match = match_histograms(image=right_image, reference=left_image, multichannel=True)
        disparity_map_match_B = get_disparity_section_B(right_image_hist_match, left_image, load_saved_disparities,
                                                        outputs_dir, disparity_range, optimal_win_size, 'hist_match_B.pkl')
        disparity_show(disparity_map_match_B, title="B7: Disparity B fixed - histogram matching")
        metrics_hist_match_B = compare_disparities(disparity_map_match_B, left_disparity_GT, left_image_segmentation)
        print("metrics question 7, image B, hist_match:", metrics_hist_match_B)

        # test on images of A -
        right_image_hist_match_A = match_histograms(image=right_image_A, reference=left_image_A, multichannel=True)
        disparity_map_match_A = get_disparity_section_B(right_image_hist_match_A, left_image_A, load_saved_disparities,
                                                        outputs_dir, disparity_range, optimal_win_size, 'hist_match_A.pkl')
        disparity_show(disparity_map_match_A, title="B7: Disparity A - histogram matching")
        metrics_hist_match_A = compare_disparities(disparity_map_match_A, left_disparity_GT, left_image_segmentation)
        print("metrics question 7, image A, hist_match:", metrics_hist_match_A)

        # histogram equalization improvement:
        # test on images of B -
        right_image_hist_equal = histogram_equalization(image=right_image)
        left_image_hist_equal = histogram_equalization(image=left_image)
        disparity_map_eq_B = get_disparity_section_B(right_image_hist_equal, left_image_hist_equal, load_saved_disparities,
                                                     outputs_dir, disparity_range, optimal_win_size, 'hist_eq_B.pkl')
        disparity_show(disparity_map_eq_B, title="B7: Disparity B fixed - histogram equalization")
        metrics_hist_eq_B = compare_disparities(disparity_map_eq_B, left_disparity_GT, left_image_segmentation)
        print("metrics question 7, image B, hist_eq:", metrics_hist_eq_B)

        # test on images of A -
        right_image_hist_equal_A = histogram_equalization(image=right_image_A)
        left_image_hist_equal_A = histogram_equalization(image=left_image_A)
        disparity_map_eq_A = get_disparity_section_B(right_image_hist_equal_A, left_image_hist_equal_A, load_saved_disparities,
                                                     outputs_dir, disparity_range, optimal_win_size, 'hist_eq_A.pkl')
        disparity_show(disparity_map_eq_A, title="B7: Disparity A - histogram equalization")
        metrics_hist_eq_A = compare_disparities(disparity_map_eq_A, left_disparity_GT, left_image_segmentation)
        print("metrics question 7, image A, hist_eq:", metrics_hist_eq_A)

        # present fixed images and histograms the results
        hist_show(left_image, title="left image histogram")
        hist_show(right_image, title="right image histogram - different exposure")
        hist_show(right_image_hist_match, title="right image histogram - matched to left")
        hist_show(right_image_hist_equal, title="right image histogram - histogram equalization")
        hist_show(left_image_hist_equal, title="left image histogram - histogram equalization")

        imshow(left_image, title="left image")
        imshow(right_image, title="right image - different exposure")
        imshow(right_image_hist_match, title="right image - histogram matched to left")
        imshow(right_image_hist_equal, title="right image - histogram equalization")
        imshow(left_image_hist_equal, title="left image - histogram equalization")

        # -------------------  8  -------------------
        # Save the disparity map
        disparity_map_path = os.path.join(outputs_dir, 'B', 'disparity_improvments.pkl')
        save_disparity_map(disparity_map_eq_B, disparity_map_path)

    # ------------------------------------------------------------
    # ------------------------ section C -------------------------
    # ------------------------------------------------------------
    if 'C' in sections2run:
        left_image, right_image = read_section_images(data_dir, section='C', scale_factor=scale_factor)

        # if running C without A
        if 'A' not in sections2run:
            optimal_win_size = default_optimal_win_size

        # we saw that histogram equalization has the most positive impact on the results
        # this will generate better results
        right_image = histogram_equalization(image=right_image)
        left_image = histogram_equalization(image=left_image)

        # evaluate the disparity of the non rectified images
        disparity_map_path = os.path.join(outputs_dir, 'C', 'saved_maps', 'disparity_non_rect.pkl')
        if load_saved_disparities:
            disparity_map = load_disparity_image(disparity_map_path)
        else:
            sad = compute_sad(bgr2gray(right_image), bgr2gray(left_image), disparity_range, window_size=optimal_win_size)
            disparity_map = get_disparity_map_from_sad(sad, disparity_range[0])
            save_disparity_map(disparity_map, disparity_map_path)

        disparity_show(disparity_map, title="C9: Disparity before rectification")
        metrics = compare_disparities(disparity_map, left_disparity_GT, left_image_segmentation)
        print("metrics question C9 - non rectified:", metrics)

        # -------------------  9  -------------------
        # Write an algorithm that fixes the rectification issue and generates new rectified image pairs
        right_rect_image = rectify_yshift_right_image(right_image, left_image, display_process=True)

        # Save the new rectified pair
        right_rect_path = os.path.join(outputs_dir, 'C', 'Rectification_fix_right_image.png')
        left_image_path = os.path.join(outputs_dir, 'C', 'left_image.png')
        cv2.imwrite(right_rect_path, right_rect_image)
        cv2.imwrite(left_image_path, left_image)

        # -------------------  10  -------------------
        # Once you fix the rectification issue, run your best algorithm and compare
        # the results to the results obtained on the set from PartA
        if load_saved_disparities:
            disparity_map_path = os.path.join(outputs_dir, 'C', 'disparity_rectification_fix.pkl')
            disparity_map = load_disparity_image(disparity_map_path)
        else:
            # the images are after histogram equalization already
            sad = compute_sad(bgr2gray(right_rect_image), bgr2gray(left_image), disparity_range, window_size=optimal_win_size)
            disparity_map = get_disparity_map_from_sad(sad, disparity_range[0])

        disparity_show(disparity_map, title="C10: Disparity after rectification and hist equal")
        metrics = compare_disparities(disparity_map, left_disparity_GT, left_image_segmentation)
        print("metrics question C10 - rectified:", metrics)

        # -------------------  11  -------------------
        #  Save the disparity map
        disparity_map_path = os.path.join(outputs_dir, 'C', 'disparity_rectification_fix.pkl')
        save_disparity_map(disparity_map, disparity_map_path)

if __name__ == '__main__':
    # main(sections2run=('A', 'B', 'C'), scale_factor=1, load_saved_disparities=True)
    main(sections2run=('C',), scale_factor=1, load_saved_disparities=True)
