# UVeye Assignment
## The Presentation
![image](https://github.com/daniel4725/UVeye_assignment/assets/95569050/1de3506f-7940-4042-a731-8cb960b3d0e9)

The link for this assignment presentation:
https://docs.google.com/presentation/d/17Y8u33Kn_6JAqBkmuVwDitS6QeK5GGOWU9EenpKjalA/edit#slide=id.g2ad8c791a83_0_1062

## Running Instructions
Please execute main.py for evaluation. During the run, all the figures will be created and the evaluation metrics will be printed to the console.

For executing each section alone adjust the input to the main function in main.py.

For faster runtime, the default option of running the script is based on pre-computed disparity maps.
To change that and compute the maps please switch the "load_saved_disparities" to "False" in the input of the main function in main.py.

That beeing said, at question A the disparity maps of the window size comparison parts are not saved due to capacity.
The maps has been pre-computed and evaluated - their metrics has been saved in "/outputs/A/tests_results.csv".
To regenerate them one can run question_A.py. They will be saved in "/outputs/A/saved_maps".
