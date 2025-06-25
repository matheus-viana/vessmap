# This is a toy example of how to use 'image_selection.py'. To run the same experiments
# from the paper one will need the full cortex dataset. Here, we will only provide 10 images.

from image_selection import draw_sample_points
from pathlib import Path

# Parameters 
# -----------
radius = 4
density = 5
resample_resolution = 0.1
n_iter = 5
n_images = 10
root_folder = Path("./cortex_samples")
verbose = True
# -----------

def main():
    _, solution = draw_sample_points(radius, density, resample_resolution, n_iter, n_images,
                                        root_folder, verbose)
    print(solution)

if __name__ == "__main__":
    main()