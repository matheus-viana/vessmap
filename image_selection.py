import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import json
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA

from typing import List, Dict, Tuple, Any
from numpy.typing import NDArray

def get_sphere(radius: int, density: complex) -> List[List[float]]:
    ''' 
    Create a sparse hypersphere with four dimensions. The hypersphere will have a radius 'radius'.
    The magnitude of the imaginary part of 'density' defines the number of points used to create
    the hypersphere.
    '''

    origin = np.zeros((4, ))

    # define a hypercube
    hcube = np.mgrid[-radius:radius:density, -radius:radius:density, -radius:radius:density, -radius:radius:density]

    # set list of points that define a hypersphere
    hsphere = []
    step = int(density.imag)
    for x in range(step):
        for y in range(step):
            for z in range(step):
                for k in range(step):
                    v1 = np.array([
                        hcube[0][x, y, z, k], 
                        hcube[1][x, y, z, k], 
                        hcube[2][x, y, z, k], 
                        hcube[3][x, y, z, k]
                    ])
                    if np.linalg.norm(v1 - origin) <= radius:
                        hsphere.append([v1[0], v1[1], v1[2], v1[3]])

    return hsphere

def get_normalized_metrics(
        all_images_dicts: List[Dict[str, Any]]
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Get the metrics from each window of 'all_images_dicts' and apply the z-score
    normalization on them.
    """

    densities = []
    contrasts = []
    skel_heterogeneities = []
    noises = []

    for entry in all_images_dicts:
        for window in entry['measures'].keys():
            densities.append(entry['measures'][window]['density'])
            contrasts.append(entry['measures'][window]['contrast'])
            skel_heterogeneities.append(entry['measures'][window]['skeleton_res_heterogeneity'])
            noises.append(entry['measures'][window]['gauss_noise_estimation'])

    avg = np.mean(densities)
    std = np.std(densities)
    densities = (densities - avg) / std

    avg = np.mean(contrasts)
    std = np.std(contrasts)
    contrasts = (contrasts - avg) / std

    avg = np.mean(skel_heterogeneities)
    std = np.std(skel_heterogeneities)
    skel_heterogeneities = (skel_heterogeneities - avg) / std

    avg = np.mean(noises)
    std = np.std(noises)
    noises = (noises - avg) / std

    return densities, contrasts, skel_heterogeneities, noises

def get_original_metrics(all_images_dicts: List[Dict[str, Any]]
                         ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """ Returns the metrics from each window of 'all_images_dicts'. """

    densities = []
    contrasts = []
    skel_heterogeneities = []
    noises = []

    for entry in all_images_dicts:
        for window in entry['measures'].keys():
            densities.append(entry['measures'][window]['density'])
            contrasts.append(entry['measures'][window]['contrast'])
            skel_heterogeneities.append(entry['measures'][window]['skeleton_res_heterogeneity'])
            noises.append(entry['measures'][window]['gauss_noise_estimation'])

    return np.array(densities), np.array(contrasts), np.array(skel_heterogeneities), np.array(noises)

def resampled_metrics(densities: NDArray[np.float64], contrasts: NDArray[np.float64], 
                      skel_heterogeneities: NDArray[np.float64], noises: NDArray[np.float64], resolution: float
                      ) -> Tuple[NDArray[np.int16], NDArray[np.int16], NDArray[np.int16], NDArray[np.int16]]:
    
    """ Resample the metrics to a new resolution. """

    densities = (densities / resolution).astype(np.int16)
    contrasts = (contrasts / resolution).astype(np.int16)
    skel_heterogeneities = (skel_heterogeneities / resolution).astype(np.int16)
    noises = (noises / resolution).astype(np.int16)

    return densities, contrasts, skel_heterogeneities, noises

def approximate_distribution(hsphere: List[Tuple[float, float, float, float]], densities: List[NDArray[np.int16]], 
                             contrasts: List[NDArray[np.int16]], skel_heterogeneities: List[NDArray[np.int16]], 
                             noises: List[NDArray[np.int16]], verbose: bool
                             ) -> Dict[Tuple[float, float, float, float], int]:
    
    """
    Algorithm for approximating the distribution of the data with a sparse hypersphere
    ------------------

    For each data point, translate the hypersphere and add the translated points to
    the distribution set. Collisions increase a hit count for each point. This set of 
    points approximates the distribution of the real data.
    """

    distribution_points = {}
    n_data_points = len(densities)

    iterate_through = tqdm(range(n_data_points)) if verbose else range(n_data_points)

    for i in iterate_through:
        data_point = [densities[i], contrasts[i], skel_heterogeneities[i], noises[i]]
        transl_hsphere = np.array(hsphere) + data_point
        for sphere_point in transl_hsphere:
            t_sphere_point = tuple(sphere_point)
            if t_sphere_point in distribution_points:
                distribution_points[t_sphere_point] += 1
            else:
                distribution_points[t_sphere_point] = 1

    return distribution_points

def get_windows_info(all_images_dicts: List[Dict[str, Any]]) -> Tuple[List[str], List[slice]]:
    """Returns the original filename and position of each window in 'all_images_dicts'."""

    fnames = []
    win_pos = []

    for image in all_images_dicts:
        windows = image['measures'].keys()
        for idx, _ in enumerate(windows):
            fnames.append(image['name'])
            win_pos.append(image['windows'][idx])

    return fnames, win_pos

def get_dist_point_from_window(distribution: NDArray[np.float64], selected_points: List[int], 
                               closest_windows: List[int], window_idx: int) -> NDArray[np.float64]:
    
    """Returns the distribution point associated with the window with index 'window_idx'."""

    # print(f"-->TODO (check type): {type(selected_points)}. It should be a List[int].")
    point_idx = selected_points[closest_windows.index(window_idx)]
    return distribution[point_idx]

def get_window_point(metrics: List[List[int]], ind: int) -> NDArray[np.int16]:
    """Returns a numpy array containing the metrics from the window point with index 'ind'."""
    return np.array([metrics[0][ind], metrics[1][ind], metrics[2][ind], metrics[3][ind]])

def euclidean_distance(v1: NDArray[np.int16], v2: NDArray[np.int16]) -> NDArray[np.float64]:
    """Returns the euclidean distance between points 'v1' and 'v2'."""
    # print(f"-->TODO (test type): {v1.dtype}, {v2.dtype}")
    return np.linalg.norm(v1-v2)

def get_window_to_swap(distribution: NDArray[np.float64], closest_windows: List[int], all_fnames: List[str], 
                       selected_windows: List[int], selected_fnames: List[str], selected_points: List[int], 
                       metrics: List[List[int]], window_idx: int) -> int:
    """
    Iteratively find another point to replace 'window_idx'. The new window point cannot have been selected before 
    nor come from an image that already has a window point selected.
    """

    orig = get_dist_point_from_window(distribution, selected_points, closest_windows, window_idx)
    distances = []
    for target_idx in range(len(metrics[0])):
        target = get_window_point(metrics, target_idx)
        # print(f"--> TODO (test type): {target.dtype}. It should be np.int16")
        distances.append(euclidean_distance(orig, target))
    
    sort_idxs = np.argsort(distances)
    for window_idx in sort_idxs:
        if window_idx not in selected_windows and all_fnames[window_idx] not in selected_fnames:
            return window_idx

    return -1

def farthest_unselected_point(closest_windows: List[int], distances_between_windows: NDArray[np.float64]) -> float:
    """Returns the largest distance between an unselected point and the drawn windows ('closest_windows')."""

    unselected_points = [p for p in range(len(distances_between_windows)) if p not in closest_windows]
    uns_to_win_distances = []
    for uns in unselected_points:
        distances = distances_between_windows[uns, closest_windows]
        uns_to_win_distances.append(np.min(distances))
    
    return np.max(uns_to_win_distances)
    
def draw_windows(n_iter: int, n_images: int, metrics: List[List[int]], distribution: NDArray[np.float64], 
                 fnames: List[str], distances_between_windows: NDArray[np.float64], verbose: bool = False
                 ) -> Tuple[int, float, List[int]]:
    
    """
    Draw 'n_images' points from 'distribution' and select the closest windows from the drawn points. A window cannot 
    be selected twice and only a single window can be selected for each original image. To do this, the windows are 
    swapped iteratively as necessary. 'n_iter' sets of windows are drawn, and the set with the smallest value of the 
    'farthest unselected point' metric is selected.
    """

    solution_distances = []
    solution_windows = []
    seeds = []
    Tmetrics = np.array(metrics).T

    iterate_through = tqdm(range(n_iter)) if verbose else range(n_iter)

    for n in iterate_through:
        seed = np.random.randint(2**32-1) # max numpy seed value
        seeds.append(seed)
        np.random.seed(seed)
        selected_points = np.random.choice(np.arange(0, len(distribution), 1), n_images, replace=False)

        # closest images to the drawn points
        closest_windows = []
        for orig_ind in selected_points:
            v1 = distribution[orig_ind]
            distances = np.sqrt(((v1 - Tmetrics)**2).sum(axis=1))
            closest_windows.append(np.argmin(distances))

        # iteratively swap windows from the same image
        unique_fnames = np.unique([fnames[idx] for idx in closest_windows])
        selected_fnames = list(unique_fnames.copy())
        selected_windows = closest_windows.copy()
        while len(unique_fnames) != len(closest_windows):
            windows_to_swap = []
            for fname in unique_fnames:
                windows_from_fname = [window for window in closest_windows if fnames[window] == fname]
                if len(windows_from_fname) > 1:
                    points_from_fname = [
                        get_dist_point_from_window(distribution, selected_points, closest_windows, window) 
                        for window in windows_from_fname]
                    distances_from_dist_point = []
                    for idx, point in enumerate(points_from_fname):
                        corr_window = windows_from_fname[idx]
                        target = get_window_point(metrics, corr_window)
                        distances_from_dist_point.append(euclidean_distance(point, target))

                    sort_idx = np.argsort(distances_from_dist_point)
                    for idx in sort_idx[1:]:
                        windows_to_swap.append(windows_from_fname[idx])

            for window in windows_to_swap:
                new_window = get_window_to_swap(distribution, closest_windows, fnames, selected_windows, 
                                                selected_fnames, selected_points, metrics, window)
                selected_windows.append(new_window)
                selected_fnames.append(fnames[new_window])
                closest_windows[closest_windows.index(window)] = new_window

            unique_fnames = np.unique([fnames[idx] for idx in closest_windows])

        solution_distances.append(farthest_unselected_point(closest_windows, distances_between_windows))
        solution_windows.append(closest_windows)

    # choose the solution in which the farthest unselected point is closest to the drawn distribution
    minind = np.argmin(solution_distances)

    return seeds[minind], solution_windows[minind], solution_distances

def get_solution_from_seed(seed: int, n_images: int, metrics: List[List[int]], distribution: NDArray[np.float64], 
                           fnames: List[str]) -> List[int]:
    
    """Similar to 'draw_windows' but returns the solution generated from a particular seed ('seed')."""
    
    # print(f"--> TODO (check type): {distribution.dtype}. It should be np.float64.")
    Tmetrics = np.array(metrics).T
    np.random.seed(seed)
    selected_points = np.random.choice(np.arange(0, len(distribution), 1), n_images, replace=False)

    # closest images to the drawn points
    closest_windows = []
    for orig_ind in selected_points:
        v1 = distribution[orig_ind]
        distances = np.sqrt(((v1 - Tmetrics)**2).sum(axis=1))
        closest_windows.append(np.argmin(distances))

    # iteratively swap windows from the same image
    unique_fnames = np.unique([fnames[idx] for idx in closest_windows])
    selected_fnames = list(unique_fnames.copy())
    selected_windows = closest_windows.copy()
    while len(unique_fnames) != len(closest_windows):
        windows_to_swap = []
        for fname in unique_fnames:
            windows_from_fname = [window for window in closest_windows if fnames[window] == fname]
            if len(windows_from_fname) > 1:
                points_from_fname = [
                    get_dist_point_from_window(distribution, selected_points, closest_windows, window) 
                    for window in windows_from_fname]
                distances_from_dist_point = []
                for idx, point in enumerate(points_from_fname):
                    corr_window = windows_from_fname[idx]
                    target = get_window_point(metrics, corr_window)
                    distances_from_dist_point.append(euclidean_distance(point, target))

                sort_idx = np.argsort(distances_from_dist_point)
                for idx in sort_idx[1:]:
                    windows_to_swap.append(windows_from_fname[idx])

        for window in windows_to_swap:
            new_window = get_window_to_swap(distribution, closest_windows, fnames, selected_windows, selected_fnames, 
                                            selected_points, metrics, window)
            selected_windows.append(new_window)
            selected_fnames.append(fnames[new_window])
            closest_windows[closest_windows.index(window)] = new_window

        unique_fnames = np.unique([fnames[idx] for idx in closest_windows])

    return closest_windows

def open_image(path: Path, filename: str) -> NDArray:
    """Open an image from 'path/filename' with Pillow. Returns a numpy array."""
    img = np.asarray(Image.open(path/filename), dtype=np.uint8)
    return img

def get_image_from_id(id: int, root_folder: Path, win_pos: List[slice], fnames: List[str]) -> NDArray:
    """Returns the window with index 'id'."""

    p0 = win_pos[id][0]
    p1 = win_pos[id][1]
    w = (slice(p0[0], p0[1], p0[2]), slice(p1[0], p1[1], p1[2]))
    fname = fnames[id]
    img = open_image(root_folder / 'images', fname)

    return img[w]

def draw_sample_points(radius: int, density: int, resample_resolution: float, n_iter: int, n_images: int, 
                       root_folder: Path, verbose: bool = True) -> Tuple[int, List[int]]:
    """
    Draw 'n_images' windows from the CORTEX dataset.

    Parameters
    -----------

    radius: int
        the radius of the sphere used to fit the distribution of all windows.

    density: int
        controls the number of points inside the sphere of radius 'radius'.

    resample_resolution: float
        ratio in which the metrics of each window will be resampled to. After
        z-score normalization, each metric value will be transformed as
        int(value / resampled_resolution).

    n_iter: int
        the number of solutions that will be drawn. This function returns the solution that
        minimizes the 'farthest unselected point' metric.

    n_images:
        the number of images to be drawn.
    
    root_folder: Path
        Path object of the directory containing the original images.

    verbose: bool
        if True, this function will output the progression bars and some useful
        charts.

    Returns
    -----------

    seed: int
        the seed used to draw the selected points. It can be used afterwards to reproduce
        the draw.

    solution: list of int
        a list of the indices of the drawn windows.
    """

    density = complex(0, density)
    hsphere = get_sphere(radius, density)
    
    if verbose:
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.3)

        # sort the hypersphere dimension-wise
        Tsphere = np.array(hsphere).T

        grid_idx = 0
        for i in range(4):
            for j in range(4):
                grid[grid_idx].scatter(Tsphere[i], Tsphere[j])
                grid[grid_idx].set_title(f'{i} / {j}')
                grid_idx += 1

        # plot the sphere
        plt.show()

    with open("measures.json", "r") as fjson:
        all_images_dicts = json.load(fjson)

    densities, contrasts, skel_heterogeneities, noises = get_normalized_metrics(all_images_dicts)
    densities, contrasts, skel_heterogeneities, noises = resampled_metrics(densities, contrasts, skel_heterogeneities, 
                                                                           noises, resample_resolution)
    metrics = [densities, contrasts, skel_heterogeneities, noises]
    Tmetrics = np.array(metrics).T

    if verbose:
        print('Approximating distribution...')
    
    distribution_points = approximate_distribution(hsphere, densities, contrasts, skel_heterogeneities, noises, verbose)
    
    # sort the distribution metric-wise
    distribution = np.array(list(distribution_points.keys()))

    if verbose:
        # check the distribution
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.3)
        
        grid_idx = 0
        for i in range(4):
            for j in range(4):
                grid[grid_idx].scatter(distribution.T[i], distribution.T[j], alpha=1, s=1)
                grid[grid_idx].scatter(metrics[i], metrics[j], alpha=1, s=1)
                grid[grid_idx].set_title(f'{i} / {j}')
                grid_idx += 1

        plt.show()

    # distance between all windows
    distances_between_windows = squareform(pdist(Tmetrics))

    # info of all windows (original filenames and positions)
    fnames, win_pos = get_windows_info(all_images_dicts)

    if verbose:
        print('Drawing windows...')
    
    seed, solution, solution_distances = draw_windows(n_iter, n_images, metrics, distribution, fnames, 
                                                      distances_between_windows, verbose)
    
    if verbose:
        print(f"To reproduce this drawing, use this seed: {seed}")

    if verbose:
        plt.figure(figsize=(10, 10))
        hist, n_edges = np.histogram(solution_distances, bins=300)
        plt.bar(n_edges[1:], hist, width=n_edges[1]-n_edges[0])
        plt.title('Histogram - farthest unselected window')
        plt.show()

    if verbose:
        # plot the selected images
        fig = plt.figure(figsize=(20, 25))
        grid = ImageGrid(fig, 111, nrows_ncols=(10, 10), axes_pad=0.3)

        grid_idx = 0
        for selected_ind in solution:
            window = get_image_from_id(selected_ind, root_folder, win_pos, fnames)
            grid[grid_idx].imshow(window, cmap='gray', vmax=100)
            grid[grid_idx].set_title(selected_ind)
            grid_idx += 1
        
        plt.show()

        # plot the distribution of the selected images (with respect to each metric)
        titles = ['Density', 'Contrast', 'Skeleton Heterogeneity', 'Noise']
        plt.figure(figsize=(10, 10))
        grid_idx = 1
        for idx, metric in enumerate(metrics):
            plt.subplot(2, 2, grid_idx)
            hist, n_edges = np.histogram(metric, bins=50)
            hist = hist / np.sum(hist)
            plt.bar(n_edges[1:], hist, width=n_edges[1]-n_edges[0], alpha=0.6)
            hist, n_edges = np.histogram(metrics[idx][solution], bins=50)
            hist = hist / np.sum(hist)
            plt.bar(n_edges[1:], hist, width=n_edges[1]-n_edges[0], alpha=0.6)
            plt.title(titles[idx])
            grid_idx += 1
        
        plt.show()

        # visualize the data distribution with a PCA projection (3 components)
        pca = PCA(n_components=3)
        pca.fit(Tmetrics)
        components = pca.transform(Tmetrics)
        proj_positions = components[solution]

        print(f'PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_)}')

        for idx, m in enumerate(metrics):
            fig = plt.figure(figsize=(10, 10))
            grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.3)
            grid_idx = 0
            colors = m / np.max(m)
            for i in range(3):
                for j in range(3):
                    grid[grid_idx].scatter(components.T[i], components.T[j], c=colors, s=1)
                    grid[grid_idx].scatter(proj_positions.T[i], proj_positions.T[j], c='red', s=1)
                    grid[grid_idx].set_title(f'{i}/{j}')
                    grid_idx += 1
                    plt.suptitle(titles[idx])
            
            plt.show()

    return seed, solution

def get_sample_points_from_seed(seed: int, radius: int, density: int, resample_resolution: float,
                                n_images: int, root_folder: Path, verbose: bool = True) -> List[int]:
    
    """
    Get the solution obtained by drawing the windows using a particular seed ('seed').

    Parameters
    -----------

    seed: int
        the seed that is used to draw points from the calculated distribution.

    radius: int
        the radius of the sphere used to fit the distribution of all windows.

    density: int
        controls the number of points inside the sphere of radius 'radius'.

    resample_resolution: float
        ratio in which the metrics of each window will be resampled to. After
        z-score normalization, each metric value will be transformed as
        int(value / resampled_resolution).

    n_images:
        the number of images to be drawn.

    root_folder: Path
        Path object of the directory containing the original images.

    verbose: bool
        if True, plots the progression bar when calculating the distribution of
        windows.
    """

    density = complex(0, density)
    hsphere = get_sphere(radius, density)

    with open("measures.json", "r") as fjson:
        all_images_dicts = json.load(fjson)

    densities, contrasts, skel_heterogeneities, noises = get_normalized_metrics(all_images_dicts)
    densities, contrasts, skel_heterogeneities, noises = resampled_metrics(densities, contrasts, skel_heterogeneities, 
                                                                           noises, resample_resolution)
    metrics = [densities, contrasts, skel_heterogeneities, noises]
    
    orig_metrics = list(get_original_metrics(all_images_dicts))
    
    print(f"Number of windows: {len(densities)}")

    if verbose:
        print('Approximating distribution...')
    
    distribution_points = approximate_distribution(hsphere, densities, contrasts, skel_heterogeneities, noises, verbose)
    
    # sort the distribution metric-wise

    distribution = np.array(list(distribution_points.keys()))
    # info of all windows (original filenames and positions)
    fnames, win_pos = get_windows_info(all_images_dicts)

    print(f"Number of images: {len(np.unique(fnames))}")
    
    solution = get_solution_from_seed(seed, n_images, metrics, distribution, fnames)

    if verbose:
        # plot the distribution of the selected images (with respect to each metric)
        titles = ['Density', 'Contrast', 'Medial line heterogeneity', 'Noise']

        plt.figure(figsize=(10, 12))
        grid_idx = 1
        for idx, metric in enumerate(orig_metrics):
            plt.subplot(2, 2, grid_idx)
            hist, n_edges = np.histogram(metric, bins=50)
            hist = hist / np.sum(hist)
            plt.bar(n_edges[1:], hist, width=n_edges[1]-n_edges[0], alpha=0.5)
            hist, n_edges = np.histogram(orig_metrics[idx][solution], bins=50)
            hist = hist / np.sum(hist)
            plt.bar(n_edges[1:], hist, width=n_edges[1]-n_edges[0], alpha=0.5)
            plt.ylim((0, 0.16))
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)
            plt.xlabel(f'{titles[idx]}', fontsize=11)
            plt.ylabel('Frequency', fontsize=11)
            grid_idx += 1
        
        plt.savefig('paper_images/metrics_histogram.pdf', bbox_inches='tight')
        plt.show()

        # typical / atypical
        
        at_low = 0.12
        low = 0.18
        high = 0.22
        
        typical = np.sum((orig_metrics[1] >= low) & (orig_metrics[1] <= high))
        atypical = np.sum((orig_metrics[1] <= at_low))
        orig_ratio = typical / atypical

        print(f"N typical: {typical}, N atypical: {atypical}")
        print(f"Orig ratio: {orig_ratio}")
        
        typical = np.sum((orig_metrics[1][solution] >= low) & (orig_metrics[1][solution] <= high))
        atypical = np.sum((orig_metrics[1][solution] <= at_low))
        solution_ratio = typical / atypical

        print(f"N typical: {typical}, N atypical: {atypical}")
        print(f"Solution ratio: {solution_ratio}")

        # Tmetrics = np.array(metrics).T
        Tmetrics = np.array(list(get_normalized_metrics(all_images_dicts))).T

        # visualize the data distribution with a PCA projection (3 components)
        pca = PCA(n_components=3)
        pca.fit(Tmetrics)
        components = pca.transform(Tmetrics)
        proj_positions = components[solution]

        print(f'PCA explained variance ratio: {np.sum(pca.explained_variance_ratio_)}')

        fig = plt.figure(figsize=(10, 10))

        for idx, m in enumerate(metrics):
            colors = m / np.max(m)
            ax = plt.subplot(2, 2, idx+1)
            ax.scatter(components.T[0], components.T[1], c=colors, s=1, rasterized=True)
            ax.scatter(proj_positions.T[0], proj_positions.T[1], c='red', s=4, rasterized=True)
            ax.set_xlabel('PC 1', fontsize=11)
            ax.set_ylabel('PC 2', fontsize=11)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_title(f'{titles[idx]}', fontweight='bold')
            ax.tick_params(labelsize=11)

        # for idx, m in enumerate(metrics):
        #     colors = m / np.max(m)
            
        #     ax = plt.subplot(4, 3, grid_idx)
        #     ax.scatter(components.T[0], components.T[1], c=colors, s=1)
        #     ax.scatter(proj_positions.T[0], proj_positions.T[1], c='red', s=4)
        #     # plt.xlabel('Feature 0')
        #     # plt.ylabel('Feature 1')
        #     ax.spines['right'].set_visible(False)
        #     ax.spines['top'].set_visible(False)
            
        #     grid_idx += 1

        #     ax = plt.subplot(4, 3, grid_idx)
        #     ax.scatter(components.T[0], components.T[2], c=colors, s=1)
        #     ax.scatter(proj_positions.T[0], proj_positions.T[2], c='red', s=4)
        #     # plt.xlabel('Feature 0')
        #     # plt.ylabel('Feature 2')
        #     # plt.axis('off')
        #     ax.spines['right'].set_visible(False)
        #     ax.spines['top'].set_visible(False)

        #     ax.set_title(titles[idx], fontweight='bold')

            
        #     grid_idx += 1

        #     ax = plt.subplot(4, 3, grid_idx)
        #     ax.scatter(components.T[1], components.T[2], c=colors, s=1)
        #     ax.scatter(proj_positions.T[1], proj_positions.T[2], c='red', s=4)
        #     # ax.xlabel('Feature 1')
        #     # ax.ylabel('Feature 2')
        #     # plt.axis('off')
        #     ax.spines['right'].set_visible(False)
        #     ax.spines['top'].set_visible(False)

        #     grid_idx += 1
        
        # ax = plt.subplot(4, 3, 4)
        # ax.set_ylabel('Feature 1')
        # ax = plt.subplot(4, 3, 5)
        # ax.set_ylabel('Feature 2')
        # ax = plt.subplot(4, 3, 6)
        # ax.set_ylabel('Feature 2')

        # ax = plt.subplot(4, 3, 10)
        # ax.set_xlabel('Feature 0')
        # ax = plt.subplot(4, 3, 11)
        # ax.set_xlabel('Feature 0')
        # ax = plt.subplot(4, 3, 12)
        # ax.set_xlabel('Feature 1')
            
        plt.savefig(f'paper_images/pca.pdf', dpi=400, bbox_inches='tight')
        plt.show()

        # plot the selected images
        fig = plt.figure(figsize=(20, 25))
        grid = ImageGrid(fig, 111, nrows_ncols=(10, 10), axes_pad=0.1, share_all=True)

        grid_idx = 0
        for selected_ind in solution:
            window = get_image_from_id(selected_ind, root_folder, win_pos, fnames)
            grid[grid_idx].imshow(window, cmap='gray', vmax=130)
            # grid[grid_idx].set_title(selected_ind)
            grid_idx += 1
        
        grid[0].get_xaxis().set_ticks([])
        grid[0].get_yaxis().set_ticks([])

        plt.axis('off')
        plt.savefig('paper_images/sampled_images.pdf', dpi=300, bbox_inches='tight')
        plt.show()

    return solution