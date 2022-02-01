import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import torch

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
######################################################################


def plot_poly_elems(ax, elems_points, elems_valid,  facecolor='0.4', alpha=0.3, edgecolor='black', label=None, is_closed=False, linewidth=1):
    first_plt = True
    for i_elem, is_valid in enumerate(elems_valid):
        if not is_valid:
            continue
        x = torch.unique(elems_points[i_elem, :, 0])
        y = torch.unique(elems_points[i_elem, :, 1])
        if first_plt:
            first_plt = False
        else:
            label = None
        if is_closed:
            ax.fill(x, y, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, label=label)
        else:
            ax.plot(x, y, alpha=alpha, color=edgecolor, linewidth=linewidth, label=label)


##############################################################################################


def plot_lanes(ax, lanes_left, lanes_left_avl, lanes_right, lanes_right_avl, facecolor='0.4', alpha=0.3,
               edgecolor='black', label='', linewidth=1):

    # Print road borders (left and right lanes separately)
    plot_poly_elems(ax, lanes_left, lanes_left_avl, facecolor='0.4', alpha=0.7, edgecolor='black',
                    is_closed=False, linewidth=1)
    plot_poly_elems(ax, lanes_right, lanes_right_avl, facecolor='0.4', alpha=0.7, edgecolor='black',
                    is_closed=False, linewidth=1)

    # Print road area in between lanes
    first_plt = True
    n_elems = lanes_left.shape[0]
    for i_elem in range(n_elems):
        if not (lanes_left_avl[i_elem] and lanes_right_avl[i_elem]):
            continue
        x_left = torch.unique(lanes_left[i_elem, :, 0])
        y_left = torch.unique(lanes_left[i_elem, :, 1])
        x_right = torch.unique(lanes_right[i_elem, :, 0])
        y_right = torch.unique(lanes_right[i_elem, :, 1])
        x = torch.cat((x_left, torch.flip(x_right, dims=[0])))
        y = torch.cat((y_left, torch.flip(y_right, dims=[0])))
        if first_plt:
            first_plt = False
        else:
            label = None
        ax.fill(x, y, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, label=label)


##############################################################################################


def plot_rectangles(ax, centroids, extents, yaws, label='car', facecolor='skyblue', alpha=0.4, edgecolor='skyblue'):
    n_elems = len(centroids)
    first_plt = True
    for i in range(n_elems):
        if first_plt:
            first_plt = False
        else:
            label = None
        height = extents[i][0]
        width = extents[i][1]
        angle = yaws[i]
        angle_deg = float(np.degrees(angle))
        xy = centroids[i] \
             - 0.5 * height * np.array([np.cos(angle), np.sin(angle)]) \
             - 0.5 * width * np.array([-np.sin(angle), np.cos(angle)])
        rect = Rectangle(xy, height, width, angle_deg, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor, linewidth=1, label=label)

        ax.add_patch(rect)


##############################################################################################


def visualize_scene_feat(agents_feat_s, map_points_s, map_points_availability_s, dataset_props):
    polygon_types = dataset_props['polygon_types']
    lanes_left = map_points_s[polygon_types.index('lanes_left')]
    lanes_left_avl = map_points_availability_s[polygon_types.index('lanes_left')]
    lanes_right = map_points_s[polygon_types.index('lanes_right')]
    lanes_right_avl = map_points_availability_s[polygon_types.index('lanes_right')]
    lanes_mid = map_points_s[polygon_types.index('lanes_mid')]
    lanes_mid_avl = map_points_availability_s[polygon_types.index('lanes_mid')]
    crosswalks = map_points_s[polygon_types.index('crosswalks')]
    crosswalks_avl = map_points_availability_s[polygon_types.index('crosswalks')]

    centroids = [af['centroid'] for af in agents_feat_s]
    yaws = [af['yaw'] for af in agents_feat_s]
    print('agents centroids: ', centroids)
    print('agents yaws: ', yaws)
    print('agents speed: ', [af['speed'] for af in agents_feat_s])
    print('agents types: ', [af['agent_label_id'] for af in agents_feat_s])
    X = [p[0] for p in centroids]
    Y = [p[1] for p in centroids]
    U = [af['speed'] * np.cos(af['yaw']) for af in agents_feat_s]
    V = [af['speed'] * np.sin(af['yaw']) for af in agents_feat_s]
    fig, ax = plt.subplots()

    plot_poly_elems(ax, crosswalks, crosswalks_avl, facecolor='orange', alpha=0.3, edgecolor='orange', label='Crosswalks',
                    is_closed=True)

    plot_lanes(ax, lanes_left, lanes_left_avl, lanes_right, lanes_right_avl, facecolor='grey', alpha=0.3, edgecolor='black',
               label='Lanes')
    plot_poly_elems(ax, lanes_mid, lanes_mid_avl, facecolor='lime', alpha=0.4, edgecolor='lime', label='Lanes mid',
                    is_closed=False, linewidth=1)


    extents = [af['extent'] for af in agents_feat_s]
    plot_rectangles(ax, centroids[1:], extents[1:], yaws[1:])
    plot_rectangles(ax, [centroids[0]], [extents[0]], [yaws[0]], label='ego', facecolor='red', edgecolor='red')

    ax.quiver(X[1:], Y[1:], U[1:], V[1:], units='xy', color='b', label='Non-ego', width=0.5)
    ax.quiver(X[0], Y[0], U[0], V[0], units='xy', color='r', label='Ego', width=0.5)

    ax.grid()
    plt.legend()
    plt.show()
##############################################################################################
