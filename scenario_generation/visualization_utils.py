import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
######################################################################


def plot_poly_elems(ax, elems_d,  facecolor='0.4', alpha=0.3, edgecolor='black', label=None, is_closed=False, linewidth=1):
    elems_valid = elems_d['elems_valid']
    n_points_per_elem = elems_d['n_points_per_elem']
    elems_points = elems_d['elems_points']

    first_plt = True
    n_elem = elems_valid.shape[0]
    for i_elem in range(n_elem):
        if not elems_valid[i_elem]:
            continue
        x = elems_points[i_elem, :n_points_per_elem[i_elem], 0]
        y = elems_points[i_elem, :n_points_per_elem[i_elem], 1]
        if first_plt:
            first_plt = False
        else:
            label = None
        if is_closed:
            ax.fill(x, y, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, linewidth=linewidth, label=label)
        else:
            ax.plot(x, y, alpha=alpha, color=edgecolor, linewidth=linewidth, label=label)


##############################################################################################


def plot_lanes(ax, l_elems_d, r_elems_d, facecolor='0.4', alpha=0.3,
               edgecolor='black', label='', linewidth=1):


    l_elems_valid = l_elems_d['elems_valid']
    l_n_points_per_elem = l_elems_d['n_points_per_elem']
    l_elems_points = l_elems_d['elems_points']
    r_elems_valid = r_elems_d['elems_valid']
    r_n_points_per_elem = r_elems_d['n_points_per_elem']
    r_elems_points = r_elems_d['elems_points']

    # Print road area in between lanes
    first_plt = True
    n_elems = l_elems_valid.shape[0]
    for i_elem in range(n_elems):
        if not (l_elems_valid[i_elem] and r_elems_valid[i_elem]):
            continue
        x_left = l_elems_points[i_elem, :l_n_points_per_elem[i_elem], 0]
        y_left = l_elems_points[i_elem, :l_n_points_per_elem[i_elem], 1]
        x_right = r_elems_points[i_elem, :r_n_points_per_elem[i_elem], 0]
        y_right = r_elems_points[i_elem, :r_n_points_per_elem[i_elem], 1]
        x = np.concatenate((x_left, x_right[::-1]))
        y = np.concatenate((y_left, y_right[::-1]))
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


def visualize_scene_feat(agents_feat_s, map_points_s, map_elems_availability_s, map_n_points_orig_s, dataset_props):

    polygon_types = dataset_props['polygon_types']
    closed_polygon_types = dataset_props['closed_polygon_types']

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

    plot_props = {'lanes_mid': ('lime', 0.4), 'lanes_left': ('black', 0.3), 'lanes_right': ('black', 0.3),
                  'crosswalks': ('orange', 0.4)}
    pd = {}
    for i_type, poly_type in enumerate(polygon_types):
        pd[poly_type] = {}

        pd[poly_type]['elems_valid'] = map_elems_availability_s[i_type]
        pd[poly_type]['n_points_per_elem'] = map_n_points_orig_s[i_type]
        pd[poly_type]['elems_points'] = map_points_s[i_type]

        plot_poly_elems(ax, pd[poly_type],
                        facecolor=plot_props[poly_type][0], alpha=plot_props[poly_type][1],
                        edgecolor=plot_props[poly_type][0], label=poly_type,
                        is_closed=poly_type in closed_polygon_types, linewidth=1)

    plot_lanes(ax, pd['lanes_left'],  pd['lanes_right'], facecolor='grey', alpha=0.3, edgecolor='black',  label='Lanes')


    extents = [af['extent'] for af in agents_feat_s]
    plot_rectangles(ax, centroids[1:], extents[1:], yaws[1:])
    plot_rectangles(ax, [centroids[0]], [extents[0]], [yaws[0]], label='ego', facecolor='red', edgecolor='red')

    ax.quiver(X[1:], Y[1:], U[1:], V[1:], units='xy', color='b', label='Non-ego', width=0.5)
    ax.quiver(X[0], Y[0], U[0], V[0], units='xy', color='r', label='Ego', width=0.5)

    ax.grid()
    plt.legend()
    plt.show()
##############################################################################################
