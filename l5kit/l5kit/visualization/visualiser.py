from typing import List, NamedTuple, Dict

import bokeh.io
import bokeh.plotting
import numpy as np
from bokeh.layouts import column
from bokeh.models import CustomJS, HoverTool, Slider
from bokeh.plotting import ColumnDataSource, output_file, save

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.filter import (filter_agents_by_frames, filter_agents_by_labels, filter_tl_faces_by_frames,
                               filter_tl_faces_by_status)
from l5kit.data.labels import PERCEPTION_LABELS
from l5kit.data.map_api import MapAPI, TLFacesColors
from l5kit.rasterization import build_rasterizer
from l5kit.rasterization.box_rasterizer import get_ego_as_agent
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
from l5kit.sampling import get_relative_poses


COLORS = {
    TLFacesColors.GREEN.name: "#33CC33",
    TLFacesColors.RED.name: "#FF3300",
    TLFacesColors.YELLOW.name: "#FFFF66",
    "PERCEPTION_LABEL_CAR": "#1F77B4",
    "PERCEPTION_LABEL_CYCLIST": "#CC33FF",
    "PERCEPTION_LABEL_PEDESTRIAN": "#66CCFF",
}


class LaneVisualisation(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    color: str


class CWVisualisation(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    color: str


class AgentVisualisation(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    color: str
    track_id: int
    type: str
    prob: float


class EgoVisualisation(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    color: str
    center_x: float
    center_y: float


class TrajectoryVisualisation(NamedTuple):
    xs: np.ndarray
    ys: np.ndarray
    color: str
    legend_label: str


class FrameVisualisation(NamedTuple):
    ego: EgoVisualisation
    agents: List[AgentVisualisation]
    lanes: List[LaneVisualisation]
    crosswalks: List[CWVisualisation]
    trajectories: List[TrajectoryVisualisation]


def get_frame_trajectories(frames: np.ndarray, agents_frames: List[np.ndarray], track_ids: np.ndarray,
                           frame_index: int) -> List[TrajectoryVisualisation]:

    traj_visualisation: List[TrajectoryVisualisation] = []
    # TODO: factor out future length
    agent_traj_length = 20
    for track_id in track_ids:
        pos, *_, avail = get_relative_poses(agent_traj_length, frames[frame_index: frame_index + agent_traj_length],
                                            track_id, agents_frames[frame_index: frame_index + agent_traj_length],
                                            np.eye(3), 0)
        traj_visualisation.append(TrajectoryVisualisation(xs=pos[avail > 0, 0],
                                                          ys=pos[avail > 0, 1],
                                                          color="blue",
                                                          legend_label="agent_trajectory"))

    # TODO: factor out future length
    ego_traj_length = 100
    pos, *_, avail = get_relative_poses(ego_traj_length, frames[frame_index: frame_index + ego_traj_length],
                                        None, agents_frames[frame_index: frame_index + ego_traj_length],
                                        np.eye(3), 0)
    traj_visualisation.append(TrajectoryVisualisation(xs=pos[avail > 0, 0],
                                                      ys=pos[avail > 0, 1],
                                                      color="red",
                                                      legend_label="ego_trajectory"))

    return traj_visualisation


def visualise(scene_index: int, frames: List[FrameVisualisation]) -> None:
    """Visualise a scene using bokeh.

    :param scene_index: the index of the scene, used only as the title
    :param frames: a list of FrameVisualisation objects (one per frame of the scene)
    :return:
    """
    output_file("scene.html")
    agent_hover = HoverTool(
        mode="mouse",
        names=["agents"],
        tooltips=[
            ("Type", "@type"),
            ("Probability", "@prob{0.00}%"),
            ("Track id", "@track_id"),
        ],
    )
    out: List[Dict[str, ColumnDataSource]] = []

    trajectories_labels = np.unique([traj.legend_label for frame in frames for traj in frame.trajectories])

    for frame_idx in range(len(frames)):
        ego = frames[frame_idx].ego

        ego_dict = {field_name: [field] for field, field_name in zip(ego, EgoVisualisation._fields)}

        agents_dict = {k: [] for k in AgentVisualisation._fields}
        for agent in frames[frame_idx].agents:
            for k, v in zip(AgentVisualisation._fields, agent):
                agents_dict[k].append(v)

        lanes_dict = {k: [] for k in LaneVisualisation._fields}
        for lane in frames[frame_idx].lanes:
            for k, v in zip(LaneVisualisation._fields, lane):
                lanes_dict[k].append(v)

        crosswalk_dict = {k: [] for k in CWVisualisation._fields}
        for crosswalk in frames[frame_idx].crosswalks:
            for k, v in zip(CWVisualisation._fields, crosswalk):
                crosswalk_dict[k].append(v)

        # for trajectory we extract the labels so that we can show them in the legend
        trajectory_super_dict = {label: {k: [] for k in TrajectoryVisualisation._fields}
                                 for label in trajectories_labels}
        for trajectory in frames[frame_idx].trajectories:
            for k, v in zip(TrajectoryVisualisation._fields, trajectory):
                trajectory_super_dict[trajectory.legend_label][k].append(v)

        frame_dict = dict(ego=ColumnDataSource(ego_dict), agents=ColumnDataSource(agents_dict),
                          lanes=ColumnDataSource(lanes_dict),
                          crosswalks=ColumnDataSource(crosswalk_dict))
        frame_dict.update({k: ColumnDataSource(v) for k, v in trajectory_super_dict.items()})

        out.append(frame_dict)

    f = bokeh.plotting.figure(
        title="Scene {}".format(scene_index),
        match_aspect=True,
        x_range=(out[0]["ego"].data["center_x"][0] - 50, out[0]["ego"].data["center_x"][0] + 50),
        y_range=(out[0]["ego"].data["center_y"][0] - 50, out[0]["ego"].data["center_y"][0] + 50),
        tools=["pan", "wheel_zoom", agent_hover, "save", "reset"],
        active_scroll="wheel_zoom",
    )

    f.xgrid.grid_line_color = None
    f.ygrid.grid_line_color = None

    f.patches(line_width=0, alpha=0.5, color="color", source=out[0]["lanes"])
    f.patches(line_width=0, alpha=0.5, color="#B5B50D", source=out[0]["crosswalks"])
    f.patches(line_width=2, color="#B53331", source=out[0]["ego"])
    f.patches(line_width=2, color="color", name="agents", source=out[0]["agents"])

    js_string = """
            sources["lanes"].data = frames[cb_obj.value]["lanes"].data;
            sources["crosswalks"].data = frames[cb_obj.value]["crosswalks"].data;
            sources["agents"].data = frames[cb_obj.value]["agents"].data;
            sources["ego"].data = frames[cb_obj.value]["ego"].data;

            figure.x_range.setv({"start": frames[cb_obj.value]["ego"].data["center_x"][0]-50, "end": frames[cb_obj.value]["ego"].data["center_x"][0]+50})
            figure.y_range.setv({"start": frames[cb_obj.value]["ego"].data["center_y"][0]-50, "end": frames[cb_obj.value]["ego"].data["center_y"][0]+50})

            sources["lanes"].change.emit();
            sources["crosswalks"].change.emit();
            sources["agents"].change.emit();
            sources["ego"].change.emit();

        """

    for trajectory_name in trajectories_labels:
        f.multi_line(alpha=0.8, line_width=3, source=out[0][trajectory_name], color="color",
                     legend_label=trajectory_name)
        js_string += f'sources["{trajectory_name}"].data = frames[cb_obj.value]["{trajectory_name}"].data;\n' \
                     f'sources["{trajectory_name}"].change.emit();\n'

    slider_callback = CustomJS(
        args=dict(figure=f, sources=out[0], frames=out),
        code=js_string,
    )

    slider = Slider(start=0, end=len(frames), value=0, step=1, title="frame")
    slider.js_on_change("value", slider_callback)

    f.legend.location = "top_left"
    f.legend.click_policy = "hide"

    layout = column(f, slider)
    save(layout)


def get_frame_data(mapAPI: MapAPI, frame: np.ndarray, agents: np.ndarray, tls_frame: np.ndarray):
    ego_xy = frame["ego_translation"][:2]

    #################
    # plot lanes
    lane_indices = indices_in_bounds(ego_xy, mapAPI.bounds_info["lanes"]["bounds"], 50)
    active_tl_ids = set(filter_tl_faces_by_status(tls_frame, "ACTIVE")["face_id"].tolist())

    lanes_vis: List[LaneVisualisation] = []

    for idx, lane_idx in enumerate(lane_indices):
        lane_idx = mapAPI.bounds_info["lanes"]["ids"][lane_idx]

        lane_tl_ids = set(mapAPI.get_lane_traffic_control_ids(lane_idx))
        lane_colour = "gray"
        for tl_id in lane_tl_ids.intersection(active_tl_ids):
            lane_colour = COLORS[mapAPI.get_color_for_face(tl_id)]

        lane_coords = mapAPI.get_lane_coords(lane_idx)
        left_lane = lane_coords["xyz_left"][:, :2]
        right_lane = lane_coords["xyz_right"][::-1, :2]

        lanes_vis.append(LaneVisualisation(xs=np.hstack((left_lane[:, 0], right_lane[:, 0])),
                                           ys=np.hstack((left_lane[:, 1], right_lane[:, 1])),
                                           color=lane_colour))



    #################
    # plot crosswalks
    crosswalk_indices = indices_in_bounds(ego_xy, mapAPI.bounds_info["crosswalks"]["bounds"], 50)
    crosswalks_vis: List[CWVisualisation] = []

    for idx in crosswalk_indices:
        crosswalk = mapAPI.get_crosswalk_coords(mapAPI.bounds_info["crosswalks"]["ids"][idx])
        crosswalks_vis.append(CWVisualisation(xs=crosswalk["xyz"][:, 0],
                                              ys=crosswalk["xyz"][:, 1],
                                              color="yellow"))
    #################
    # plot ego and agent cars
    agents = np.insert(agents, 0, get_ego_as_agent(frame))

    # TODO: move to a function
    corners_base_coords = (np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * 0.5)[None, :, :]
    corners_m = corners_base_coords * agents["extent"][:, None, :2]  # corners in zero
    s = np.sin(agents["yaw"])
    c = np.cos(agents["yaw"])
    rotation_m = np.moveaxis(np.array(((c, s), (-s, c))), 2, 0)

    box_world_coords = corners_m @ rotation_m + agents["centroid"][:, None, :2]

    # ego
    ego_vis = EgoVisualisation(xs=box_world_coords[0, :, 0], ys=box_world_coords[0, :, 1],
                               color="red", center_x=agents["centroid"][0, 0],
                               center_y=agents["centroid"][0, 1])

    # agents
    agents = agents[1:]
    box_world_coords = box_world_coords[1:]

    agents_vis: List[AgentVisualisation] = []
    for agent, box_coord in zip(agents, box_world_coords):

        label_index = np.argmax(agent["label_probabilities"])
        agent_type = PERCEPTION_LABELS[label_index]
        agents_vis.append(AgentVisualisation(xs=box_coord[..., 0],
                                             ys=box_coord[..., 1],
                                             color="#1F77B4" if agent_type not in COLORS else COLORS[agent_type],
                                             track_id=agent["track_id"],
                                             type=PERCEPTION_LABELS[label_index],
                                             prob=agent["label_probabilities"][label_index]))

    return lanes_vis, crosswalks_vis, ego_vis, agents_vis


if __name__ == "__main__":
    zarr_dt = ChunkedDataset("/tmp/l5kit_data/scenes/sample.zarr").open()
    print(zarr_dt)

    cfg = load_config_data(
        "/Users/lucabergamini/Desktop/l5kit/examples/agent_motion_prediction/agent_motion_config.yaml"
    )

    rast = build_rasterizer(cfg, LocalDataManager("/tmp/l5kit_data"))
    mapAPI = rast.sem_rast.mapAPI

    scene_dataset = zarr_dt.get_scene_dataset(0)
    frames = scene_dataset.frames
    agents = scene_dataset.agents
    tl_faces = scene_dataset.tl_faces

    agents_frames = filter_agents_by_frames(frames, agents)
    tls_frames = filter_tl_faces_by_frames(frames, tl_faces)

    out: List[FrameVisualisation] = []
    for frame_idx in range(len(frames)):
        frame = frames[frame_idx]
        agents_frame = agents_frames[frame_idx]

        # TODO: hardcoded threshold
        agents_frame = filter_agents_by_labels(agents_frame, 0.1)
        tls_frame = tls_frames[frame_idx]

        traj_vis = get_frame_trajectories(frames, agents_frames, agents_frame["track_id"], frame_idx)

        lanes_vis, crosswalks_vis, ego_vis, agents_vis = get_frame_data(mapAPI, frame, agents_frame, tls_frame)
        out.append(FrameVisualisation(ego=ego_vis,
                                      agents=agents_vis,
                                      lanes=lanes_vis,
                                      crosswalks=crosswalks_vis,
                                      trajectories=traj_vis))

        # trajs, traj_ego = get_frame_trajectories(frames, agents_frames, agents_frame["track_id"], frame_idx)


    visualise(10, out)

    #visualise_scene(zarr_dt, scene_index=0, mapAPI=mapAPI)
