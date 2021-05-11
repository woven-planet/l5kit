from collections import defaultdict

from typing import Any, DefaultDict, Dict, List, NamedTuple, no_type_check, Set, Tuple

import bokeh.io
import bokeh.plotting
import numpy as np
from bokeh.layouts import column, LayoutDOM
from bokeh.models import CustomJS, HoverTool, Slider
from bokeh.plotting import ColumnDataSource

from l5kit.visualization.visualizer.common import (AgentVisualization, EgoVisualization, FrameVisualization,
                                                   MapElementVisualization, TrajectoryVisualization)
from pathlib import Path

import geckodriver_autoinstaller
geckodriver_autoinstaller.install()


@no_type_check
def _visualization_list_to_dict(visualisation_list: List[NamedTuple], null_el: NamedTuple) -> Dict[str, Any]:
    """Convert a list of NamedTuple into a dict, where:
    - the NamedTuple fields are the dict keys;
    - the dict value are lists;

    :param visualisation_list: a list of NamedTuple
    :param null_el: an element to be used as null if the list is empty (it can crash visualisation)
    :return: a dict with the same information
    """
    visualisation_list = visualisation_list if len(visualisation_list) else [null_el]
    visualisation_dict: DefaultDict[str, Any] = defaultdict(list)

    keys_set: Set[str] = set(visualisation_list[0]._asdict().keys())
    for el in visualisation_list:
        for k, v in el._asdict().items():
            if k not in keys_set:
                raise ValueError("keys set is not consistent between elements in the list")
            visualisation_dict[k].append(v)
    return dict(visualisation_dict)


def visualize(scene_index: int, frames: List[FrameVisualization], halfside: int = 50) -> LayoutDOM:
    """Visualise a scene using Bokeh.

    :param scene_index: the index of the scene, used only as the title
    :param frames: a list of FrameVisualization objects (one per frame of the scene)
    :param halfside: side in metres of the visualisation area

    """

    agent_hover = HoverTool(
        mode="mouse",
        names=["agents"],
        tooltips=[
            ("Type", "@agent_type"),
            ("Probability", "@prob{0.00}%"),
            ("Track id", "@track_id"),
        ],
    )
    out: List[Dict[str, ColumnDataSource]] = []

    trajectories_labels = np.unique([traj.legend_label for frame in frames for traj in frame.trajectories])

    for frame_idx, frame in enumerate(frames):
        # we need to ensure we have something otherwise js crashes
        ego_dict = _visualization_list_to_dict(frame.ego, EgoVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                             color="black", center_x=0,
                                                                             center_y=0, alpha=0.))

        agents_dict = _visualization_list_to_dict(frame.agents, AgentVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                   color="black", track_id=-2,
                                                                                   agent_type="", prob=0.,
                                                                                   alpha=0.))

        patches_dict = _visualization_list_to_dict(frame.map_patches,
                                                   MapElementVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                           color="black", alpha=0.))

        lines_dict = _visualization_list_to_dict(frame.map_lines,
                                                 MapElementVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                         color="black", alpha=0.))

        # for trajectory we extract the labels so that we can show them in the legend
        trajectory_dict: Dict[str, Dict[str, Any]] = {}
        for trajectory_label in trajectories_labels:
            trajectories = [el for el in frame.trajectories if el.legend_label == trajectory_label]
            trajectory_dict[trajectory_label] = _visualization_list_to_dict(trajectories,
                                                                            TrajectoryVisualization(xs=np.empty(0),
                                                                                                    ys=np.empty(0),
                                                                                                    color="black",
                                                                                                    legend_label="none",
                                                                                                    track_id=-2))

        frame_dict = dict(ego=ColumnDataSource(ego_dict), agents=ColumnDataSource(agents_dict),
                          map_patches=ColumnDataSource(patches_dict), map_lines=ColumnDataSource(lines_dict))
        frame_dict.update({k: ColumnDataSource(v) for k, v in trajectory_dict.items()})

        out.append(frame_dict)

    center_x = out[0]["ego"].data["center_x"][0]
    center_y = out[0]["ego"].data["center_y"][0]

    f = bokeh.plotting.figure(
        #title="Scene {}".format(scene_index),
        match_aspect=True,
        x_range=(center_x - halfside, center_x + halfside),
        y_range=(center_y - halfside, center_y + halfside),
        #tools=["pan", "wheel_zoom", agent_hover, "save", "reset"],
        #active_scroll="wheel_zoom",
        output_backend="svg",
        toolbar_location=None
    )

    f.axis.visible = False
    f.xgrid.grid_line_color = None
    f.ygrid.grid_line_color = None

    f.patches(line_width=1, alpha="alpha", color="color", source=out[0]["map_patches"])
    f.multi_line(line_width=1, alpha="alpha", source=out[0]["map_lines"], color="color")
    f.patches(line_width=2, fill_alpha="alpha", color="color", source=out[0]["ego"])
    f.patches(line_width=2, fill_alpha="alpha", color="color", name="agents", source=out[0]["agents"])

    js_string = """
            sources["map_patches"].data = frames[cb_obj.value]["map_patches"].data;
            sources["map_lines"].data = frames[cb_obj.value]["map_lines"].data;

            sources["agents"].data = frames[cb_obj.value]["agents"].data;
            sources["ego"].data = frames[cb_obj.value]["ego"].data;

            var center_x = frames[cb_obj.value]["ego"].data["center_x"][0];
            var center_y = frames[cb_obj.value]["ego"].data["center_y"][0];

            figure.x_range.setv({"start": center_x-halfside, "end": center_x+halfside})
            figure.y_range.setv({"start": center_y-halfside, "end": center_y+halfside})

            sources["map_patches"].change.emit();
            sources["map_lines"].change.emit();
            sources["agents"].change.emit();
            sources["ego"].change.emit();
        """

    for trajectory_name in trajectories_labels:
        f.multi_line(alpha=0.8, line_width=3, source=out[0][trajectory_name], color="color",
                     legend_label=trajectory_name, line_dash="dotted")
        js_string += f'sources["{trajectory_name}"].data = frames[cb_obj.value]["{trajectory_name}"].data;\n' \
                     f'sources["{trajectory_name}"].change.emit();\n'

    slider_callback = CustomJS(
        args=dict(figure=f, sources=out[0], frames=out, halfside=halfside),
        code=js_string,
    )

    slider = Slider(start=0, end=len(frames) - 1, value=0, step=1, title="frame")
    slider.js_on_change("value", slider_callback)

    f.legend.location = "top_left"
    f.legend.click_policy = "hide"

    layout = column(f, slider)
    return layout


# TODO this function has a lot of repeated stuff
def save_svgs(scene_index: int, frames: List[FrameVisualization], path: Path, halfside: int = 50) -> None:
    """Get the raw images instead of the html

    :param scene_index: the scene index
    :param frames: the frames to visualise
    :param resolution: the resolution as W,H
    :return: a stacked numpy array of all the frames
    """

    trajectories_labels = np.unique([traj.legend_label for frame in frames for traj in frame.trajectories])

    for frame_idx, frame in enumerate(frames):
        # we need to ensure we have something otherwise js crashes
        ego_dict = _visualization_list_to_dict(frame.ego, EgoVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                           color="black", center_x=0,
                                                                           center_y=0, alpha=0.))

        agents_dict = _visualization_list_to_dict(frame.agents, AgentVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                   color="black", track_id=-2,
                                                                                   agent_type="", prob=0.,
                                                                                   alpha=0.))

        patches_dict = _visualization_list_to_dict(frame.map_patches,
                                                   MapElementVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                           color="black", alpha=0.))

        lines_dict = _visualization_list_to_dict(frame.map_lines,
                                                 MapElementVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                         color="black", alpha=0.))

        # for trajectory we extract the labels so that we can show them in the legend
        trajectory_dict: Dict[str, Dict[str, Any]] = {}
        for trajectory_label in trajectories_labels:
            trajectories = [el for el in frame.trajectories if el.legend_label == trajectory_label]
            trajectory_dict[trajectory_label] = _visualization_list_to_dict(trajectories,
                                                                            TrajectoryVisualization(xs=np.empty(0),
                                                                                                    ys=np.empty(0),
                                                                                                    color="black",
                                                                                                    legend_label="none",
                                                                                                    track_id=-2))

        frame_dict = dict(ego=ColumnDataSource(ego_dict), agents=ColumnDataSource(agents_dict),
                          map_patches=ColumnDataSource(patches_dict),
                          map_lines=ColumnDataSource(lines_dict))
        frame_dict.update({k: ColumnDataSource(v) for k, v in trajectory_dict.items()})

        f = bokeh.plotting.figure(
            #title="Scene {}".format(scene_index),
            match_aspect=True,
            x_range=(frame_dict["ego"].data["center_x"][0] - halfside, frame_dict["ego"].data["center_x"][0] + halfside),
            y_range=(frame_dict["ego"].data["center_y"][0] - halfside, frame_dict["ego"].data["center_y"][0] + halfside),
            toolbar_location=None,
            output_backend="svg"
        )
        f.axis.visible = False
        f.xgrid.grid_line_color = None
        f.ygrid.grid_line_color = None

        f.patches(line_width=1, alpha="alpha", color="color", source=frame_dict["map_patches"])
        f.multi_line(line_width=1, alpha="alpha", source=frame_dict["map_lines"], color="color")
        f.patches(line_width=2, fill_alpha="alpha", color="color", source=frame_dict["ego"])
        f.patches(line_width=2, fill_alpha="alpha", color="color", name="agents", source=frame_dict["agents"])

        for trajectory_name in trajectories_labels:
            f.multi_line(alpha=0.8, line_width=3, source=frame_dict[trajectory_name], color="color",
                         legend_label=trajectory_name, line_dash="dotted")

        bokeh.io.export_svg(f, filename=str(path / f"frame_{frame_idx}.svg"))


# TODO this function has a lot of repeated stuff
def get_raw_buffer(scene_index: int, frames: List[FrameVisualization], halfside: int = 50, resolution: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Get the raw images instead of the html

    :param scene_index: the scene index
    :param frames: the frames to visualise
    :param resolution: the resolution as W,H
    :return: a stacked numpy array of all the frames
    """

    trajectories_labels = np.unique([traj.legend_label for frame in frames for traj in frame.trajectories])
    frame_images: List[np.ndarray] = []

    for frame_idx, frame in enumerate(frames):
        # we need to ensure we have something otherwise js crashes
        ego_dict = _visualization_list_to_dict(frame.ego, EgoVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                           color="black", center_x=0,
                                                                           center_y=0, alpha=0.))

        agents_dict = _visualization_list_to_dict(frame.agents, AgentVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                   color="black", track_id=-2,
                                                                                   agent_type="", prob=0.,
                                                                                   alpha=0.))

        patches_dict = _visualization_list_to_dict(frame.map_patches,
                                                   MapElementVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                           color="black", alpha=0.))

        lines_dict = _visualization_list_to_dict(frame.map_lines,
                                                 MapElementVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                         color="black", alpha=0.))

        # for trajectory we extract the labels so that we can show them in the legend
        trajectory_dict: Dict[str, Dict[str, Any]] = {}
        for trajectory_label in trajectories_labels:
            trajectories = [el for el in frame.trajectories if el.legend_label == trajectory_label]
            trajectory_dict[trajectory_label] = _visualization_list_to_dict(trajectories,
                                                                            TrajectoryVisualization(xs=np.empty(0),
                                                                                                    ys=np.empty(0),
                                                                                                    color="black",
                                                                                                    legend_label="none",
                                                                                                    track_id=-2))

        frame_dict = dict(ego=ColumnDataSource(ego_dict), agents=ColumnDataSource(agents_dict),
                          map_patches=ColumnDataSource(patches_dict),
                          map_lines=ColumnDataSource(lines_dict))
        frame_dict.update({k: ColumnDataSource(v) for k, v in trajectory_dict.items()})

        f = bokeh.plotting.figure(
            # title="Scene {}".format(scene_index),
            match_aspect=True,
            x_range=(frame_dict["ego"].data["center_x"][0] - halfside, frame_dict["ego"].data["center_x"][0] + halfside),
            y_range=(frame_dict["ego"].data["center_y"][0] - halfside, frame_dict["ego"].data["center_y"][0] + halfside),
            toolbar_location=None,
        )
        f.axis.visible = False
        f.xgrid.grid_line_color = None
        f.ygrid.grid_line_color = None

        f.patches(line_width=1, alpha="alpha", color="color", source=frame_dict["map_patches"])
        f.multi_line(line_width=1, alpha="alpha", source=frame_dict["map_lines"], color="color")
        f.patches(line_width=2, fill_alpha="alpha", color="color", source=frame_dict["ego"])
        f.patches(line_width=2, fill_alpha="alpha", color="color", name="agents", source=frame_dict["agents"])

        for trajectory_name in trajectories_labels:
            f.multi_line(alpha=0.8, line_width=3, source=frame_dict[trajectory_name], color="color",
                         legend_label=trajectory_name, line_dash="dotted")

        img = np.asarray(bokeh.io.export.get_screenshot_as_png(f, height=resolution[0], width=resolution[1]))
        frame_images.append(img[..., :3])

    return np.stack(frame_images)
