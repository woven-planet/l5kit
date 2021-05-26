from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Set

import bokeh.io
import bokeh.plotting
import numpy as np
from bokeh.layouts import column, LayoutDOM
from bokeh.models import CustomJS, HoverTool, Slider
from bokeh.plotting import ColumnDataSource

from l5kit.visualization.visualizer.common import (AgentVisualization, CWVisualization, EgoVisualization,
                                                   FrameVisualization, LaneVisualization, TrajectoryVisualization)


def _visualization_list_to_dict(visualisation_list: List[Any], null_el: Any) -> Dict[str, Any]:
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


def visualize(scene_index: int, frames: List[FrameVisualization]) -> LayoutDOM:
    """Visualise a scene using Bokeh.

    :param scene_index: the index of the scene, used only as the title
    :param frames: a list of FrameVisualization objects (one per frame of the scene)
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
        ego_dict = _visualization_list_to_dict([frame.ego], EgoVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                             color="black", center_x=0,
                                                                             center_y=0))

        agents_dict = _visualization_list_to_dict(frame.agents, AgentVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                   color="black", track_id=-2,
                                                                                   agent_type="", prob=0.))

        lanes_dict = _visualization_list_to_dict(frame.lanes, LaneVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                color="black"))

        crosswalk_dict = _visualization_list_to_dict(frame.crosswalks, CWVisualization(xs=np.empty(0), ys=np.empty(0),
                                                                                       color="black"))

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
                          lanes=ColumnDataSource(lanes_dict),
                          crosswalks=ColumnDataSource(crosswalk_dict))
        frame_dict.update({k: ColumnDataSource(v) for k, v in trajectory_dict.items()})

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

            var center_x = frames[cb_obj.value]["ego"].data["center_x"][0];
            var center_y = frames[cb_obj.value]["ego"].data["center_y"][0];

            figure.x_range.setv({"start": center_x-50, "end": center_x+50})
            figure.y_range.setv({"start": center_y-50, "end": center_y+50})

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
    return layout
