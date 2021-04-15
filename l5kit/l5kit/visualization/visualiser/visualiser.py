from typing import Dict, List, Optional

import bokeh.io
import bokeh.plotting
import numpy as np
from bokeh.layouts import column
from bokeh.models import CustomJS, HoverTool, Slider
from bokeh.plotting import ColumnDataSource, output_notebook, save, show
from bokeh.resources import Resources

from l5kit.visualization.visualiser.common import (AgentVisualisation, CWVisualisation, EgoVisualisation,
                                                   FrameVisualisation, LaneVisualisation, TrajectoryVisualisation)


def visualise(scene_index: int, frames: List[FrameVisualisation], do_show: bool = False,
              do_notebook: bool = False, save_path: Optional[str] = None,
              resources: str = "inline") -> None:
    """Visualise a scene using Bokeh.

    :param scene_index: the index of the scene, used only as the title
    :param frames: a list of FrameVisualisation objects (one per frame of the scene)
    :param do_show: if to call the show method
    :param do_notebook: if to call output_notebook
    :param save_path: optional html destination file
    :param resources: how to load js (inline will embed it, cdn will download it)
    """

    assert not (save_path is not None and do_notebook), "can't save and show in notebook"

    if do_notebook:
        output_notebook()

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

        ego_dict = {k:[v] for k, v in ego._asdict().items()}

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
    if save_path is not None:
        save(layout, filename=save_path, title=f"{scene_index}", resources=Resources(resources))
    if do_show:
        show(layout)
