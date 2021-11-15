import torch
from l5kit.planning.vectorized.common import pad_avail, pad_points, transform_points
from l5kit.planning.vectorized.global_graph import VectorizedEmbedding


def table_to_features(input_dict, cfg):

    future_num_frames = input_dict["target_availabilities"].shape[1]
    history_num_frames_ego = cfg["model_params"]["history_num_frames_ego"]
    history_num_frames_agents = cfg["model_params"]["history_num_frames_agents"]
    vec_embedding = VectorizedEmbedding(256)


    # ==== Past and Static info
    agents_past_polys = torch.cat(
        [input_dict["agent_trajectory_polyline"].unsqueeze(1), input_dict["other_agents_polyline"]], dim=1
    )
    agents_past_avail = torch.cat(
        [input_dict["agent_polyline_availability"].unsqueeze(1), input_dict["other_agents_polyline_availability"]],
        dim=1,
    )

    static_keys = ["lanes_mid", "crosswalks"]
    if not cfg["model_params"]["disable_lane_boundaries"]:
        static_keys += ["lanes"]
    avail_keys = [f"{k}_availabilities" for k in static_keys]

    max_num_vectors = max([input_dict[key].shape[-2] for key in static_keys])

    static_polys = torch.cat([pad_points(input_dict[key], max_num_vectors) for key in static_keys], dim=1)
    static_polys[..., -1] = 0  # NOTE: this is an hack
    static_avail = torch.cat([pad_avail(input_dict[key], max_num_vectors) for key in avail_keys], dim=1)

    # ===== Future information
    agents_future_positions = torch.cat(
        [input_dict["target_positions"].unsqueeze(1), input_dict["all_other_agents_future_positions"]], dim=1
    )
    agents_future_yaws = torch.cat(
        [input_dict["target_yaws"].unsqueeze(1), input_dict["all_other_agents_future_yaws"]], dim=1
    )
    agents_future_avail = torch.cat(
        [input_dict["target_availabilities"].unsqueeze(1), input_dict["all_other_agents_future_availability"]],
        dim=1,
    )

    # concat XY and yaw to mimic past
    agents_future_polys = torch.cat([agents_future_positions, agents_future_yaws], dim=3)

    # Combine past and future agent information.
    # Future information is ordered [T+1, T+2, ...], past information [T, T-1, T-2, ...].
    # We thus flip past vectors and by concatenating get [..., T-2, T-1, T, T+1, T+2, ...].
    # Now, at each step T the current time window of interest simply is represented by the indices
    # T + agents_past_polys.shape[2] - window_size + 1: T + agents_past_polys.shape[2] + 1.
    # During the training loop, we will fetch this information, as well as static features,
    # which is all represented in the space of T = 0.
    # We then transform this into the space of T and feed this to the model.
    # Eventually, we shift our time window one step into the future.
    # See below for more information about used coordinate spaces.
    agents_polys = torch.cat([torch.flip(agents_past_polys, [2]), agents_future_polys], dim=2)
    agents_avail = torch.cat([torch.flip(agents_past_avail.contiguous(), [2]), agents_future_avail], dim=2)
    window_size = agents_past_polys.shape[2]
    current_timestep = agents_past_polys.shape[2] - 1

    outputs_ts = []  # buffer for predictions in local spaces
    gts_ts = []  # buffer for gts in local spaces
    outputs_t0 = []  # buffer for final prediction in t0 space (for eval only)
    attns = []

    batch_size = agents_polys.shape[0]
    lane_bdry_len = input_dict["lanes"].shape[1]

    type_embedding = vec_embedding(input_dict).transpose(0, 1)

    one = torch.ones_like(input_dict["target_yaws"][:, 0])
    zero = torch.zeros_like(input_dict["target_yaws"][:, 0])

    # ====== Transformation between local spaces
    # NOTE: we use the standard convention A_from_B to indicate that a matrix/yaw/translation
    # converts a point from the B space into the A space
    # e.g. if pB = (1,0) and A_from_B = (-1, 1) then pA = (0, 1)
    # NOTE: we use the following convention for names:
    # t0 -> space at 0, i.e. the space we pull out of the data for which ego is in (0, 0) with no yaw
    # ts -> generic space at step t = s > 0 (predictions at t=s are in this space)
    # tsplus -> space at s+1 (proposal new ts, built from prediction at t=s)
    # A_from_B -> indicate a full 2x3 RT matrix from B to A
    # yaw_A_from_B -> indicate a yaw from B to A
    # tr_A_from_B -> indicate a translation (XY) from B to A
    # NOTE: matrices (and yaw) we need to keep updated while we loop:
    # t0_from_ts -> bring a point from the current space into the data one (e.g. for visualisation)
    # ts_from_t0 -> bring a point from data space into the current one (e.g. to compute loss
    t0_from_ts = torch.eye(3, device=one.device).unsqueeze(0).repeat(batch_size, 1, 1)
    ts_from_t0 = t0_from_ts.clone()
    yaw_t0_from_ts = zero
    yaw_ts_from_t0 = zero

    # === STEP FORWARD ====
    # pick the right point in time
    agents_polys_step = torch.flip(
        agents_polys[:, :, current_timestep - window_size + 1: current_timestep + 1], [2]
    ).clone()
    agents_avail_step = torch.flip(
        agents_avail[:, :, current_timestep - window_size + 1: current_timestep + 1].contiguous(), [2]
    ).clone()
    # PAD
    agents_polys_step = pad_points(agents_polys_step, max_num_vectors)
    agents_avail_step = pad_avail(agents_avail_step, max_num_vectors)

    # crop agents history accordingly
    # NOTE: before padding, agent_polys_step has a number of elements equal to:
    # max_history_num_frames + 1 (where the +1 comes from T0, which is the 0-th element)
    # so in general we want to add +1 to ensure we always keep T0
    # in case of max_history_num_frames=0 we effectively leave only T0
    # ego
    agents_polys_step[:, 0, history_num_frames_ego + 1:] = 0
    agents_avail_step[:, 0, history_num_frames_ego + 1:] = 0
    # agents
    agents_polys_step[:, 1:, history_num_frames_agents + 1:] = 0
    agents_avail_step[:, 1:, history_num_frames_agents + 1:] = 0

    # transform agents and statics into right coordinate system (ts)
    agents_polys_step = transform_points(agents_polys_step, ts_from_t0, agents_avail_step, yaw_ts_from_t0)
    static_avail_step = static_avail.clone()
    static_polys_step = transform_points(static_polys.clone(), ts_from_t0, static_avail_step)
    return agents_polys_step, static_polys_step, agents_avail_step, static_avail_step, type_embedding, lane_bdry_len