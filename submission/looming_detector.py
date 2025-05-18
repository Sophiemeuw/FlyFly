import numpy as np
import logging
from collections import deque
from enum import IntEnum
import scipy.signal
from pathlib import Path
import imageio
from copy import deepcopy
from flygym.vision import Retina
from PIL import Image
import numba as nb
import flyvis
from flygym.examples.vision import RealTimeVisionNetworkView, RetinaMapper
from torch import Tensor
from flyvis.utils.activity_utils import LayerActivity


class RawVideoHandler:
    def __init__(self, file_name_prefix, retina):
        self.save_path = Path("debug") / f"{file_name_prefix}_raw_vision.mp4"
        self.flat_save_path = Path("debug") / f"{file_name_prefix}_flat_frames.mp4"
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame = None
        self.flat_frame = None
        self.retina = retina

    def __enter__(self):
        fps = 10
        self.writer = imageio.get_writer(self.save_path, fps=fps)  # unsure about fps
        self.flat_writer = imageio.get_writer(self.flat_save_path, fps=fps)

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.close()
        self.flat_writer.close()

    def add_frame(self, frame, frame_type="normal"):
        if frame_type == "normal":
            left_readable = self.retina.hex_pxls_to_human_readable(
                frame[0, :, :], color_8bit=True
            )
            right_readable = self.retina.hex_pxls_to_human_readable(
                frame[1, :, :], color_8bit=True
            )

            frame = np.hstack((left_readable, right_readable))
            if type(self.frame) != np.ndarray:
                self.frame = frame
            else:
                self.frame = np.vstack((self.frame, frame))
        elif frame_type == "flat":
            frame = frame.astype(np.uint8)
            if type(self.flat_frame) != np.ndarray:
                self.flat_frame = frame
            else:
                self.flat_frame = np.vstack((self.flat_frame, frame))
        else:
            raise ValueError(f"Type {frame_type} not recognized")

    def commit_frame(self):
        # This is to make imageio stop complaining:
        # input image is not divisible by macro_block_size=16, resizing from (900, 514) to (912, 528) to ensure video compatibility with most codecs
        # and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1
        # (risking incompatibility).
        if type(self.frame) is np.ndarray:
            if self.frame.shape[0] % 16 != 0 or self.frame.shape[1] % 16 != 0:
                new_height = (self.frame.shape[0] + 15) // 16 * 16
                new_width = (self.frame.shape[1] + 15) // 16 * 16
                self.frame = np.pad(
                    self.frame,
                    (
                        (0, new_height - self.frame.shape[0]),
                        (0, new_width - self.frame.shape[1]),
                        (0, 0),
                    ),
                    mode="constant",
                    constant_values=0,
                )
            self.writer.append_data(self.frame)

        if type(self.flat_frame) is np.ndarray:
            if self.flat_frame.shape[0] % 16 != 0 or self.flat_frame.shape[1] % 16 != 0:
                new_height = (self.flat_frame.shape[0] + 15) // 16 * 16
                new_width = (self.flat_frame.shape[1] + 15) // 16 * 16
                self.flat_frame = np.pad(
                    self.flat_frame,
                    (
                        (0, new_height - self.flat_frame.shape[0]),
                        (0, new_width - self.flat_frame.shape[1]),
                    ),
                    mode="constant",
                    constant_values=0,
                )
            self.flat_writer.append_data(self.flat_frame)

        self.frame = None
        self.flat_frame = None


class LoomDetector:
    def __init__(self, input_size: tuple, debug=False):
        self.vision_results_dir = flyvis.results_dir / "flow/0000/000"
        self.vision_network_view = RealTimeVisionNetworkView(self.vision_results_dir)
        self.vision_network = self.vision_network_view.init_network(chkpt="best_chkpt")
        self.vision_retina_mapper = RetinaMapper()

        self.vision_refresh_rate = 1 / 0.01

        self._vision_network_initialized = False
        self._vision_nn_buf = None
        self._vision_nn_arr_buf = None

        self.debug = debug

    # Thanks to flygym/examples/vision/realistic_vision.py for the infra code :)
    def _init_network(self, vision_obs: np.ndarray):
        vision_obs_grayscale = vision_obs.max(axis=-1)
        visual_input = self.vision_retina_mapper.flygym_to_flyvis(vision_obs_grayscale)
        visual_input = Tensor(visual_input).to(flyvis.device)
        initial_state = self.vision_network.fade_in_state(
            t_fade_in=1.0,
            dt=1 / self.vision_refresh_rate,
            initial_frames=visual_input.unsqueeze(1),
        )
        self.vision_network.setup_step_by_step_simulation(
            dt=1 / self.vision_refresh_rate,
            initial_state=initial_state,
            as_states=False,
            num_samples=2,
        )
        self._initial_state = initial_state
        self._vision_network_initialized = True

    def _get_visual_nn_activities(self, vision_obs):
        vision_obs_grayscale = vision_obs.max(axis=-1)
        visual_input = self.vision_retina_mapper.flygym_to_flyvis(vision_obs_grayscale)
        visual_input = Tensor(visual_input).to(flyvis.device)
        nn_activities_arr = self.vision_network.forward_one_step(visual_input)
        nn_activities_arr = nn_activities_arr.cpu().numpy()
        nn_activities = LayerActivity(
            nn_activities_arr,
            self.vision_network.connectome,
            keepref=True,
            use_central=False,
        )
        return nn_activities, nn_activities_arr

    def __enter__(self):
        if self.debug:
            self.rvh.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.debug:
            self.rvh.__exit__(exc_type, exc_value, traceback)

    def process(self, obs: dict, info: dict):
        if "vision" not in obs:
            return

        if not self._vision_network_initialized:
            self._init_network(obs["vision"])

        if info["vision_updated"] or self._vision_nn_buf is None:
            nn_activities, nn_activities_arr = self._get_visual_nn_activities(
                obs["vision"]
            )
            self._vision_nn_buf = nn_activities
            self._vision_nn_arr_buf = nn_activities_arr


def generate_visual_pattern(
    ld: LoomDetector, pattern_type: str, scan_frames, num_scans
):
    id_map = ld.retina.ommatidia_id_map
    id_map_sz_y, id_map_sz_x = id_map.shape

    output = []
    flip = False
    if pattern_type.endswith("lr") or pattern_type.endswith("rl"):
        increment_per_frame = int(id_map_sz_x / scan_frames)
        increment_dir = "x"
        if pattern_type.endswith("rl"):
            flip = True
    elif pattern_type.endswith("ud") or pattern_type.endswith("du"):
        increment_per_frame = int(id_map_sz_y / scan_frames)
        increment_dir = "y"
        if pattern_type.endswith("du"):
            flip = True
    else:
        raise ValueError(f"Pattern type {pattern_type} not recognized")

    for scan in range(num_scans):
        for frame in range(scan_frames):
            frame_data = np.zeros((id_map_sz_y, id_map_sz_x))
            if increment_dir == "x":
                frame_data[:, 0 : frame * increment_per_frame] = 255
            elif increment_dir == "y":
                frame_data[0 : frame * increment_per_frame, :] = 255

            if flip:
                frame_data = np.flipud(np.fliplr(frame_data))

            fly_view = id_mask_to_fly(ld.retina.ommatidia_id_map, frame_data)
            frame_data = np.stack([frame_data, frame_data], axis=0)
            output.append((frame_data, fly_view))

    return output


@nb.njit(parallel=True)
def id_mask_to_fly(id_map: np.ndarray, id_mask: np.ndarray):
    data = np.zeros(721)
    num_added = np.zeros(721)
    ys, xs = id_mask.shape

    for y in nb.prange(ys - 1):
        for x in nb.prange(xs - 1):
            target = id_map[y, x] - 1
            data[target] += id_mask[y, x]
            num_added[target] += 1

    reshaped_data = np.zeros((2, 721, 2))
    reshaped_data[0, :, 0] = data
    reshaped_data[1, :, 0] = data
    reshaped_data[0, :, 1] = data
    reshaped_data[1, :, 1] = data
    return reshaped_data


if __name__ == "__main__":
    import pickle
    import sys
    import os
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generator_parser = subparsers.add_parser(
        "generator", help="Generate visual patterns"
    )

    generator_parser.add_argument(
        "--pattern",
        type=str,
        default="ud",
        choices=["ud", "du", "lr", "rl"],
        help="Pattern to generate. Options are: ud, du, lr, rl",
    )
    generator_parser.add_argument(
        "--num_scans",
        type=int,
        default=1,
        help="Number of scans to generate. Default is 1.",
    )
    generator_parser.add_argument(
        "--scan_frames",
        type=int,
        default=200,
        help="Number of frames to scan. Default is 200.",
    )

    generator_parser.add_argument(
        "--force-generate",
        action="store_true",
        help="Force generate the pattern even if it already exists.",
    )

    pkl_source = subparsers.add_parser("pkl", help="Process a pkl file")
    pkl_source.add_argument(
        "--file",
        type=str,
        default=".stuff/incoming_ball.pkl",
        help="Path to the pkl file to process. Default is .stuff/incoming_ball.pkl.",
    )
    args = parser.parse_args()
    ld = LoomDetector((2, 721, 2), debug=False)

    if args.command == "generator":
        pattern = args.pattern
        num_scans = args.num_scans
        scan_frames = args.scan_frames

        if not args.force_generate and os.path.exists(f".stuff/{pattern}.pkl"):
            with open(f".stuff/{pattern}.pkl", "rb") as f:
                views = pickle.load(f)
            print("Loaded existing data")
        else:
            print("Generating new data")
            views = generate_visual_pattern(ld, pattern, scan_frames, num_scans)
            views = [view[1] for view in views]

            with open(f".stuff/{pattern}.pkl", "wb") as f:
                pickle.dump(views, f)
    elif args.command == "pkl":
        file_name = args.file
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File {file_name} does not exist.")
        with open(file_name, "rb") as f:
            data = pickle.load(f)
        views = data["vision"]

    print(views[0])

    obs_arr = [{"vision": arr} for arr in views]
    info_arr = [{"vision_updated": True} for _ in range(len(obs_arr))]

    length = len(obs_arr)
    i = 1
    with ld:
        for i in range(length):
            print(f"{i}/{length}")
            i += 1
            ld.process(obs_arr[i], info_arr[i])
