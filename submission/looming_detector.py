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


class RawVideoHandler:
    def __init__(self, file_name_prefix, retina):
        self.writers = {}
        self.file_name_prefix = file_name_prefix
        self.retina = retina

    def _make_new_writer(self, name, frame_type):
        fps = 10
        print(f"Making new writer with name {name}")
        path = Path("debug") / f"{self.file_name_prefix}_{name}.mp4"
        os.makedirs(path.parent, exist_ok=True)

        self.writers[name] = {
            "writer": imageio.get_writer(path, fps=fps),
            "pending_frame": None,
            "frame_type": frame_type,
        }

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        for writer in self.writers.values():
            writer["writer"].close()

    def add_frame(self, frame, video_file, frame_type="flat"):
        if video_file not in self.writers.keys():
            self._make_new_writer(video_file, frame_type)

        writer = self.writers[video_file]
        if writer["frame_type"] != frame_type:
            raise RuntimeError("Frame type changing not supported!")

        match frame_type:
            case "fly":
                left_readable = self.retina.hex_pxls_to_human_readable(
                    frame[0, :, :], color_8bit=True
                )
                right_readable = self.retina.hex_pxls_to_human_readable(
                    frame[1, :, :], color_8bit=True
                )

                frame = np.hstack((left_readable, right_readable))
            case "flat":
                frame = frame.astype(np.uint8)
            case _:
                raise ValueError(f"Type {frame_type} not recognized")

        if type(writer["pending_frame"]) != np.ndarray:
            writer["pending_frame"] = frame
        else:
            writer["pending_frame"] = np.vstack((writer["pending_frame"], frame))

    def commit_frame(self):
        for writer in self.writers.values():
            pending_frame = writer["pending_frame"]
            if type(pending_frame) is np.ndarray:
                if pending_frame.shape[0] % 16 != 0 or pending_frame.shape[1] % 16 != 0:
                    new_height = (pending_frame.shape[0] + 15) // 16 * 16
                    new_width = (pending_frame.shape[1] + 15) // 16 * 16
                    match writer["frame_type"]:
                        case "fly":
                            reshape_shape = (
                                (0, new_height - pending_frame.shape[0]),
                                (0, new_width - pending_frame.shape[1]),
                                (0, 0),
                            )
                        case "flat":
                            reshape_shape = (
                                (0, new_height - pending_frame.shape[0]),
                                (0, new_width - pending_frame.shape[1]),
                            )

                    # This is to make imageio stop complaining:
                    # input image is not divisible by macro_block_size=16, resizing from (900, 514) to (912, 528) to ensure video compatibility with most codecs
                    # and players. To prevent resizing, make your input image divisible by the macro_block_size or set the macro_block_size to 1
                    # (risking incompatibility).

                    pending_frame = np.pad(
                        pending_frame,
                        reshape_shape,
                        mode="constant",
                        constant_values=0,
                    )

                writer["writer"].append_data(pending_frame)
                writer["pending_frame"] = None


class EMDDirection(IntEnum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class RectificationType(IntEnum):
    ON = 1
    OFF = 2


class PathwayIndices(IntEnum):
    HORIZ_LR = 0
    HORIZ_RL = 1
    VERT_UD = 2
    VERT_DU = 3


@staticmethod
@nb.njit(parallel=True, cache=True)
def project_to_rect(id_map, image) -> np.ndarray:

    px_y, px_x = id_map.shape
    result = np.zeros((2, px_y, px_x))

    for side in nb.prange(2):
        for y in nb.prange(px_y - 1):
            for x in nb.prange(px_x - 1):
                id = id_map[y, x] - 1
                if id < 0:
                    continue

                result[side, y, x] = image[side, id, 0]

    return result


def project_all_to_rect(id_map, images):
    data = np.stack([image for image in images], axis=0)
    num_images = len(images)
    results = np.zeros((num_images, 2, id_map.shape[0], id_map.shape[1]))

    for img_id in nb.prange(num_images):
        results[img_id, ...] = project_to_rect(id_map, data[img_id, ...])

    return [results[i, ...] for i in range(num_images)]


class LoomDetector:
    def __init__(self, input_size: tuple, debug=False):
        self.DELAY_FRAME = 5
        TS = 0.01
        HPF_TAU = 0.01
        HPF_CORNER = 1 / (2 * np.pi * HPF_TAU)
        BUF_SIZE = 10

        LFP_TAU = 0.005
        LPF_CORNER = 1 / (2 * np.pi * LFP_TAU)

        self.retina = Retina()
        self.frames_recvd = 0
        self.logger = logging.getLogger("ld")

        self.frame_buffer = deque(
            [np.zeros(self.retina.ommatidia_id_map.shape) for _ in range(BUF_SIZE)],
            BUF_SIZE,
        )

        self.emd_signals = {dir: 0 for dir in EMDDirection}
        self.hpf_b, self.hpf_a = scipy.signal.butter(
            1, HPF_CORNER, "high", output="ba", fs=1 / TS
        )

        self.lpf_b, self.lpf_a = scipy.signal.butter(
            1, LPF_CORNER, "low", output="ba", fs=1 / TS
        )

        self.hpf_history = deque([np.zeros(self.retina.ommatidia_id_map.shape)], 10)
        self.on_rect_history = deque(
            [np.zeros(self.retina.ommatidia_id_map.shape)], self.DELAY_FRAME
        )
        self.off_rect_history = deque(
            [np.zeros(self.retina.ommatidia_id_map.shape)], self.DELAY_FRAME
        )

        self.lpf_on_history = deque(
            [
                (
                    np.zeros(self.retina.ommatidia_id_map.shape),
                    np.zeros(self.retina.ommatidia_id_map.shape),
                )
            ],
            2,
        )
        self.lpf_off_history = deque(
            [
                (
                    np.zeros(self.retina.ommatidia_id_map.shape),
                    np.zeros(self.retina.ommatidia_id_map.shape),
                )
            ],
            2,
        )

        self.initialized = False

        self.debug = debug

        self.kernels = {}
        for dir in EMDDirection:
            self.kernels[dir] = self.create_kernel(dir)

        if debug:
            self.rvh = RawVideoHandler("hpf", self.retina)

    def __enter__(self):
        if self.debug:
            self.rvh.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.debug:
            self.rvh.__exit__(exc_type, exc_value, traceback)

    def kernel_to_image(self, kernel: np.ndarray) -> np.ndarray:
        img = np.zeros((2, 721, 2))
        img[0, :, 0] = kernel
        img[1, :, 0] = kernel
        return img

    def create_kernel(
        self, direction: EMDDirection, off_dir_start=0.333, off_dir_end=0.666
    ) -> np.ndarray:
        image = np.zeros(721)

        id_map = self.retina.ommatidia_id_map
        id_map_shape_y, id_map_shape_x = id_map.shape

        off_dir_x_range = (
            int(off_dir_start * id_map_shape_x),
            int(off_dir_end * id_map_shape_x),
        )
        off_dir_y_range = (
            int(off_dir_start * id_map_shape_y),
            int(off_dir_end * id_map_shape_y),
        )

        if direction == EMDDirection.UP:
            ids = np.unique(
                id_map[
                    : id_map_shape_y // 2, off_dir_x_range[0] : off_dir_x_range[1]
                ].ravel()
            )
        elif direction == EMDDirection.LEFT:
            ids = np.unique(
                id_map[
                    off_dir_y_range[0] : off_dir_y_range[1], : id_map_shape_x // 2
                ].ravel()
            )
        elif direction == EMDDirection.RIGHT:
            ids = np.unique(
                id_map[
                    off_dir_y_range[0] : off_dir_y_range[1], id_map_shape_x // 2 :
                ].ravel()
            )
        elif direction == EMDDirection.DOWN:
            ids = np.unique(
                id_map[
                    id_map_shape_y // 2 :, off_dir_x_range[0] : off_dir_x_range[1]
                ].ravel()
            )

        ids = ids - 1
        image[ids] = 1
        return image

    def edge_rectification(
        self, input: np.ndarray, threshold: float, type: RectificationType
    ):
        input = input.copy()
        pre_shape = input.shape
        flattened = input.ravel()

        if type == RectificationType.ON:
            pattern = flattened > threshold
        elif type == RectificationType.OFF:
            pattern = flattened < threshold

        flattened[pattern] = 10 * np.abs(flattened[pattern] - threshold)
        flattened[~pattern] = np.zeros(np.count_nonzero(~pattern))

        return flattened.reshape(pre_shape)

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def make_motion_detector_images(on, off, lpf_on, lpf_off):
        _, px_y, px_x = on.shape

        on_path = np.zeros((2, px_y, px_x, 4))
        off_path = np.zeros((2, px_y, px_x, 4))

        for side in nb.prange(2):
            for y in nb.prange(px_y - 1):
                for x in nb.prange(px_x - 1):
                    if x + 1 < px_x:
                        on_path[side, y, x, PathwayIndices.HORIZ_LR.value] = (
                            lpf_on[side, y, x] * on[side, y, x + 1]
                        )

                        on_path[side, y, x, PathwayIndices.HORIZ_RL.value] = (
                            on[side, y, x] * lpf_on[side, y, x + 1]
                        )

                        off_path[side, y, x, PathwayIndices.HORIZ_LR.value] = (
                            lpf_off[side, y, x] * off[side, y, x + 1]
                        )

                        off_path[side, y, x, PathwayIndices.HORIZ_RL.value] = (
                            off[side, y, x] * lpf_off[side, y, x + 1]
                        )

                    if y + 1 < px_y:
                        on_path[side, y, x, PathwayIndices.VERT_UD.value] = (
                            lpf_on[side, y, x] * on[side, y + 1, x]
                        )
                        on_path[side, y, x, PathwayIndices.VERT_DU.value] = (
                            on[side, y, x] * lpf_on[side, y + 1, x]
                        )

                        off_path[side, y, x, PathwayIndices.VERT_UD.value] = (
                            lpf_off[side, y, x] * off[side, y + 1, x]
                        )
                        off_path[side, y, x, PathwayIndices.VERT_DU.value] = (
                            off[side, y, x] * lpf_off[side, y + 1, x]
                        )

        return on_path, off_path

    def process(self, images: np.ndarray) -> float:
        self.frames_recvd += 1

        images[:, :, 0] = (
            images[:, :, 0] + images[:, :, 1]
        )  # combine the channels since they are useless to us.

        images *= 255  # scale up to 8bit

        images = project_to_rect(self.retina.ommatidia_id_map, images)

        self.frame_buffer.appendleft(images)

        hpf = (
            self.hpf_b[0] * (images)
            + self.hpf_b[1] * self.frame_buffer[1]
            - self.hpf_a[1] * self.hpf_history[0]
        ) / self.hpf_a[0]

        # first order hpf in pipeline
        self.hpf_history.appendleft(hpf)

        on_rect = self.edge_rectification(hpf, 5, RectificationType.ON)
        off_rect = self.edge_rectification(hpf, 5, RectificationType.OFF)

        on_last, lpf_on_last = self.lpf_on_history[0]
        off_last, lpf_off_last = self.lpf_off_history[0]

        lpf_on = (
            self.lpf_b[0] * on_rect
            + self.lpf_b[1] * on_last
            - self.lpf_a[1] * lpf_on_last
        ) / self.lpf_a[0]
        lpf_off = (
            self.lpf_b[0] * off_rect
            + self.lpf_b[1] * off_last
            - self.lpf_a[1] * lpf_off_last
        ) / self.lpf_a[0]

        self.on_rect_history.appendleft(lpf_on)
        self.off_rect_history.appendleft(lpf_off)
        # maybe lpf before all of this to remove jitteryness??

        self.lpf_on_history.appendleft((on_rect, lpf_on))
        self.lpf_off_history.appendleft((off_rect, lpf_off))

        combined = on_rect + off_rect

        if self.frames_recvd < self.DELAY_FRAME:
            print("Accumulating frames...")
            return

        on_path, off_path = self.make_motion_detector_images(
            lpf_on,
            lpf_off,
            self.on_rect_history[self.DELAY_FRAME - 1],
            self.off_rect_history[self.DELAY_FRAME - 1],
        )

        # if self.frames_recvd % 50 == 0:
        #     print("Saving...")
        #     np.save(f".stuff/on_path_{self.frames_recvd}", on_path)

        if self.debug:
            self.rvh.add_frame(np.hstack(list(self.frame_buffer[0])), "first_pass")
            self.rvh.add_frame(np.hstack(list(self.hpf_history[0])), "first_pass")
            self.rvh.add_frame(np.hstack(list(on_rect)), "first_pass")
            self.rvh.add_frame(
                np.hstack(list(self.on_rect_history[self.DELAY_FRAME - 1])),
                "first_pass",
            )
            self.rvh.add_frame(np.hstack(list(off_rect)), "first_pass")
            self.rvh.add_frame(
                np.hstack(list(self.off_rect_history[self.DELAY_FRAME - 1])),
                "first_pass",
            )

            on_path = 255 * (on_path / on_path.max())
            off_path = 255 * (off_path / off_path.max())

            self.rvh.add_frame(
                np.hstack(
                    [
                        off_path[0, :, :, PathwayIndices.HORIZ_LR.value],
                        on_path[0, :, :, PathwayIndices.HORIZ_LR.value],
                        off_path[1, :, :, PathwayIndices.HORIZ_LR.value],
                        on_path[1, :, :, PathwayIndices.HORIZ_LR.value],
                    ]
                ),
                "directional_pathways",
            )
            self.rvh.add_frame(
                np.hstack(
                    [
                        off_path[0, :, :, PathwayIndices.HORIZ_RL.value],
                        on_path[0, :, :, PathwayIndices.HORIZ_RL.value],
                        off_path[1, :, :, PathwayIndices.HORIZ_RL.value],
                        on_path[1, :, :, PathwayIndices.HORIZ_RL.value],
                    ]
                ),
                "directional_pathways",
            )
            self.rvh.add_frame(
                np.hstack(
                    [
                        off_path[0, :, :, PathwayIndices.VERT_UD.value],
                        on_path[0, :, :, PathwayIndices.VERT_UD.value],
                        off_path[1, :, :, PathwayIndices.VERT_UD.value],
                        on_path[1, :, :, PathwayIndices.VERT_UD.value],
                    ]
                ),
                "directional_pathways",
            )
            self.rvh.add_frame(
                np.hstack(
                    [
                        off_path[0, :, :, PathwayIndices.VERT_DU.value],
                        on_path[0, :, :, PathwayIndices.VERT_DU.value],
                        off_path[1, :, :, PathwayIndices.VERT_DU.value],
                        on_path[1, :, :, PathwayIndices.VERT_DU.value],
                    ]
                ),
                "directional_pathways",
            )

            self.rvh.commit_frame()

        return combined
        # else:
        #     pass # maybe log


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
                frame_data[:, 0 : frame * increment_per_frame] = 1
            elif increment_dir == "y":
                frame_data[0 : frame * increment_per_frame, :] = 1

            if flip:
                frame_data = np.flipud(np.fliplr(frame_data))

            fly_view = id_mask_to_fly(ld.retina.ommatidia_id_map, frame_data)
            frame_data = np.stack([frame_data, frame_data], axis=0)
            output.append((frame_data, fly_view))

    return output


@nb.njit(parallel=True, cache=True)
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
            ld = LoomDetector((2, 721, 2), debug=True)
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

    ld = LoomDetector((2, 721, 2), debug=True)

    length = len(views)
    i = 1
    with ld:
        for fly_view in views:
            print(f"{i}/{length}")
            i += 1
            ld.process(fly_view)
