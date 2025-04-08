from flygym import Fly
import numpy as np


def get_fly_vision(fly: Fly):
    assert (
        fly._curr_visual_input is not None
    ), "fly vision isn't enabled. Make sure `enable_vision` is set to True."
    return (
        255
        * np.hstack(
            [
                fly.retina.hex_pxls_to_human_readable(
                    fly._curr_visual_input[eye], True
                ).max(axis=2)[::2, ::2]
                for eye in range(2)
            ]
        )
    ).astype(np.uint8)


def get_fly_vision_raw(fly: Fly):
    assert (
        fly._curr_raw_visual_input is not None
    ), "fly vision isn't enabled. Make sure `render_raw_vision` is set to True."

    return np.hstack(tuple(fly._curr_raw_visual_input))


def render_image_with_vision(image: np.ndarray, vision: np.ndarray):
    if vision.ndim < 3:
        vision = vision[..., np.newaxis]
    if vision.shape[2] == 1:
        vision = vision.repeat(3, axis=2)

    if image.shape[1] > vision.shape[1]:
        vision = np.pad(
            vision,
            ((0,), ((image.shape[1] - vision.shape[1]) // 2,), (0,)),
            constant_values=255,
        )
    elif vision.shape[1] > image.shape[1]:
        image = np.pad(
            image,
            ((0,), ((vision.shape[1] - image.shape[1]) // 2,), (0,)),
            constant_values=255,
        )

    return np.vstack((vision, image))
