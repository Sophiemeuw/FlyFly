import argparse
import cv2
import tqdm
from cobar_miniproject import levels
from cobar_miniproject.keyboard_controller import KeyBoardController
from cobar_miniproject.cobar_fly import CobarFly
from cobar_miniproject.vision import (
    get_fly_vision,
    get_fly_vision_raw,
    render_image_with_vision,
)
from flygym import YawOnlyCamera, SingleFlySimulation
from flygym.arena import FlatTerrain

# OPTIONS
# what to display as the simulation is running
ONLY_CAMERA = 0
WITH_FLY_VISION = 1
WITH_RAW_VISION = 2

VISUALISATION_MODE = WITH_FLY_VISION

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fly simulation.")
    parser.add_argument(
        "--level", type=int, default=1, help="Level to load (default: -1)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the simulation (default: 0)"
    )
    parser.add_argument(
        "--pickle",
        type=str,
        default="",
        help="Path to save the visual sequences as a pickle",
    )
    args = parser.parse_args()

    level = args.level
    seed = args.seed
    timestep = 1e-4

    # you can pass in parameters to enable different senses here
    fly = CobarFly(debug=False, enable_vision=True, render_raw_vision=True)

    if level <= -1:
        level_arena = FlatTerrain()
    elif level <= 1:
        # levels 0 and 1 don't need the timestep
        level_arena = levels[level](fly=fly, seed=seed)
    else:
        # levels 2-4 need the timestep
        level_arena = levels[level](fly=fly, timestep=timestep, seed=seed)

    cam = YawOnlyCamera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_back_track_game",
        targeted_fly_names=[fly.name],
        play_speed=0.2,
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=level_arena,
    )

    controller = KeyBoardController(timestep=timestep, seed=seed)

    # run cpg simulation
    obs, info = sim.reset()
    obs_hist = []
    info_hist = []

    # create window
    cv2.namedWindow("Simulation", cv2.WINDOW_NORMAL)

    with tqdm.tqdm(desc="running simulation") as progress_bar:
        itr = 0
        t = []

        while True:
            # Get observations
            obs, reward, terminated, truncated, info = sim.step(
                controller.get_actions(obs)
            )
            itr += 1
            if controller.done_level(obs):
                # finish the path integration level
                break

            if not obs["vision_updated"]:
                if "vision" in obs:
                    del obs["vision"]
                if "raw_vision" in obs:
                    del obs["raw_vision"]
                if "raw_vision" in info:
                    del info["raw_vision"]
            else:
                t.append(itr)
            obs_hist.append(obs)
            info_hist.append(info)

            rendered_img = sim.render()[0]
            if rendered_img is not None:
                if VISUALISATION_MODE == WITH_FLY_VISION:
                    rendered_img = render_image_with_vision(
                        rendered_img,
                        get_fly_vision(fly),
                        obs["odor_intensity"],
                    )
                elif VISUALISATION_MODE == WITH_RAW_VISION:
                    rendered_img = render_image_with_vision(
                        rendered_img,
                        get_fly_vision_raw(fly),
                        obs["odor_intensity"],
                    )
                rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
                cv2.imshow("Simulation", rendered_img)
                cv2.waitKey(1)

            if controller.quit:
                print("Simulation terminated by user.")
                break
            if hasattr(level_arena, "quit") and level_arena.quit:
                print("Target reached. Simulation terminated.")
                break

            progress_bar.update()

    print("Simulation finished")

    pickle_save_path = args.pickle
    if pickle_save_path:
        import os
        import pickle

        video_frames = {
            "t": t,
            "raw_vision": [
                hist_entry["raw_vision"]
                for hist_entry in obs_hist
                if "raw_vision" in hist_entry
            ],
            "vision": [
                hist_entry["vision"]
                for hist_entry in obs_hist
                if "vision" in hist_entry
            ],
        }
        with open(f"{pickle_save_path}.pkl", "wb") as f:
            pickle.dump(video_frames, f)

    # Save video
    cam.save_video("./outputs/hybrid_controller.mp4", 0)
