from pathlib import Path
import argparse
from tqdm import trange
from flygym import Camera
from cobar_miniproject import levels
from cobar_miniproject.cobar_fly import CobarFly
from flygym import Camera, SingleFlySimulation
from flygym.arena import FlatTerrain
from submission import Controller
import numpy as np
import pickle


def run_simulation(
    submission_dir,
    level,
    seed,
    debug,
    max_steps,
    output_dir="outputs",
    progress=True,
    save_video="", 
    save_obs=False
):
    # sys.path.append(str(submission_dir.parent))
    # module = importlib.import_module(submission_dir.name)
    controller = Controller()
    timestep = 1e-4

    fly = CobarFly(
        debug=debug,
        enable_vision=True,
        render_raw_vision=True,
    )

    if level <= -1:
        level_arena = FlatTerrain()
    elif level <= 1:
        # levels 0 and 1 don't need the timestep
        level_arena = levels[level](fly=fly, seed=seed)
    else:
        # levels 2-4 need the timestep
        level_arena = levels[level](fly=fly, timestep=timestep, seed=seed)
    
    cam_params = {"pos": (0, 0, 80)}

    cam = Camera(
        attachment_point=level_arena.root_element.worldbody,
        camera_name="camera_top_zoomout",
        targeted_fly_names=[fly.name],
        camera_parameters=cam_params,
        play_speed=0.2,
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=level_arena,
    )

    # run cpg simulation
    obs, info = sim.reset()
    obs_hist = []
    info_hist = []

    if progress:
        step_range = trange(max_steps)
    else:
        step_range = range(max_steps)

    t = []
    clean_quit = False
    flip_itr = 0
    for i in step_range:
        # Get observations
        obs, reward, terminated, truncated, info = sim.step(controller.get_actions(obs))
        sim.render()
        if controller.done_level(obs):
            # finish the path integration level
            clean_quit = True
            break
            
        obs_ = obs.copy()
        if not obs_["vision_updated"]:
            if "vision" in obs_:
                del obs_["vision"]
            if "raw_vision" in obs_:
                del obs_["raw_vision"]
        else: 
            t.append(i)
        if "raw_vision" in info:
            del info["raw_vision"]

        obs_["intrinsic_pos"] = controller.get_integrated_position()
        obs_["drive"] = controller.get_last_drive()

        for key, value in controller.get_intermediate_signals().items():
            obs_[key] = value
        obs_hist.append(obs_)
        info_hist.append(info)

        if info["flip"]: 
            flip_itr += 1
        else:
            flip_itr = 0

        if flip_itr > 500:
            print("Flipped, exiting early...")
            break

        if hasattr(controller, "quit") and controller.quit:
            print("Simulation terminated by user.")
            clean_quit = True
            break
        if hasattr(level_arena, "quit") and level_arena.quit:
            print("Target reached. Simulation terminated.")
            clean_quit = True
            break

    # Save video
    file_name_prefix = f"level{level}_seed{seed}"
    save_path = Path(output_dir) / f"{file_name_prefix}.mp4"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cam.save_video(save_path, stabilization_time=0)

   
    if save_video:
        pickle_save_path = Path(output_dir) / f"{file_name_prefix}_video.pkl"
        print(f"Saving video data to {pickle_save_path}")

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

        with open(pickle_save_path, "wb") as f:
            pickle.dump(video_frames, f)

    if save_obs:
        from dm_control.mjcf.physics import SynchronizingArrayWrapper

        save_obs_path = Path(output_dir) / f"{file_name_prefix}_obs.pkl"
        save_obs_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving non-video data to {save_obs_path}")

        for obs in obs_hist: 
            if "raw_vision" in obs:
                del obs["raw_vision"]
            if "vision" in obs:
                del obs["vision"]
            
            for k in obs.keys():
                if type(obs[k]) is SynchronizingArrayWrapper:
                    obs[k] = np.array(obs[k])
        for info in info_hist: 
            if "raw_vision" in info:
                del info["raw_vision"]
        
        data = {
            "obs_hist": obs_hist,
            "info_hist": info_hist
        }
        with open(save_obs_path, "wb") as f:
            pickle.dump(data, f)  

    if clean_quit: 
        exit(0)
    else:
        exit(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fly simulation.")
    parser.add_argument(
        "submission_dir",
        type=Path,
        help="Path to the submission directory containing the controller module.",
    )
    parser.add_argument(
        "--level",
        type=int,
        help="Simulation level to run (e.g., -1 for FlatTerrain, 0-4 for specific levels).",
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for the simulation.",
        default=0,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of steps to run the simulation.",
        default=10000,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for the simulation.",
        default=False,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save the simulation outputs (default: 'outputs').",
        default="outputs",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar during simulation.",
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
        help="save images from observations in a pickle."
    )

    parser.add_argument(
        "--save-obs", 
        action="store_true",
        help="Save the observation and info history"
    )

    args = parser.parse_args()

    run_simulation(
        submission_dir=args.submission_dir,
        level=args.level,
        seed=args.seed,
        debug=args.debug,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        progress=args.progress,
        save_video=args.save_video, 
        save_obs=args.save_obs
    )
