import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numba as nb
    import numpy as np
    import pickle
    import polars as pl
    return nb, np, pickle, pl, plt


@app.cell(hide_code=True)
def _(nb, np, plt):
    from flygym.examples.vision.vision_network import Retina
    import matplotlib.patches as mpatches

    def _color_image(id_map: np.ndarray, image: np.ndarray, color, range: tuple):
        px_y, px_x = id_map.shape
        for y in nb.prange(px_y):
            for x in nb.prange(px_x):
                id = id_map[y, x] - 1
                if id >= range[0] and id < range[1]:
                    image[y, x, :] = color

        return image

    def make_id_map_plot(): 
        retina = Retina()
        id_map = retina.ommatidia_id_map

        color_map = {
            "neutral": np.array([0.2, 0.2, 0.2]),
            "front": np.array([1, 0, 0]),
            "side": np.array([0, 1, 0])
        }
    
        px_y, px_x = id_map.shape
        image_left = np.zeros((px_y, px_x, 3))
        image_left = _color_image(id_map, image_left, color_map["neutral"], (0, 721))
        image_left = _color_image(id_map, image_left, color_map["front"], (610, 721))
        image_left = _color_image(id_map, image_left, color_map["side"], (490, 610))

        image_right = np.zeros((px_y, px_x, 3))
        image_right = _color_image(id_map, image_right, color_map["neutral"], (0, 721))
        image_right = _color_image(id_map, image_right, color_map["front"], (0, 111))
        image_right = _color_image(id_map, image_right, color_map["side"], (111, 231))

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image_left)
        ax[0].set_title("Left Eye Mapping")
        ax[0].axis('off')
        ax[1].imshow(image_right)
        ax[1].set_title("Right Eye Mapping")
        ax[1].axis('off')
        # create one patch for each of your three categories
        patches = [
            mpatches.Patch(color=color_map["neutral"], label="Neutral"),
            mpatches.Patch(color=color_map["front"],   label="Front"),
            mpatches.Patch(color=color_map["side"],    label="Side"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=3)
        fig.set_tight_layout(True)
        fig.savefig("plots/image_regions.png")
        plt.show()
    
    make_id_map_plot()
    return


@app.cell
def _(pickle, pl):
    def load_data(pickle_path):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        obs_hist = pl.from_records(data["obs_hist"])
        obs_hist = obs_hist.with_row_index("i").with_columns(pl.col("i").mul(1e-4).alias("t"))
        info_hist = pl.from_records(data["info_hist"])


        return obs_hist, info_hist

    return (load_data,)


@app.cell
def _(load_data, pl, plt):
    def make_time_axis_labels(ax, ylabel):
        ax.set_xlabel("t [s]")
        ax.set_ylabel(ylabel)

    def make_xy_plot(ax, obs_hist: pl.DataFrame, info_hist: pl.DataFrame) -> plt.Axes:
        pos = obs_hist.select("intrinsic_pos").to_series().to_list()
        x = [p[0] for p in pos]
        y = [p[1] for p in pos]
        ax.plot(x, y)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        return ax


    def make_odor_time_plot(ax, obs_hist, info_hist) -> plt.Axes:
        t = obs_hist.select("t")
        I_left = obs_hist["I_left"]
        I_right = obs_hist["I_right"]

        ax.plot(t, I_left, label="left intensity")
        ax.plot(t, I_right, label="right intensity")

        make_time_axis_labels(ax, "Odor Intensity")
        return ax

    def make_drive_time_plot(ax, obs_hist, info_hist, drive_name: str) -> plt.Axes:
        t = obs_hist.select("t")
        drive = obs_hist.select(drive_name).to_series().to_list()
        drive_l = [d[0] for d in drive]
        drive_r = [d[1] for d in drive]

        ax.plot(t, drive_l, label="left")
        ax.plot(t, drive_r, label="right")
        make_time_axis_labels(ax, drive_name.replace("_", " ").title())

    def make_single_value_time_plot(ax, obs_hist, info_hist, value_key, ylabel="") -> plt.Axes:
        t = obs_hist.select("t")
        y = obs_hist.select(value_key)

        ax.plot(t, y)
        if not ylabel:
            ylabel = value_key.replace("_", " ").title()
        make_time_axis_labels(ax, ylabel)

    def make_plot_level0_overall():
        obs_hist, info_hist = load_data("data/level0_seed38_obs.pkl")
        fig, ax = plt.subplots()
        ax.set_title("Odor Intensity")
        make_odor_time_plot(ax, obs_hist, info_hist)

        ax.legend()
        ax.grid()

        return fig, ax

    def make_plot_level0_zoomed():
        obs_hist, info_hist = load_data("data/level0_seed38_obs.pkl")

        obs_hist = obs_hist.tail(500)

        fig, ax = plt.subplots(4, 1)
        make_odor_time_plot(ax[0], obs_hist, info_hist)
        make_single_value_time_plot(ax[1], obs_hist, info_hist, "s", ylabel="Assymmetry")
        make_drive_time_plot(ax[2], obs_hist, info_hist, "odor_taxis_drive")
        make_drive_time_plot(ax[3], obs_hist, info_hist, "drive")


        for axis in ax:
            axis.legend()
            axis.grid()

        return fig, ax
    
    



    return make_plot_level0_overall, make_plot_level0_zoomed


@app.cell
def _(make_plot_level0_overall):
    make_plot_level0_overall()
    return


@app.cell
def _(make_plot_level0_zoomed):
    make_plot_level0_zoomed()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
