import marimo

__generated_with = "0.13.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numba as nb
    import numpy as np
    return nb, np, plt


@app.cell
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
        image_left = _color_image(id_map, image_left, color_map["front"], (620, 721))
        image_left = _color_image(id_map, image_left, color_map["side"], (500, 620))

        image_right = np.zeros((px_y, px_x, 3))
        image_right = _color_image(id_map, image_right, color_map["neutral"], (0, 721))
        image_right = _color_image(id_map, image_right, color_map["front"], (0, 100))
        image_right = _color_image(id_map, image_right, color_map["side"], (100, 220))

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


if __name__ == "__main__":
    app.run()
