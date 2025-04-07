import cv2
import numpy as np
from flygym.arena import FlatTerrain
from flygym.examples.vision.arena import ObstacleOdorArena


def _get_random_target_position(
    distance_range: tuple[float, float],
    angle_range: tuple[float, float],
    rng: np.random.Generator,
):
    """Generate a random target position.

    Parameters
    ----------
    distance_range : tuple[float, float]
        Distance range from the origin.
    angle_range : tuple[float, float]
        Angle rabge in radians.
    rng : np.random.Generator
        The random number generator.
    Returns
    -------
    np.ndarray
        The target position in the form of [x, y].
    """
    p = rng.uniform(*distance_range) * np.exp(1j * rng.uniform(*angle_range))
    return np.array([p.real, p.imag], float)


def _circ(
    img: np.ndarray,
    xy: tuple[float, float],
    r: float,
    value: bool,
    xmin: float,
    ymin: float,
    res: float,
    outer=False,
):
    """Draw a circle on a 2D image.

    Parameters
    ----------
    img : np.ndarray
        The image to draw on.
    xy : tuple[float, float]
        The center of the circle.
    r : float
        The radius of the circle.
    value : bool
        The value to set the pixels to.
    xmin : float
        The minimum x value of the grid.
    ymin : float
        The minimum y value of the grid.
    res : float
        The resolution of the grid.
    outer : bool, optional
        If True, draw the outer circle. Otherwise, draw a filled circle.

    Returns
    -------
    None
    """
    center = ((np.asarray(xy) - (xmin, ymin)) / res).astype(int)
    radius = int(r / res) + 1 if outer else int(r / res)
    color = bool(value)
    thickness = 1 if outer else -1
    cv2.circle(img, center, radius, color, thickness)


class ScatteredPillarsArena(ObstacleOdorArena):
    """
    An arena with scattered pillars and a target marker.

    This class generates an arena with randomly placed pillars and a target marker.
    The target marker is placed at a random position within a specified distance
    and angle range. Pillars are placed randomly while maintaining a minimum
    separation from the target, the fly, and other pillars.

    Parameters
    ----------
    target_distance_range : tuple[float, float], optional
        Range of distances from the origin for the target marker. Default is (29, 31).
    target_angle_range : tuple[float, float], optional
        Range of angles (in radians) for the target marker. Default is (-π, π).
    target_clearance_radius : float, optional
        Minimum clearance radius around the target marker. Default is 4.
    target_marker_size : float, optional
        Size of the target marker. Default is 0.3.
    target_marker_color : tuple[float, float, float, float], optional
        RGBA color of the target marker. Default is (1, 0.5, 14/255, 1).
    pillar_height : float, optional
        Height of the pillars. Default is 3.
    pillar_radius : float, optional
        Radius of the pillars. Default is 0.3.
    pillars_minimum_separation : float, optional
        Minimum separation between pillars. Default is 6.
    fly_clearance_radius : float, optional
        Minimum clearance radius around the fly. Default is 4.
    seed : int or None, optional
        Seed for the random number generator. Default is None.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Attributes
    ----------
    odor_source : np.ndarray
        Position of the odor source.
    marker_colors : np.ndarray
        Colors of the markers.
    peak_odor_intensity : np.ndarray
        Peak intensity of the odor.
    obstacle_positions : np.ndarray
        Positions of the pillars.
    obstacle_radius : float
        Radius of the pillars.
    obstacle_height : float
        Height of the pillars.
    terrain : FlatTerrain
        Terrain of the arena.
    marker_size : float
        Size of the target marker.
    """

    def __init__(
        self,
        target_distance_range=(29, 31),
        target_angle_range=(-np.pi, np.pi),
        target_clearance_radius=4,
        target_marker_size=0.3,
        target_marker_color=(1, 0.5, 14 / 255, 1),
        pillar_height=3,
        pillar_radius=0.3,
        pillars_minimum_separation=6,
        fly_clearance_radius=4,
        seed=None,
        **kwargs,
    ):
        rng = np.random.default_rng(seed)

        target_position = _get_random_target_position(
            distance_range=target_distance_range,
            angle_range=target_angle_range,
            rng=rng,
        )

        pillar_positions = self._get_pillar_positions(
            target_position=target_position,
            target_clearance_radius=target_clearance_radius,
            pillar_radius=pillar_radius,
            pillars_minimum_separation=pillars_minimum_separation,
            fly_clearance_radius=fly_clearance_radius,
            rng=rng,
        )

        super().__init__(
            odor_source=np.array([[*target_position, 1]]),
            marker_colors=np.array([target_marker_color]),
            peak_odor_intensity=np.array([[1, 0]]),
            obstacle_positions=pillar_positions,
            obstacle_radius=pillar_radius,
            obstacle_height=pillar_height,
            terrain=FlatTerrain(ground_alpha=0),
            marker_size=target_marker_size,
            **kwargs,
        )

    @staticmethod
    def _get_pillar_positions(
        target_position: tuple[float, float],
        target_clearance_radius: float,
        pillar_radius: float,
        pillars_minimum_separation: float,
        fly_clearance_radius: float,
        rng: np.random.Generator,
        res: float = 0.05,
    ):
        """Generate random pillar positions.

        Parameters
        ----------
        target_position : tuple[float, float]
            The target x and y coordinates.
        target_clearance_radius : float
            The radius of the area around the target that should be clear of pillars.
        pillar_radius : float
            The radius of the pillars.
        pillars_minimum_separation : float
            Minimum separation between pillars.
        fly_clearance_radius : float
            The radius of the area around the fly that should be clear of pillars.
        rng : np.random.Generator
            The random number generator.
        res : float, optional
            The resolution of the grid. Default is 0.05.

        Returns
        -------
        np.ndarray
            The positions of the pillars in the form of [[x1, y1], [x2, y2], ...].
        """
        pillar_clearance_radius = pillar_radius * 2 + pillars_minimum_separation
        target_clearance_radius = target_clearance_radius + pillar_radius
        fly_clearance_radius = fly_clearance_radius + pillar_radius

        target_position = np.asarray(target_position)
        distance = np.linalg.norm(target_position)
        xmin = ymin = -distance
        xmax = ymax = distance
        n_cols = int((xmax - xmin) / res)
        n_rows = int((ymax - ymin) / res)
        im1 = np.zeros((n_rows, n_cols), dtype=np.uint8)
        im2 = np.zeros((n_rows, n_cols), dtype=np.uint8)

        _circ(im1, (0, 0), distance, 1, xmin, ymin, res)
        _circ(im1, (0, 0), fly_clearance_radius, 0, xmin, ymin, res)
        _circ(im1, target_position, target_clearance_radius, 0, xmin, ymin, res)

        pillars_xy = [target_position / 2]
        _circ(im1, pillars_xy[0], pillar_clearance_radius, 0, xmin, ymin, res)
        _circ(
            im2, pillars_xy[0], pillar_clearance_radius, 1, xmin, ymin, res, outer=True
        )

        while True:
            argwhere = np.argwhere(im1 & im2)
            try:
                p = argwhere[rng.choice(len(argwhere)), ::-1] * res + (xmin, ymin)
            except ValueError:
                break
            pillars_xy.append(p)
            _circ(im1, p, pillar_clearance_radius, 0, xmin, ymin, res)
            _circ(im2, p, pillar_clearance_radius, 1, xmin, ymin, res, outer=True)

        return np.array(pillars_xy)
