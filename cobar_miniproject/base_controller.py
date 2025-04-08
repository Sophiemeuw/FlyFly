from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    def __init__(self, **kwargs):
        """
        Initialize the CobarController with any necessary parameters.
        """

    @abstractmethod
    def get_actions(self, obs):
        """
        The heart of your controller: goes from observations to joint angles.

        Parameters
        ----------
        obs: dict
            The observations from the environment. Namely
            - The fly's acceleration along x, y, z
            - The end effector positions in egocentric coordinates
            - The contact forces
            - The vision both raw and fly-like
            - The odour intensity in each antenna
            - The fly's heading

        Returns
        -------
        action: dict
            The fly's joint angles under the "joints" key
            The adhesion state under the "adhesion" key
        """

        action = {
            "joints": np.zeros(42),
            "adhesion": np.zeros(6),
        }
        return action

    @abstractmethod
    def done_level(self, obs):
        """
        For level 5 (path integration) - if the fly has returned back home
        after collecting the odour, return `True` here to stop the simulation.
        """
        pass

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the controller to its initial state.
        """
        pass
