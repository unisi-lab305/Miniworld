import math

from gymnasium import spaces, utils

from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv


class GradientMaze(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Maze like environment.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +(obs-brightness - 0.2 * (step_count / max_episode_steps))

    ## Arguments

    ```python
    Hallway(length=12)
    ```

    `length`: length of the entire space

    """

    def __init__(self, length=12, **kwargs):
        assert length >= 2
        self.length = length

        MiniWorldEnv.__init__(self, max_episode_steps=100, **kwargs)
        utils.EzPickle.__init__(self, length, **kwargs)

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):

        room0 = self.add_rect_room(
            min_x=4,
            max_x=10,
            min_z=4,
            max_z=10,
            wall_tex="brick_wall",
            floor_tex="brick_wall",
            no_ceiling=False
            ,
        )

        wall1 = self.add_rect_room(
            min_x=5,
            max_x=5,
            min_z=5,
            max_z=9,
            wall_tex="gradient4",
            no_ceiling=True,
        )

        wall2 = self.add_rect_room(
            min_x=5,
            max_x=9,
            min_z=5,
            max_z=5,
            wall_tex="gradient3",
            no_ceiling=True,
        )

        wall3 = self.add_rect_room(
            min_x=9,
            max_x=9,
            min_z=5,
            max_z=9,
            wall_tex="gradient2",
            no_ceiling=True,
        )

        wall4 = self.add_rect_room(
            min_x=5,
            max_x=9,
            min_z=9,
            max_z=9,
            wall_tex="gradient1",
            no_ceiling=True,
        )

        self.place_agent(room=room0, min_x=7, max_x=7, min_z=7, max_z=7)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        print()
        # print(self.agent.pos)
        # print(info)
        # print(obs)
        # print(type(obs))
        # print(obs.shape)
        # print(obs.min())
        # print(obs.max())
        # print(obs.mean())

        # step decay
        reward += self._reward()
        # brightness reward
        brightness = obs.mean()
        # norm_brightness = brightness/255.
        # reward += norm_brightness
        reward += brightness

        return obs, reward, termination, truncation, info
