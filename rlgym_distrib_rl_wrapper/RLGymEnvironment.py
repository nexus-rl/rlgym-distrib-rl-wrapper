import os
from typing import Optional, Tuple, Union
from numpy import ndarray
import random

from rlgym_sim import make
from rlgym_sim.gym import Gym as BaseRLGymEnvironment
from .StateSetters import DynamicGMSetter

from gym import Env
from gym.utils import seeding

from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.action_parsers import DiscreteAction
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.reward_functions import DefaultReward
from rlgym_sim.utils.state_setters import DefaultState
from .ActionParserFactory import build_action_parser_from_config
from .ObsBuilderFactory import build_obs_builder_from_config
from .RewardFunctionFactory import build_reward_function_from_config
from .StateSetterFactory import build_state_setter_from_config
from .TerminalConditionsFactory import build_terminal_conditions_from_config

_make_config_parsers = {
    "action_parser": build_action_parser_from_config,
    "obs_builder": build_obs_builder_from_config,
    "state_setter": build_state_setter_from_config,
    "reward_function": build_reward_function_from_config,
    "terminal_conditions": build_terminal_conditions_from_config,
}

_make_kwarg_names = [
    "reward_function",
    "terminal_conditions",
    "obs_builder",
    "action_parser",
    "state_setter",
    "team_size",
    "spawn_opponents",
    "copy_gamestate_every_step",
    "tick_skip",
    "gravity",
    "boost_consumption",
    "dodge_deadzone",
]

_make_kwarg_key_transformations = {
    "reward_function": "reward_fn"
}

class RLGymEnvironment(Env):
    """The main Rocket League Gym class. It encapsulates the process of managing
    the RLGym environment according to a dynamic, declarative configuration.

    The methods are accessed publicly as "step", "reset", etc...
    """

    def __init__(self, **kwargs):
        self._config = kwargs

        self._env: BaseRLGymEnvironment = None
        self._first_reset = True
        self._state_setter = None
        self._team_size = 1
        self._spawn_opponents = False

        make_kwargs = self._parse_make_kwargs(kwargs)

        self._env = make(**make_kwargs)

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self._init_step_counter()

    def _init_step_counter(self):
        spawn_opponents = self._config.get("spawn_opponents", [False])
        team_size = self._config.get("team_size", [1])

        if type(spawn_opponents) is not list:
            spawn_opponents = [spawn_opponents]
        if type(team_size) is not list:
            team_size = [team_size]

        self._steps_by_team_size = {spawn: {size: 0 for size in team_size} for spawn in spawn_opponents}


    def step(
        self, action: ndarray
    ) -> Union[
        Tuple[ndarray, float, bool, bool, dict], Tuple[ndarray, float, bool, dict]
    ]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns either a tuple `(observation, reward, terminated, truncated, info)`, or a tuple
        (observation, reward, done, info). The latter is deprecated and will be removed in future versions.

        Args:
            action (ActType): an action provided by the agent

        Returns:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.

            (deprecated)
            done (bool): A boolean value for if the episode has ended, in which case further :meth:`step` calls will return undefined results.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """

        # add one step per agent
        steps_to_add = self._team_size * 2 if self._spawn_opponents else self._team_size
        self._steps_by_team_size[self._spawn_opponents][self._team_size] += steps_to_add

        obs, reward, done, info = self._env.step(action)

        # Note: RLGym doesn't return a value for terminated, so we'll just
        # assume it's False
        return obs, reward, done, False, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ndarray, Tuple[ndarray, dict]]:
        """Resets the environment to an initial state and returns the initial observation.

        This method can reset the environment's random number generator(s) if ``seed`` is an integer or
        if the environment has not yet initialized a random number generator.
        If the environment already has a random number generator and :meth:`reset` is called with ``seed=None``,
        the RNG should not be reset. Moreover, :meth:`reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG.
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            return_info (bool): If true, return additional information along with initial observation.
                This info should be analogous to the info returned in :meth:`step`
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)


        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (optional dictionary): This will *only* be returned if ``return_info=True`` is passed.
                It contains auxiliary information complementing ``observation``. This dictionary should be analogous to
                the ``info`` returned by :meth:`step`.
        """

        if options is not None:
            self._config = options

            make_kwargs = self._parse_make_kwargs(self._config)

            self._env.close()
            self._env = make(**make_kwargs)

            self._init_step_counter()

            self.observation_space = self._env.observation_space
            self.action_space = self._env.action_space


        team_size = self._config.get("team_size", 1)
        spawn_opponents = self._config.get("spawn_opponents", False)

        if type(spawn_opponents) is list:
            if self._first_reset or len(self._steps_by_team_size[self._spawn_opponents]) == 0:
                # always prefer max agent count on first reset, so the RLGym
                # version of the env loads the correct number of cars
                self._spawn_opponents = spawn_opponents = True
            else:
                counts = {
                    False: sum(self._steps_by_team_size[False].values()),
                    True: sum(self._steps_by_team_size[True].values())
                }
                # get the spawn_opponents value that has had the fewest total steps
                self._spawn_opponents = spawn_opponents = min(counts, key=counts.get)

        if type(team_size) is list:
            if self._first_reset or len(self._steps_by_team_size[self._spawn_opponents]) == 0:
                # always prefer max agent count on first reset, so the RLGym
                # version of the env loads the correct number of cars
                self._team_size = team_size = max(team_size)
            else:
                # make sure to initialize the dict for this spawn_opponents value if it's empty
                steps_by_team_size = self._steps_by_team_size.get(spawn_opponents, {a: 0 for a in team_size})

                # get the team size that has had the fewest steps
                self._team_size = team_size = min(steps_by_team_size, key=steps_by_team_size.get)

        # no longer our first rodeo
        self._first_reset = False

        blue_team_size = team_size
        orange_team_size = team_size if spawn_opponents else 0
        self._state_setter.set_team_size(blue=blue_team_size, orange=orange_team_size)

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        return self._env.reset(return_info=return_info)

    def _parse_make_kwargs(self, config):
        """Parses the config and returns the kwargs for the make function

        Args:
            config (dict): The config to parse.

        Returns:
            Dict: The kwargs dict for the make function
        """
        kwargs = {
            "reward_fn": DefaultReward(),
            "terminal_conditions": [TimeoutCondition(225), GoalScoredCondition()],
            "obs_builder": DefaultObs(),
            "action_parser": DiscreteAction(),
            "state_setter": DefaultState(),
            "team_size": 1,
            "spawn_opponents": False,
            "copy_gamestate_every_step": False,
            "tick_skip": 8,
            "gravity": 1.0,
            "boost_consumption": 1.0,
            "dodge_deadzone": 0.8
        }

        for key, value in config.items():
            if key in _make_config_parsers:
                make_key = _make_kwarg_key_transformations.get(key, key)
                kwargs[make_key] = _make_config_parsers[key](value)
            elif key in _make_kwarg_names:
                make_key = _make_kwarg_key_transformations.get(key, key)
                kwargs[make_key] = value
            else:
                print(f"WARNING: skipping key {key}")
                pass
                #raise ValueError(f"Unknown config key for environment `RocketLeague-v0`: {key}")

        if type(kwargs["spawn_opponents"]) is list:
            kwargs["spawn_opponents"] = True
            self._spawn_opponents = True

        if type(kwargs["team_size"]) is list:
            kwargs["team_size"] = max(kwargs["team_size"])
            self._team_size = kwargs["team_size"]

        kwargs["state_setter"] = DynamicGMSetter(kwargs["state_setter"])
        self._state_setter = kwargs["state_setter"]

        return kwargs


