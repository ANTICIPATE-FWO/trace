import numpy as np

from gymnasium import Env, Wrapper
from gymnasium.core import ObsType, ActType
import mo_gymnasium.envs.minecart.minecart as minecart

from trace.core.maths import dist, seg_angle, seg_proj, trail_comb, is_connected


class MinecartTrailWrapper(Wrapper):
    def __init__(self, env:Env[ObsType, ActType], tolerance:float=0.01):
        super().__init__(env)
        self.tolerance = tolerance
        minecart.BASE_RADIUS = 0.05


        self.on_trail = False
        self.current_trail = (np.array([0, 0]), np.array([0, 0]))

        self.base = np.array(minecart.HOME_POS)
        self.mines = [np.array(mine.pos) for mine in env.unwrapped.mines]
        self.trails = trail_comb(self.mines, minecart.HOME_POS)
        self.trail_angles = [seg_angle(*trail) for trail in self.trails]


    def step(self, action):
        spin_allowed = self._spin_allowed()
        pos = self.env.unwrapped.get_state()[:2]
        print(f'\tpos {pos} spin allowed: {spin_allowed}')

        if not self.on_trail:
            if spin_allowed and action == minecart.ACT_ACCEL:
                self._pick_trail()
                self.on_trail = True
                self._lock_orientation()
        else:
            if spin_allowed:
                self.on_trail = False
            else:
                if action in (minecart.ACT_LEFT, minecart.ACT_RIGHT):
                    action = minecart.ACT_NONE
                self._lock_orientation()

        obs, reward, terminated, truncated, info = self.env.step(action)

        new_pos = seg_proj(obs[:2], *self.current_trail)
        self.env.unwrapped.cart.pos = obs[:2] = new_pos
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_trail = (np.array([0, 0]), np.array([0, 0]))
        self.on_trail = False
        return obs, info

    def _pick_trail(self):
        pos, current_angle = self.env.unwrapped.cart.pos, self.env.unwrapped.cart.angle
        best_angle_dif = float('inf')
        next_trail, next_angle = None, 0

        for trail, angle in zip(self.trails, self.trail_angles):
            current_start, current_end = self.current_trail
            if dist(current_start, pos) < dist(current_end, pos): self.current_trail = (current_end, current_start)

            if is_connected(self.current_trail, trail): pass
            elif is_connected(self.current_trail, trail[::-1]): trail = trail[::-1]
            else: continue

            angle_diff = (360 + angle - current_angle) % 180
            if angle_diff < best_angle_dif:
                best_angle_dif = angle_diff
                next_trail, next_angle = trail, angle

        assert next_trail is not None, f'Trail not found for {self.current_trail}'
        self.current_trail, self.current_angle = next_trail, next_angle
        print(f'\tpos {pos} picked trail: {next_trail} with angle: {next_angle}')

    def _lock_orientation(self):
        self.env.unwrapped.cart.angle = self.current_angle

    def _spin_allowed(self):
        pos = self.env.unwrapped.get_state()[:2]
        start, end = self.current_trail

        if dist(pos, self.base) <= self.tolerance:
            return True
        if dist(pos, start) <= self.tolerance:
            return True
        if dist(pos, end) <= self.tolerance:
            return True

        return False

