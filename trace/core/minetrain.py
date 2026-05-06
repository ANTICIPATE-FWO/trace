import numpy as np

from gymnasium import Env, Wrapper, core
import mo_gymnasium.envs.minecart.minecart as minecart

from trace.core.maths import point_dist, seg_angle, seg_proj, trail_comb, is_connected


class MinecartTrailWrapper(Wrapper):
    def __init__(self, env:Env[core.ObsType, core.ActType], tolerance:float=0.01):
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
        spin_allowed = self.spin_allowed()

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
        obs[:2], self.env.unwrapped.cart.pos = new_pos, new_pos

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_trail = (np.array([0, 0]), np.array([0, 0]))
        self.on_trail = False
        return obs, info

    def _pick_trail(self):
        pos, angle = self.get_pos(), self.get_angle()
        current_start, current_end = self.current_trail

        next_trail, next_angle, best_angle_dif = None, 0, float('inf')
        for trail, trail_angle in zip(self.trails, self.trail_angles):
            if point_dist(pos, current_start) < point_dist(pos, current_end):
                self.current_trail, angle = flip(self.current_trail, angle)

            if is_connected(self.current_trail, trail): pass
            elif is_connected(self.current_trail, trail[::-1]):
                trail, trail_angle = flip(trail, trail_angle)
            else: continue

            angle_diff = (trail_angle - angle) % 360
            if abs(angle_diff) < best_angle_dif:
                next_trail, next_angle, best_angle_dif = trail, trail_angle, abs(angle_diff)

        assert next_trail is not None, f'Trail not found for {self.current_trail}'
        self.current_trail, self.current_angle = next_trail, next_angle

    def _lock_orientation(self):
        self.env.unwrapped.cart.angle = self.current_angle

    def spin_allowed(self):
        pos = self.get_pos()

        for point in (self.base, *self.current_trail):
            if point_dist(pos, point) <= self.tolerance:
                return True
        return False

    def get_angle(self):
        return self.env.unwrapped.cart.angle

    def get_pos(self):
        return self.env.unwrapped.cart.pos


def flip(trail: tuple, angle: float):
    return trail[::-1], (angle + 180) % 360

