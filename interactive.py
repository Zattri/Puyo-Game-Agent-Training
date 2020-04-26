"""
Interact with Gym environments using the keyboard

An adapter object is defined for each environment to map keyboard commands to actions and extract observations as pixels.
"""

import sys
import ctypes
import argparse
import abc
import time
import random

import numpy as np
import retro
import pyglet
from pyglet import gl
from pyglet.window import key as keycodes

import experience_replay as ExpRep

class Interactive(abc.ABC):
    """
    Base class for making gym environments interactive for human use
    """
    def __init__(self, env, sync=True, tps=60, aspect_ratio=None, verbose="0"):
        self.args_verbose = verbose
        obs = env.reset()
        self._image = self.get_image(obs, env)
        assert len(self._image.shape) == 3 and self._image.shape[2] == 3, 'must be an RGB image'
        image_height, image_width = self._image.shape[:2]

        if aspect_ratio is None:
            aspect_ratio = image_width / image_height

        # guess a screen size that doesn't distort the image too much but also is not tiny or huge
        display = pyglet.canvas.get_display() # Modified code
        screen = display.get_default_screen()
        max_win_width = screen.width * 0.85
        max_win_height = screen.height * 0.85
        win_width = image_width
        win_height = int(win_width / aspect_ratio)

        while win_width > max_win_width or win_height > max_win_height:
            win_width //= 2
            win_height //= 2
        while win_width < max_win_width / 2 and win_height < max_win_height / 2:
            win_width *= 2
            win_height *= 2

        win = pyglet.window.Window(width=win_width, height=win_height)

        self._key_handler = pyglet.window.key.KeyStateHandler()
        win.push_handlers(self._key_handler)
        win.on_close = self._on_close

        gl.glEnable(gl.GL_TEXTURE_2D)
        self._texture_id = gl.GLuint(0)
        gl.glGenTextures(1, ctypes.byref(self._texture_id))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, image_width, image_height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)

        self._env = env
        self._win = win

        # self._render_human = render_human
        self._key_previous_states = {}

        self._steps = 0
        self._episode_steps = 0
        self._episode_returns = 0
        self._prev_episode_returns = 0

        self._tps = tps
        self._sync = sync
        self._current_time = 0
        self._sim_time = 0
        self._max_sim_frames_per_update = 4

        self.obs_memory = []
        self.action_memory = []
        self.reward_threshold = 20
        self.recording_memory_size = 4
        self.obs_record_rate = 4
        self.action_record_rate = 6

    def _update(self, dt):
        exp_rep = ExpRep.ExperienceReplay()
        # cap the number of frames rendered so we don't just spend forever trying to catch up on frames
        # if rendering is slow
        max_dt = self._max_sim_frames_per_update / self._tps
        if dt > max_dt:
            dt = max_dt

        # catch up the simulation to the current time
        self._current_time += dt
        while self._sim_time < self._current_time:
            self._sim_time += 1 / self._tps

            keys_clicked = set()
            keys_pressed = set()
            for key_code, pressed in self._key_handler.items():
                if pressed:
                    keys_pressed.add(key_code)

                if not self._key_previous_states.get(key_code, False) and pressed:
                    keys_clicked.add(key_code)
                self._key_previous_states[key_code] = pressed

            if keycodes.ESCAPE in keys_pressed:
                self._on_close()

            # assume that for async environments, we just want to repeat keys for as long as they are held
            inputs = keys_pressed
            if self._sync:
                inputs = keys_clicked

            keys = []
            for keycode in inputs:
                for name in dir(keycodes):
                    if getattr(keycodes, name) == keycode:
                        keys.append(name)

            act = self.keys_to_act(keys)

            if not self._sync or act is not None:
                obs, rew, done, _info = self._env.step(act)
                self._image = self.get_image(obs, self._env)
                self._episode_returns += rew
                self._steps += 1
                self._episode_steps += 1
                np.set_printoptions(precision=2)

                if self._steps == 6:
                    print("3...")
                    time.sleep(1)
                    print("2..")
                    time.sleep(1)
                    print("1..")
                    time.sleep(1)
                    print("GO!")

                if self._steps % self.action_record_rate == 0:
                    # Appending latest action to memory, stops key / input roll over so only appends every 0.2 secs
                    action_array = list(map(convertBoolToInt, act[:]))
                    action_int = parseActionArrayToInt(action_array)
                    if action_int != 5 and action_int != 2: # If not NONE or DOWN
                        if len(self.action_memory) >= self.recording_memory_size:
                            self.action_memory.pop(0)
                        self.action_memory.append(action_int)

                if self._steps % self.obs_record_rate == 0:
                    obs_img = self._image[4: 206, 18: 110]
                    # Appending latest observations to memory
                    if len(self.obs_memory) >= self.recording_memory_size:
                        self.obs_memory.pop(0)
                    self.obs_memory.append(exp_rep.compressObservation(obs_img))

                if rew >= self.reward_threshold:
                    # Fill up blank memory with NONE
                    while len(self.action_memory) < self.recording_memory_size:
                        self.action_memory.append(5)

                    compressed_array = list(map(exp_rep.compressObservation, self.obs_memory[:]))
                    exp_rep.appendObservation(compressed_array, self.action_memory[:])
                    self.obs_memory.clear()

                if self._steps % 60 == 0:
                    self.last_play_time = self.current_play_time
                    self.current_play_time = _info.get("play_time")

                if self._sync:
                    done_int = int(done)  # shorter than printing True/False
                    mess = 'action={act}\nsteps={self._steps} returns_delta={episode_returns_delta} ep_returns={self._episode_returns} info={_info}'.format(
                        **locals()
                    )
                elif self._steps % self._tps == 0 or done:
                    episode_returns_delta = self._episode_returns - self._prev_episode_returns
                    self._prev_episode_returns = self._episode_returns
                    mess = 'action={act}\nsteps={self._steps} returns_delta={episode_returns_delta} ep_returns={self._episode_returns} info={_info}'.format(
                        **locals()
                    )
                if self.args_verbose == 1:
                    print(mess)

                if done or (self.last_play_time != None and self.current_play_time == self.last_play_time):
                    self._env.reset()
                    self._episode_steps = 0
                    self._episode_returns = 0
                    self._prev_episode_returns = 0
                    exp_rep.saveFile(input("Please input save file name: "))
                    self._on_close()

    def _draw(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        video_buffer = ctypes.cast(self._image.tobytes(), ctypes.POINTER(ctypes.c_short))
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self._image.shape[1], self._image.shape[0], gl.GL_RGB, gl.GL_UNSIGNED_BYTE, video_buffer)

        x = 0
        y = 0
        w = self._win.width
        h = self._win.height

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ('v2f', [x, y, x + w, y, x + w, y + h, x, y + h]),
            ('t2f', [0, 1, 1, 1, 1, 0, 0, 0]),
        )

    def _on_close(self):
        self._env.close()
        sys.exit(0)

    @abc.abstractmethod
    def get_image(self, obs, venv):
        """
        Given an observation and the Env object, return an rgb array to display to the user
        """
        pass

    @abc.abstractmethod
    def keys_to_act(self, keys):
        """
        Given a list of keys that the user has input, produce a gym action to pass to the environment

        For sync environments, keys is a list of keys that have been pressed since the last step
        For async environments, keys is a list of keys currently held down
        """
        pass

    def run(self):
        """
        Run the interactive window until the user quits
        """
        # pyglet.app.run() has issues like https://bitbucket.org/pyglet/pyglet/issues/199/attempting-to-resize-or-close-pyglet
        # and also involves inverting your code to run inside the pyglet framework
        # avoid both by using a while loop
        prev_frame_time = time.time()
        self.current_play_time, self.last_play_time = None, None
        print("LOADING GAME...")
        time.sleep(1)
        while True:
            self._win.switch_to()
            self._win.dispatch_events()
            now = time.time()
            self._update(now - prev_frame_time)
            prev_frame_time = now
            self._draw()
            self._win.flip()


class RetroInteractive(Interactive):
    """
    Interactive setup for retro games
    """
    def __init__(self, game, state, scenario, verbose):
        env = retro.make(game=game, state=state, scenario=scenario)
        self._buttons = env.buttons
        super().__init__(env=env, sync=False, tps=60, aspect_ratio=4/3, verbose=verbose)

    def get_image(self, _obs, env):
        return env.render(mode='rgb_array')

    def keys_to_act(self, keys):
        inputs = {
            None: False,

            'BUTTON': 'Z' in keys,
            'A': 'LEFT' in keys,
            'B': 'RIGHT' in keys,

            'C': 'C' in keys,
            'X': 'V' in keys,
            'Y': 'B' in keys,
            'Z': 'N' in keys,

            'L': 'UP' in keys,
            'R': 'DOWN' in keys,

            'UP': 'W' in keys,
            'DOWN': 'S' in keys,
            'LEFT': 'A' in keys,
            'RIGHT': 'D' in keys,

            'MODE': 'TAB' in keys,
            'SELECT': 'TAB' in keys,
            'RESET': 'ENTER' in keys,
            'START': 'ENTER' in keys,
        }
        return [inputs[b] for b in self._buttons]

def convertBoolToInt(boolValue):
    if boolValue:
        return 1
    else:
        return 0
    
def getRandomState(difficulty=0):
    modifier = "1"

    if difficulty == 0:
        stage = random.randint(1,3)

    elif difficulty == 1:
        stage = 1
        modifier = random.randint(1, 5)

    else:
        stage = difficulty

    return f"p1_s{stage}_0{modifier}"

def parseActionArrayToInt(array):
    if array[0] == 1: #B
        return 0
    elif array[1] == 1: #A
        return 1
    elif array[5] == 1: #DOWN
        return 2
    elif array[6] == 1: #LEFT
        return 3
    elif array[7] == 1: #RIGHT
        return 4
    else: #NONE
        return 5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='Puyo-Genesis')
    parser.add_argument('--state', "-st", default=retro.State.DEFAULT)
    parser.add_argument('--scenario', default=None)
    parser.add_argument('--difficulty', '-d', default=0, help='the difficulty stage of the game state')
    parser.add_argument('--verbose', '-v', type=int, default=0, help='print verbose logging of actions, rewards and game steps')
    args = parser.parse_args()

    if args.state == "random":
        args.state = getRandomState(int(args.difficulty))

    ia = RetroInteractive(game=args.game, state=args.state, scenario=args.scenario, verbose=args.verbose)
    ia.run()


if __name__ == '__main__':
    main()
