# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pyglet
from pyglet.window import key
import numpy as np
import argparse


from oculoenv import Environment, RedCursorEnvironment
from oculoenv import PointToTargetContent, ChangeDetectionContent, OddOneOutContent, VisualSearchContent, \
    MultipleObjectTrackingContent, RandomDotMotionDiscriminationContent


class Contents(object):
    RED_CURSOR = -1
    POINT_TO_TARGET = 1
    CHANGE_DETECTION = 2
    ODD_ONE_OUT = 3
    VISUAL_SEARCH = 4
    MULTIPLE_OBJECT_TRACKING = 5
    RANDOM_DOT_MOTION_DISCRIMINATION = 6


class KeyHandler(object):
    def __init__(self, env, step_debug=False):
        self.env = env
        self.env.window.push_handlers(self.on_key_press, self.on_key_release)

        self.step_debug = step_debug
        
        self.need_run_step = True

        if self.step_debug:
            self.need_run_step = False
        
        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False
        self.one_pressed = False
        self.two_pressed = False
        self.r_pressed = False
        self.esc_pressed = False
        
        pyglet.clock.schedule_interval(self.update, 1.0/60.0)

    def on_key_press(self, symbol, modifiers):
        if symbol == key.LEFT:
            self.left_pressed = True
        elif symbol == key.RIGHT:
            self.right_pressed = True
        elif symbol == key.UP:
            self.up_pressed = True
        elif symbol == key.DOWN:
            self.down_pressed = True
        elif symbol == key._1:
            self.one_pressed = True
        elif symbol == key._2:
            self.two_pressed = True
        elif symbol == key.R:
            self.r_pressed = True
        elif symbol == key.ESCAPE:
            self.esc_pressed = True

        if self.step_debug:
            self.need_run_step = True

    def on_key_release(self, symbol, modifiers):
        if symbol == key.LEFT:
            self.left_pressed = False
        elif symbol == key.RIGHT:
            self.right_pressed = False
        elif symbol == key.UP:
            self.up_pressed = False
        elif symbol == key.DOWN:
            self.down_pressed = False
        elif symbol == key._1:
            self.one_pressed = False
        elif symbol == key._2:
            self.two_pressed = False
        elif symbol == key.R:
            self.r_pressed = False
        elif symbol == key.ESCAPE:
            self.esc_pressed = False

    def update(self, dt):
        dh = 0.0 # Horizontal delta angle
        dv = 0.0 # Vertical delta angle

        delta_angle = 0.02

        if self.left_pressed:
            dh += delta_angle
        if self.right_pressed:
            dh -= delta_angle
        if self.up_pressed:
            dv += delta_angle
        if self.down_pressed:
            dv -= delta_angle
        if self.one_pressed:
            self.env.retina = not self.env.retina
        if self.two_pressed:
            self.env.saliency = not self.env.saliency
        if self.r_pressed:
            self.env.reset()
        if self.esc_pressed:
            self.env.close()
            sys.exit(0)
            return

        if self.need_run_step:
            action = np.array([dh, dv])
            # Step environment
            obs, reward, done, info = self.env.step(action)
            if reward != 0:
                print("reward = {}".format(reward))

            if done:
                print('done!')
                obs = self.env.reset()

            if 'result' in info:
                print("result={}".format(info['result']))

            if 'reaction_step' in info:
                print("reaction step={}".format(info['reaction_step']))

            # Udpate window display
            self.env.render()

            if self.step_debug:
                self.need_run_step = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", help="\n1: Point To Target\n2: Change Detection\n"
                        + "3: Odd One Out\n4: Visual Search\n"
                        + "5: Multiple Object Tracking\n"
                        + "6: Random Dot Motion Descrimination\n"
                        + "-1: Red Cursor",
                        type=int,
                        default=1)
    parser.add_argument("--step_debug",
                        help="Flag to debug execute step by step with one key press",
                        type=bool,
                        default=False)
    parser.add_argument('--skip_red_cursor', action='store_true', help='Flag to skip red cursor.')
    parser.add_argument('--retina', action='store_true', help='Flag to use retina image.')
    parser.add_argument('--saliency', action='store_true', help='Flag to use saliency image.')

    args = parser.parse_args()

    if args.content == Contents.POINT_TO_TARGET:
        content = PointToTargetContent()
    elif args.content == Contents.CHANGE_DETECTION:
        content = ChangeDetectionContent()
    elif args.content == Contents.ODD_ONE_OUT:
        content = OddOneOutContent()
    elif args.content == Contents.VISUAL_SEARCH:
        content = VisualSearchContent()
    elif args.content == Contents.MULTIPLE_OBJECT_TRACKING:
        content = MultipleObjectTrackingContent()
    elif args.content == Contents.RANDOM_DOT_MOTION_DISCRIMINATION:
        content = RandomDotMotionDiscriminationContent()
    elif args.content == Contents.RED_CURSOR:
        content = None
    else:
        print("Unknown argument")
        sys.exit(1)

    env = Environment(content, on_buffer_width=128, skip_red_cursor=args.skip_red_cursor, retina=args.retina, saliency=args.saliency) if content else RedCursorEnvironment(None, on_buffer_width=128, retina=args.retina, saliency=args.saliency)
    env.render()  # env.window is created here

    handler = KeyHandler(env, args.step_debug)

    pyglet.app.run()

    env.close()
