# -*- coding: utf-8 -*-

import sys
import subprocess
import cv2
import numpy as np

from oculoenv import Environment, PointToTargetContent, ChangeDetectionContent, OddOneOutContent, VisualSearchContent, MultipleObjectTrackingContent, RandomDotMotionDiscriminationContent
from oculoenv.environment import CAMERA_VERTICAL_ANGLE_MAX
from oculoenv.environment import CAMERA_HORIZONTAL_ANGLE_MAX


def save_screen_images(env, random_rate, retry, filename):
    for i in range(retry):
        obs = env.reset()
        # go to the red cursor
        obs, reward, done, info = env.step([-obs['angle'][0], -obs['angle'][1]])
        # step randomly
        if random_rate != 0:
            dh = (np.random.rand() * 2 - 1) * CAMERA_HORIZONTAL_ANGLE_MAX * random_rate
            dv = (np.random.rand() * 2 - 1) * CAMERA_VERTICAL_ANGLE_MAX * random_rate
            obs, reward, done, info = env.step([dh, dv])
        if reward != 0 or done or 'result' in info: # accidentally random step hit target! :D
            sys.stderr.write('retrying({}, {})...'.format(filename, i))
        else:
            break
    assert True == cv2.imwrite(filename, cv2.cvtColor(obs['screen'], cv2.COLOR_RGB2BGR))
    #print(filename)

def make_contents():
    return [PointToTargetContent(), ChangeDetectionContent(), OddOneOutContent(), VisualSearchContent(), MultipleObjectTrackingContent(), RandomDotMotionDiscriminationContent()]

def save_ocoloenv_images(retina, random_rate, retry, datasets, dirname, suffix):
    for content_id, content in enumerate(make_contents(), 1):
        sys.stderr.write('generating images (content_id == {})...'.format(content_id))
        sys.stderr.flush()
        env = Environment(content, on_buffer_width=128, retina=retina)
        for datatype, n in datasets:
            prefix = '{}/{}/{}/'.format(dirname, datatype, content_id)
            subprocess.call(['mkdir', '-p', prefix])
            for i in range(n):
                filename = '{}{}{}'.format(prefix, i, suffix)
                save_screen_images(env, random_rate, retry, filename)
        env.close()
        sys.stderr.write('done\n')


if __name__ == '__main__':
    # number of images should be (batch_size * N)
    #save_ocoloenv_images(False, 0, 100, (('train', 640), ('test', 640)), './data', '.png')
    save_ocoloenv_images(True, 0.1, 100, (('train', 6400), ('test', 640)), './data', '.png')
    # then, load images like this for pytorch datasets
    # trainset = torchvision.datasets.ImageFolder('./data/train/')
    # testset = torchvision.datasets.ImageFolder('./data/test/')
