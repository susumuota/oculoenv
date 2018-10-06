# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import cv2

import pyglet
from pyglet.gl import *
from ctypes import POINTER

from .graphics import MultiSampleFrameBuffer, FrameBuffer
from .objmesh import ObjMesh
from .utils import clamp, rad2deg, deg2rad
from .geom import Matrix4

from .contents.point_to_target_content import PointToTargetContent
from .contents.change_detection_content import ChangeDetectionContent

BG_COLOR = np.array([0.45, 0.82, 1.0, 1.0])
WHITE_COLOR = np.array([1.0, 1.0, 1.0])

# Camera vertical field of view angle (degree)
CAMERA_FOV_Y = 50

# Initial vertical angle of camera (radian)
CAMERA_INITIAL_ANGLE_V = deg2rad(10.0)

# Max vertical angle of camera (radian)
CAMERA_VERTICAL_ANGLE_MAX = deg2rad(45.0)

# Max horizontal angle of camera (radian)
CAMERA_HORIZONTAL_ANGLE_MAX = deg2rad(45.0)
    

PLANE_DISTANCE = 3.0  # Distance to content plane

# copied from https://github.com/wbap/oculomotor/blob/master/application/functions/lip.py
GAUSSIAN_KERNEL_SIZE = (5,5)


class PlaneObject(object):
    def __init__(self):
        # TODO: グリッドをもう少し細かく分けて、TextureのSkewが軽減されるかどうか調べる
        verts = [
            -1,  1, 0,
            -1, -1, 0,
             1, -1, 0,
             1,  1, 0,
        ]
        texcs = [
            0, 1,
            0, 0,
            1, 0,
            1, 1,
        ]

        self.panel_vlist = pyglet.graphics.vertex_list(4, ('v3f', verts),
                                                       ('t2f', texcs))

    def render(self, content):
        content.bind()
        self.panel_vlist.draw(GL_QUADS)


class SceneObject(object):
    """ A class for drawing .obj mesh object with drawing property (pos, scale etc).

    Arguments:
      obj_name: String, file name of wavefront .obj file.
      pos:      Float array, position of the object
      scale:    Float, scale of the object.
      rot:      Float (radian), rotation angle around Y axis.
    """

    def __init__(self, obj_name, pos=[0, 0, 0], scale=1.0, rot=0.0):

        self.mesh = ObjMesh.get(obj_name)
        self.pos = pos
        self.scale = scale
        self.rot = rad2deg(rot)

    def render(self):
        glPushMatrix()
        glTranslatef(*self.pos)
        glScalef(self.scale, self.scale, self.scale)
        glRotatef(self.rot, 0, 1, 0)
        self.mesh.render()
        glPopMatrix()


class Camera(object):
    """ 3D camera class. """

    def __init__(self):
        self.reset()

    def _update_mat(self):
        m0 = Matrix4()
        m1 = Matrix4()
        m0.set_rot_x(self.cur_angle_v)
        m1.set_rot_y(self.cur_angle_h)
        self.m = m1.mul(m0)

    def reset(self):
        self.cur_angle_h = 0  # Horizontal
        self.cur_angle_v = CAMERA_INITIAL_ANGLE_V  # Vertical

        self._update_mat()

    def get_forward_vec(self):
        """ Get forward vector
    
        Returns:
          numpy ndarray (float): size=3
        """
        v = self.m.get_axis(2)  # Get z-axis
        return -1.0 * v  # forward direction is minus z-axis of the matrix

    def change_angle(self, d_angle_h, d_angle_v):
        self.cur_angle_h += d_angle_h  # left-right angle
        self.cur_angle_v += d_angle_v  # top-down angle
        
        self.cur_angle_h = clamp(self.cur_angle_h,
                                 -CAMERA_HORIZONTAL_ANGLE_MAX,
                                 CAMERA_HORIZONTAL_ANGLE_MAX)        
        self.cur_angle_v = clamp(self.cur_angle_v, -CAMERA_VERTICAL_ANGLE_MAX,
                                 CAMERA_VERTICAL_ANGLE_MAX)

        self._update_mat()

    def get_inv_mat(self):
        """ Get invererted camera matrix

        Returns:
          numpy ndarray: inverted camera matrix
        """
        m_inv = self.m.invert()
        return m_inv


class Environment(object):
    """ Task Environmenet class. """

    def __init__(self, content, off_buffer_width=128, on_buffer_width=640, skip_red_cursor=False, retina=False, saliency=False, diff=False):
        """ Oculomotor task environment class.

        Arguments:
          content: (Content) object
          off_buffer_width: (int) pixel width and height size of offscreen render buffer.
          on_buffer_width: (int) pixel width and height size of display window.
        """

        # skip
        self.skip_red_cursor = skip_red_cursor

        # retina images
        self.retina = retina
        if self.retina:
            self.on_blur_rates, self.on_inv_blur_rates = self._create_rate_datas(on_buffer_width) # 640
            self.on_gray_rates, self.on_inv_gray_rates = self._create_rate_datas(on_buffer_width, gain=0.5) # 640
            self.off_blur_rates, self.off_inv_blur_rates = self._create_rate_datas(off_buffer_width) # 128
            self.off_gray_rates, self.off_inv_gray_rates = self._create_rate_datas(off_buffer_width, gain=0.5) # 128
        self.saliency = saliency
        self.diff = diff
        if self.diff:
            self.prev_frame_buffer_off = None
            self.prev_frame_buffer_on = None

        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(
            width=1, height=1, visible=False)

        self.frame_buffer_off = FrameBuffer(off_buffer_width, off_buffer_width)
        self.frame_buffer_on = FrameBuffer(on_buffer_width, on_buffer_width)

        self.camera = Camera()

        # Window for displaying the environment to humans
        self.window = None

        self.content = content
        self.plane = PlaneObject()

        # Add scene objects
        self._init_scene()

        self.reset()

    def _init_scene(self):
        # Create the objects array
        self.objects = []

        obj = SceneObject("frame0", pos=[0.0, 0.0, -PLANE_DISTANCE], scale=2.0)
        self.objects.append(obj)

    def _get_observation(self):
        # Get rendered image
        image = self._render_offscreen()

        # Change upside-down
        image = np.flip(image, 0)
        
        # Current absolute camera angle
        angle = (self.camera.cur_angle_h, self.camera.cur_angle_v)
        
        obs = {
            "screen":image,
            "angle":angle
        }

        return obs

    def reset(self):
        """ Reset environment.
        
        Returns:
          Dictionary
            "screen" numpy ndarray (Rendered Image)
            "angle" (horizontal angle, vertical angle) Absoulte angles of the camera
        """
        
        self.content.reset()
        self.camera.reset()
        obs = self._get_observation()

        if self.skip_red_cursor and self._is_start_phase():
            obs, reward, done, info = self._step_to_center(obs) # skip!!!

        return obs

    def _calc_local_focus_pos(self, camera_forward_v):
        """ Calculate local coordinate of view focus point on the content panel. """

        tz = -camera_forward_v[2]
        tx = camera_forward_v[0]
        ty = camera_forward_v[1]

        local_x = tx * (PLANE_DISTANCE / tz)
        local_y = ty * (PLANE_DISTANCE / tz)
        return [local_x, local_y]

    def step(self, action):
        """ Execute one environment step. 
        
        Arguments:
          action: Float array, (horizonal delta angle, vertical delta angle) in radian.
        
        Returns:
          obs, reward, done, info
            obs: Dictionary
              "screen" numpy ndarray (Rendered Image)
              "angle" (horizontal angle, vertical angle) Absoulte angles of the camera
            reward: (Float) Reward 
            done: (Bool) Terminate flag
            info: (Dictionary) Response time and trial result information.
        """
        
        d_angle_h = action[0]  # left-right angle
        d_angle_v = action[1]  # top-down angle
        
        self.camera.change_angle(d_angle_h, d_angle_v)

        camera_forward_v = self.camera.get_forward_vec()
        local_focus_pos = self._calc_local_focus_pos(camera_forward_v)
        reward, done, info = self.content.step(local_focus_pos)

        obs = self._get_observation()

        if self.skip_red_cursor and self._is_start_phase():
            obs, _, _, _ = self._step_to_center(obs) # skip!!! don't update reward, done and info!

        return obs, reward, done, info

    def _step_to_center(self, obs):
        '''this is a cheat code.'''
        return self.step([-obs['angle'][0], -obs['angle'][1]])

    def _is_start_phase(self):
        if type(self.content) is ChangeDetectionContent:
            return self.content.current_phase == self.content.start_phase
        else:
            return self.content.phase == 0 # PHASE_START == 0

    def close(self):
        pass

    def _render_offscreen(self):
        img = self._render_sub(self.frame_buffer_off)
        if self.retina:
            img = self._create_retina_image(img, self.off_blur_rates, self.off_inv_blur_rates, self.off_gray_rates, self.off_inv_gray_rates)
        if self.saliency:
            img = self._get_saliency_map(img)
            img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
            img = np.stack([img for _ in range(3)], axis=2)
        if self.diff:
            org_image = img
            img = self._get_diff_map(self.prev_frame_buffer_off, img)
            self.prev_frame_buffer_off = org_image
        return img

    def render(self, mode='human', close=False):
        if close:
            if self.window:
                self.window.close()
            return

        img = self._render_sub(self.frame_buffer_on)

        if self.retina:
            img = self._create_retina_image(img, self.on_blur_rates, self.on_inv_blur_rates, self.on_gray_rates, self.on_inv_gray_rates)

        if self.saliency:
            img = self._get_saliency_map(img)
            img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
            img = np.stack([img for _ in range(3)], axis=2)

        if self.diff:
            org_image = img
            img = self._get_diff_map(self.prev_frame_buffer_on, img)
            self.prev_frame_buffer_on = org_image

        if mode == 'rgb_array':
            return img

        if self.window is None:
            config = pyglet.gl.Config(double_buffer=False)
            self.window = pyglet.window.Window(
                width=self.frame_buffer_on.width,
                height=self.frame_buffer_on.height,
                resizable=False,
                config=config)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # Bind the default frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Setup orghogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, self.frame_buffer_on.width, 0, self.frame_buffer_on.height,
                0, 10)

        # Draw the image to the rendering window
        width = img.shape[1]
        height = img.shape[0]
        img_data = pyglet.image.ImageData(
            width,
            height,
            'RGB',
            img.ctypes.data_as(POINTER(GLubyte)),
            pitch=width * 3,
        )
        img_data.blit(
            0,
            0,
            0,
            width=self.frame_buffer_on.width,
            height=self.frame_buffer_on.height)

        # Force execution of queued commands
        glFlush()

    def _render_sub(self, frame_buffer):
        self.shadow_window.switch_to()

        frame_buffer.bind()

        # Clear the color and depth buffers
        glClearColor(*BG_COLOR)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            CAMERA_FOV_Y,
            frame_buffer.width / float(frame_buffer.height),
            0.04,  # near plane
            100.0  # far plane
        )

        # Apply camera angle
        glMatrixMode(GL_MODELVIEW)
        m = self.camera.get_inv_mat()
        glLoadMatrixf(m.get_raw_gl().ctypes.data_as(POINTER(GLfloat)))

        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        # For each object
        glColor3f(*WHITE_COLOR)
        for obj in self.objects:
            obj.render()

        # Draw content panel
        glEnable(GL_TEXTURE_2D)
        glPushMatrix()
        glTranslatef(0.0, 0.0, -PLANE_DISTANCE)
        glScalef(1.0, 1.0, 1.0)
        self.plane.render(self.content)
        glPopMatrix()

        return frame_buffer.read()

    # copied from https://github.com/wbap/oculomotor/blob/master/application/functions/retina.py
    def _gauss(self, x, sigma):
        sigma_sq = sigma * sigma
        return 1.0 / np.sqrt(2.0 * np.pi * sigma_sq) * np.exp(-x*x/(2 * sigma_sq))

    def _create_rate_datas(self, width, sigma=0.32, clipping_gain=1.2, gain=1.0):
        """ Create mixing rate.
        Arguments:
            width: (int) width of the target image.
            sigma: (float) standard deviation of the gaussian.
            clipping_gain: (float) To make the top of the curve flat, apply gain > 1.0
            gain: (float) Final gain for the mixing rate.
                          e.g.) if gain=0.8, mixing rates => 0.2~1.0
        Returns:
            Float ndarray (128, 128, 1): Mixing rates and inverted mixing rates.
        """
        rates = [0.0] * (width * width)
        hw = width // 2
        for i in range(width):
            x = (i - hw) / float(hw)
            for j in range(width):
                y = (j - hw) / float(hw)
                r = np.sqrt(x*x + y*y)
                rates[j*width + i] = self._gauss(r, sigma=sigma)
        rates = np.array(rates)
        # Normalize
        rates = rates / np.max(rates)

        # Make top flat by multipying and clipping
        rates = rates * clipping_gain
        rates = np.clip(rates, 0.0, 1.0)

        # Apply final gain
        if gain != 1.0:
            rates = rates * gain + (1-gain)
        rates = rates.reshape([width, width, 1])
        inv_rates = 1.0 - rates
        return rates, inv_rates

    def _create_blur_image(self, image):
        h = image.shape[0]
        w = image.shape[1]

        # Resizeing to 1/2 size
        resized_image0 = cv2.resize(image,
                                  dsize=(h//2, w//2),
                                  interpolation=cv2.INTER_LINEAR)
        # Resizeing to 1/4 size
        resized_image1 = cv2.resize(resized_image0,
                                  dsize=(h//4, w//4),
                                  interpolation=cv2.INTER_LINEAR)
        # Resizeing to 1/8 size
        resized_image2 = cv2.resize(resized_image1,
                                  dsize=(h//8, w//8),
                                  interpolation=cv2.INTER_LINEAR)

        # Resizing to original size
        blur_image = cv2.resize(resized_image2,
                                dsize=(h, w),
                                interpolation=cv2.INTER_LINEAR)

        # Conver to Grayscale
        gray_blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
        gray_blur_image = np.reshape(gray_blur_image,
                                     [gray_blur_image.shape[0],
                                      gray_blur_image.shape[0], 1])
        gray_blur_image = np.tile(gray_blur_image, 3)
        return blur_image, gray_blur_image

    def _create_retina_image(self, image, blur_rates, inv_blur_rates, gray_rates, inv_gray_rates):
        blur_image, gray_blur_image = self._create_blur_image(image)
        # Mix original and blur image
        blur_mix_image = image * blur_rates + blur_image * inv_blur_rates
        # Mix blur mixed image and gray blur image.
        gray_mix_image = blur_mix_image * gray_rates + gray_blur_image * inv_gray_rates
        return gray_mix_image.astype(np.uint8)

    # copied from https://github.com/wbap/oculomotor/blob/master/application/functions/lip.py
    def _get_saliency_magnitude(self, image):
        # Calculate FFT
        dft = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        magnitude, angle = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])

        log_magnitude = np.log10(magnitude.clip(min=1e-10))

        # Apply box filter
        log_magnitude_filtered = cv2.blur(log_magnitude, ksize=(3, 3))

        # Calculate residual
        magnitude_residual = np.exp(log_magnitude - log_magnitude_filtered)

        # Apply residual magnitude back to frequency domain
        dft[:, :, 0], dft[:, :, 1] = cv2.polarToCart(magnitude_residual, angle)
    
        # Calculate Inverse FFT
        image_processed = cv2.idft(dft)
        magnitude, _ = cv2.cartToPolar(image_processed[:, :, 0],
                                       image_processed[:, :, 1])
        return magnitude

    def _get_saliency_map(self, image):
        resize_shape = (64, 64) # (h,w)

        # Size argument of resize() is (w,h) while image shape is (h,w,c)
        image_resized = cv2.resize(image, resize_shape[1::-1])
        # (64,64,3)

        saliency = np.zeros_like(image_resized, dtype=np.float32)
        # (64,64,3)
    
        channel_size = image_resized.shape[2]
    
        for ch in range(channel_size):
            ch_image = image_resized[:, :, ch]
            saliency[:, :, ch] = self._get_saliency_magnitude(ch_image)

        # Calclate max over channels
        saliency = np.max(saliency, axis=2)
        # (64,64)

        saliency = cv2.GaussianBlur(saliency, GAUSSIAN_KERNEL_SIZE, sigmaX=8, sigmaY=0)

        SALIENCY_ENHANCE_COEFF = 2.0 # Strong saliency contrst
        #SALIENCY_ENHANCE_COEFF = 0.5 # Low saliency contrast, but sensible for weak saliency

        # Emphasize saliency
        saliency = (saliency ** SALIENCY_ENHANCE_COEFF)

        # Normalize to 0.0~1.0
        saliency = saliency / np.max(saliency)
    
        # Resize to original size
        saliency = cv2.resize(saliency, image.shape[1::-1])
        return saliency

    def _get_diff_map(self, prev_image, image):
        if prev_image is None:
            prev_image = image
        return cv2.absdiff(prev_image, image)

class RedCursorEnvironment(Environment):
    # TODO: 0.5 is enough?
    CAMERA_HORIZONTAL_ANGLE_RAND_MAX = 1.0 * CAMERA_HORIZONTAL_ANGLE_MAX
    CAMERA_VERTICAL_ANGLE_RAND_MAX = 1.0 * CAMERA_VERTICAL_ANGLE_MAX

    def __init__(self, content, off_buffer_width=128, on_buffer_width=640, skip_red_cursor=False, retina=False, saliency=False, diff=False):
        assert content == None # ignore content!!!
        assert skip_red_cursor == False
        super().__init__(PointToTargetContent(), off_buffer_width, on_buffer_width, False, retina, saliency, diff)
        self.reaction_step = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.reaction_step += 1
        if self.content.phase == 1: # PHASE_TARGET
             # skip to the target!!!
            reward, done, info = self.content.step([self.content.target_sprite.pos_x, self.content.target_sprite.pos_y])
            assert self.content.phase == 0 # PHASE_START
            assert reward == 2
            assert info['reaction_step'] == 1
            reward = 1
            info['reaction_step'] = self.reaction_step
            self.reaction_step = 0
            # randomly change camera angle.
            self.camera.cur_angle_h = 0
            self.camera.cur_angle_v = 0
            rand_h = (np.random.rand() * 2.0 - 1.0) * self.CAMERA_HORIZONTAL_ANGLE_RAND_MAX
            rand_v = (np.random.rand() * 2.0 - 1.0) * self.CAMERA_VERTICAL_ANGLE_RAND_MAX
            self.camera.change_angle(rand_h, rand_v)
            obs = self._get_observation()
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self.reaction_step = 0
        return obs
