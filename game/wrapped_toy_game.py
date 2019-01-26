"""
This game engine simulates a segmentation tracing game environment (env) for a DRL agent (agent).

Overview:
    - The agent is a dot with position (y, x). It can take 4 + 1 actions, UP, DOWN, LEFT and RIGHT, and DO NOTHING.

    - The env takes an image and a label, and a list of seed points to initialize.
      The env returns the image of the next step, the reward for the action and if
      the game has finished.

Details:
    - The env only reveals a SCREENHEIGHT x SCREENWIDTH window centered around the agent's position at a time;
    - The env takes in an action taken by the agent, and moves the window in that direction,
      returns the cropped window in the next slice (i.e., the env cycles through the slices automatically);
    - If the agent's position is within the masks of that slice, then it gets a reward;
    - The reward is higher if the agent's position is closer to the center of the segmentation mask in that slice;
    - The maximum reward is 1, and falls off the center following a Gaussian distribution;
    - If the agent is out of the segmentation for MAX_STRAY slices, then the game is over, and gets a reward of -10.

"""
import SimpleITK as sitk
import scipy
import pygame
import numpy as np
from pygame.locals import *


SCREENHEIGHT = 512
SCREENWIDTH = 512
PATCHHEIGHT = 128
PATCHWIDTH = 128
MAX_STRAY = 5
STEP_SIZE = 1 # number of pixels per move

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Toy Game')


def load():
    image = '/data/pliu/test/rib_nii/0_DICOMC_PA10_ST0_SE3.nii.gz'
    label = '/data/pliu/test/rib_nii/0_DICOMC_PA10_ST0_SE3.post.seg.nii.gz'
    # in the order of z, y, x
    images = sitk.GetArrayFromImage(sitk.ReadImage(image))[::-1, ...]
    labels = sitk.GetArrayFromImage(sitk.ReadImage(label))[::-1, ...]
    return images, labels


def get_2d_centroid(image_array):
    assert len(image_array.shape) == 2
    y, x = scipy.ndimage.measurements.center_of_mass(image_array)
    return y, x


def get_init_agent_position(labels, cc_id=None, mode='random'):
    """Get the seed position in a new tracing episode

    Args:
        mode: random or top

    Returns:

    """
    if cc_id is None:
        cc_id = np.random.choice(np.unique(labels[labels > 0]))
    assert cc_id > 0
    binary_mask = (labels == cc_id)
    num_z = binary_mask.shape[0]
    sum_z = binary_mask.reshape([num_z, -1]).sum(axis=1)
    if mode == 'random':
        pos_z = np.random.choice(np.where(sum_z > 0)[0])
    elif mode == 'top':
        pos_z = np.min(np.where(sum_z > 0)[0])
    label_slice = binary_mask[pos_z, ...]
    pos_y, pos_x = get_2d_centroid(label_slice)
    return pos_z, pos_y, pos_x


def get_cc_mask(label_slice, center_yx):
    y, x = center_yx
    cc_id = label_slice[y, x]
    cc_mask = (label_slice == cc_id)
    return cc_mask


def crop_around_point(image_array, center_yx, target_shape):
    """Center crop an image array around a point

    Args:
        image_array:
        center_yx:
        target_shape:

    Returns:

    """
    pad_y, pad_x = ((np.array(target_shape) + 1) // 2).astype(np.int)
    pad_width = ((pad_y, pad_y), (pad_x, pad_x))
    image_array = np.pad(image_array, pad_width=pad_width, mode='constant')
    ymin, xmin = (np.array(center_yx) + np.array([pad_y, pad_x]) - np.array(target_shape) // 2).astype(np.int)
    ymax, xmax = (np.array([ymin, xmin]) + np.array(target_shape)).astype(np.int)
    cropped_array = image_array[ymin:ymax, xmin:xmax]
    assert cropped_array.shape == tuple(target_shape)
    return cropped_array


images, labels = load()


class GameState:
    def __init__(self, images, labels, show_play=True, FPS=30):
        self.score = 0
        # 3d images
        self.images = images
        self.labels = labels

        # get initial position of agent
        self.playerz, self.playery, self.playerx = get_init_agent_position(labels, mode='random')
        self.playeryx = (self.playery, self.playerx)

        # crop patch
        self.image_slice = self.images[self.playerz, ...]
        self.label_slice = get_cc_mask(self.labels[self.playerz, ...], self.playeryx)
        self.patch = self.crop(self.image_slice, self.playeryx)
        self.patch_mask = self.crop(self.label_slice, (self.playery, self.playerx))

        self.stray_counter = 0
        self.show_play = show_play
        self.FPS = FPS

    @staticmethod
    def crop(image_slice, center_yx):
        cropped_array = crop_around_point(image_slice, center_yx, target_shape=(PATCHHEIGHT, PATCHWIDTH))
        return cropped_array

    def frame_step(self, input_actions):
        """

        Args:
            input_actions: one hot vector of length 5

        Returns:

        """
        pygame.event.pump()

        reward = 1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: UP,    y - 1
        # input_actions[2] == 1: DOWN,  y + 1
        # input_actions[3] == 1: LEFT,  x - 1
        # input_actions[4] == 1: RIGHT, x + 1
        if input_actions[1] == 1:
            self.playery -= 1
        if input_actions[2] == 1:
            self.playery += 1
        if input_actions[3] == 1:
            self.playerx -= 1
        if input_actions[4] == 1:
            self.playerx += 1

        # check for score
        self.score += 1
        reward = 1

        # player's movement
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)

        # check if the point is stray (outside mask)
        is_stray = check_stray({'x': self.playerx, 'y': self.playery}, self.cc_mask)
        if is_stray:
            self.stray_counter += 1
            if self.stray_counter >= MAX_STRAY:
                terminal = True
                # init with the old parameters
                self.__init__(FPS=self.FPS, show_play=self.show_play, game_level=self.game_level)
                reward = -1
        else:
            self.stray_counter = 0

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        if self.show_play:
            pygame.display.update()
        if self.FPS > 0:
            FPSCLOCK.tick(self.FPS)
        # print self.upperPipes[0]['y'] + PIPE_HEIGHT - int(BASEY * 0.2)
        return image_data, reward, terminal



def check_stray(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                                 player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False

