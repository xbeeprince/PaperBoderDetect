# encoding: utf-8
import os
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
import cStringIO
import Help.ToolUtil as  ToolUtil
from Config.ConfigManager import ConfigManager
from Help.ImageReader import ImageReader
from DetectModel import DetectModel

class DetectModelTest():

    def __init__(self):
        self.init = True
        configManager = ConfigManager()
        self.cfgs = configManager.cfgs

    def setup(self,session):

        try:

            self.detectModel = DetectModel(self.cfgs, run='testing')

            save_dir = ToolUtil.getDirWithPath(self.cfgs['save_dir'])
            meta_model_file = os.path.join(save_dir,'models/hed-model-{}'.format(self.cfgs['test_snapshot']))

            saver = tf.train.Saver()
            saver.restore(session, meta_model_file)

            ToolUtil.print_info('Done restoring VGG-16 model from {}'.format(meta_model_file))

        except Exception as err:

            ToolUtil.print_error('Error setting up VGG-16 model, {}'.format(err))
            self.init = False

    def run(self,session):
        if not self.init:
            return

        self.detectModel.setup_testing(session)

        filepath = os.path.join(self.cfgs['download_path'], self.cfgs['testing']['list'])
        train_list = ToolUtil.read_file_list(filepath)

        ToolUtil.print_info('Writing PNGs at {}'.format(self.cfgs['test_output']))

        for index, img in enumerate(train_list):
            test_filename = os.path.join(self.cfgs['download_path'], self.cfgs['testing']['dir'], img)
            im = self.fetch_image(test_filename)

            edgemap = session.run(self.detectModel.predictions, feed_dict={self.detectModel.images: [im]})
            self.save_egdemaps(edgemap, index)

            ToolUtil.print_info('Done testing {}, {}'.format(test_filename, im.shape))

    def save_egdemaps(self, em_maps, index):

        # Take the edge map from the network from side layers and fuse layer
        em_maps = [e[0] for e in em_maps]
        em_maps = em_maps + [np.mean(np.array(em_maps), axis=0)]

        for idx, em in enumerate(em_maps):
            em[em < self.cfgs['testing_threshold']] = 0.0

            em = 255.0 * (1.0 - em)
            em = np.tile(em, [1, 1, 3])

            em = Image.fromarray(np.uint8(em))
            em.save(os.path.join(self.cfgs['test_output'], 'testing-{}-{:03}.png'.format(index, idx)))

    def fetch_image(self, test_image):

        image = None

        try:

            fid = open(test_image, 'r')
            stream = fid.read()
            fid.close()

            image_buffer = cStringIO.StringIO(stream)
            image = self.capture_pixels(image_buffer)

        except Exception as err:

            ToolUtil.print_error('[Testing] Error with image file {0} {1}'.format(test_image, err))
            return None

        return image

    def capture_pixels(self, image_buffer):

        image = Image.open(image_buffer)
        image = image.resize((self.cfgs['testing']['image_width'], self.cfgs['testing']['image_height']))
        image = np.array(image, np.float32)
        image = self.colorize(image)

        image = image[:, :, self.cfgs['channel_swap']]
        image -= self.cfgs['mean_pixel_value']

        return image

    def colorize(self, image):

        # BW to 3 channel RGB image
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
            image = np.tile(image, (1, 1, 3))
        elif image.shape[2] == 4:
            image = image[:, :, :3]

        return image

def test():

    print "test success"


if __name__ == '__main__':
    print "test begine"
    test()
    print "test end"
