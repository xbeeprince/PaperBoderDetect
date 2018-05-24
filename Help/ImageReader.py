# encoding: utf-8
import os
import time
import numpy as np
from PIL import Image
import Help.ToolUtil as  ToolUtil
from Config.ConfigManager import ConfigManager

class ImageReader():
    def __init__(self,cfgs):
        self._cfgs = cfgs
        self.setup(cfgs)

    def setup(self,cfgs):
        download_path = ToolUtil.getDirWithPath(cfgs['download_path'])
        self.train_file = os.path.join(download_path, cfgs['training']['list'])
        self.train_data_dir = os.path.join(download_path, cfgs['training']['dir'])

        ToolUtil.print_info('Training data set-up from {}'.format(self.train_file))
        self.training_pairs = ToolUtil.read_file_list(self.train_file)
        self.samples = self.split_pair_names(self.training_pairs, self.train_data_dir)

        self.n_samples = len(self.training_pairs)
        self.all_ids = range(self.n_samples)
        np.random.shuffle(self.all_ids)

        self.training_ids = self.all_ids[:int(self.cfgs['train_split'] * len(self.training_pairs))]
        self.validation_ids = self.all_ids[int(self.cfgs['train_split'] * len(self.training_pairs)):]

        ToolUtil.print_info('Training samples {}'.format(len(self.training_ids)))
        ToolUtil.print_info('Validation samples {}'.format(len(self.validation_ids)))

    @property
    def cfgs(self):
        return  self._cfgs

    def read_file_list(self, filelist):
        pfile = open(filelist)
        filenames = pfile.readlines()
        pfile.close()

        filenames = [f.strip() for f in filenames]

        return filenames

    def split_pair_names(self, filenames, base_dir):
        filenames = [c.split(' ') for c in filenames]
        filenames = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1])) for c in filenames]

        return filenames

    def get_training_batch(self):
        batch_ids = np.random.choice(self.training_ids, self.cfgs['batch_size_train'])

        return self.get_batch(batch_ids)

    def get_validation_batch(self):
        batch_ids = np.random.choice(self.validation_ids, self.cfgs['batch_size_val'])

        return self.get_batch(batch_ids)

    def get_batch(self, batch):

        tstart = time.time()

        filenames = []
        images = []
        edgemaps = []

        for idx, b in enumerate(batch):

            im = Image.open(self.samples[b][0])
            em = Image.open(self.samples[b][1])

            im = im.resize((self.cfgs['training']['image_width'], self.cfgs['training']['image_height']))
            em = em.resize((self.cfgs['training']['image_width'], self.cfgs['training']['image_height']))

            im = np.array(im, dtype=np.float32)
            im = im[:, :, self.cfgs['channel_swap']]
            im -= self.cfgs['mean_pixel_value']

            # Labels needs to be 1 or 0 (edge pixel or not)
            # or can use regression targets as done by the author
            # https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/src/caffe/layers/image_labelmap_data_layer.cpp#L213

            em = np.array(em.convert('L'), dtype=np.float32)

            if self.cfgs['target_regression']:
                bin_em = em / 255.0
            else:
                bin_em = np.zeros_like(em)
                bin_em[np.where(em)] = 1

            # Some edge maps have 3 channels some dont
            bin_em = bin_em if bin_em.ndim == 2 else bin_em[:, :, 0]
            # To fit [batch_size, H, W, 1] output of the network
            bin_em = np.expand_dims(bin_em, 2)

            images.append(im)
            edgemaps.append(bin_em)
            filenames.append(self.samples[b])

        return images, edgemaps, filenames


def ImageReaderTest():
    print "test run begine ..."
    configManager = ConfigManager()
    imagereader = ImageReader(configManager.cfgs)
    images,edgemaps,filenames = imagereader.get_training_batch()

    print "test run end ..."



if __name__ == '__main__':
    ImageReaderTest()
