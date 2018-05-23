
import os
import sys
import tensorflow as tf
import PaperBorderDetect.Help.ToolUtil as  ToolUtil
from PaperBorderDetect.Config.ConfigManager import ConfigManager
from PaperBorderDetect.Help.ImageReader import ImageReader
from DetectModel import DetectModel

class DetectModelTrain():

    def __init__(self,config_file):
        self.init = True
        configManager = ConfigManager(config_file)
        self.cfgs = configManager.cfgs
        self.setup()

    def setup(self):

        try:
            self.detectModel = DetectModel(self.cfgs)
            ToolUtil.print_info('Done initializing VGG-16 Detect model')

            dirs = ['train', 'val', 'test', 'models']
            dirs = [os.path.join(self.cfgs['save_dir'] + '/{}'.format(d)) for d in dirs]
            _ = [os.makedirs(d) for d in dirs if not os.path.exists(d)]

        except Exception as err:
            ToolUtil.print_error('Error setting up VGG-16 model, {}'.format(err))
            self.init = False

    def run(self,session):
        if not self.init:
            return

        train_data = ImageReader(self.cfgs)
        self.detectModel.setup_training(session)
        opt = tf.train.AdadeltaOptimizer(self.cfgs['optimizer_params']['learning_rate'])
        train = opt.minimize(self.detectModel.loss)
        session.run(tf.global_variables_initializer())

        for index in range(self.cfgs["max_iterations"]):
            im, em, _ = train_data.get_training_batch();



def test():
    config_path = "/Users/prince/PycharmProjects/worddetect/PaperBorderDetect/Config/config.yaml"

    print "test success"


if __name__ == '__main__':
    print "test begine"
    test()
    print "test end"
