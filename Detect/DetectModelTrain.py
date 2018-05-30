# encoding: utf-8
import os
import sys
import tensorflow as tf
import Help.ToolUtil as  ToolUtil
from Config.ConfigManager import ConfigManager
from Help.ImageReader import ImageReader
from DetectModel import DetectModel

class DetectModelTrain():

    def __init__(self,dataDir=None,saveDir=None,initmodelfile=None,configfile=None):
        self.init = True
        self.dataDir = dataDir
        self.saveDir = saveDir
        self.initmodelfile = initmodelfile
        self.configfile = configfile
        configManager = ConfigManager()
        self.cfgs = configManager.cfgs

    def setup(self):

        try:
            self.detectModel = DetectModel(self.cfgs,self.saveDir,self.initmodelfile)
            ToolUtil.print_info('Done initializing VGG-16 Detect model')

            dirs = ['train', 'val', 'test', 'models']
            if self.saveDir == None:
                save_dir = ToolUtil.getDirWithPath(self.cfgs['save_dir'])
            else:
                save_dir = self.saveDir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

            dirs = [os.path.join(save_dir + '/{}'.format(d)) for d in dirs]
            _ = [os.makedirs(d) for d in dirs if not os.path.exists(d)]

        except Exception as err:
            ToolUtil.print_error('Error setting up VGG-16 model, {}'.format(err))
            self.init = False

    def run(self,session):
        if not self.init:
            return

        train_data = ImageReader(self.cfgs,self.dataDir)
        self.detectModel.setup_training(session)
        opt = tf.train.AdadeltaOptimizer(self.cfgs['optimizer_params']['learning_rate'])
        train = opt.minimize(self.detectModel.loss)
        session.run(tf.global_variables_initializer())

        for index in range(self.cfgs["max_iterations"]):
            im, em, _ = train_data.get_training_batch()
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            _, summary, loss = session.run([train, self.detectModel.merged_summary, self.detectModel.loss],
                                           feed_dict={self.detectModel.images: im, self.detectModel.edgemaps: em},
                                           options=run_options,
                                           run_metadata=run_metadata)

            self.detectModel.train_writer.add_run_metadata(run_metadata, 'step{:06}'.format(index))
            self.detectModel.train_writer.add_summary(summary, index)

            ToolUtil.print_info('[{}/{}] TRAINING loss : {}'.format(index, self.cfgs['max_iterations'], loss))

            if index % self.cfgs['save_interval'] == 0:
                saver = tf.train.Saver()
                if self.saveDir == None:
                    save_dir = ToolUtil.getDirWithPath(self.cfgs['save_dir'])
                else:
                    save_dir = self.saveDir
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                saver.save(session, os.path.join(save_dir, 'models/detect-model'), global_step=index)
                ToolUtil.print_info('save detect model with index : {}'.format(index))

            if index % self.cfgs['val_interval'] == 0:
                im, em, _ = train_data.get_validation_batch()

                summary, error = session.run([self.detectModel.merged_summary, self.detectModel.error],
                                             feed_dict={self.detectModel.images: im, self.detectModel.edgemaps: em})

                self.detectModel.val_writer.add_summary(summary, index)
                ToolUtil.print_error('[{}/{}] VALIDATION error : {}'.format(index, self.cfgs['max_iterations'], error))

        self.detectModel.train_writer.close()

def test():
    session = ToolUtil.get_session(0.4)
    trainer = DetectModelTrain()
    trainer.setup()
    trainer.run(session)
    print "test success"


if __name__ == '__main__':
    print "test begine : " + ToolUtil.get_local_time()
    test()
    print "test end : " + ToolUtil.get_local_time()
