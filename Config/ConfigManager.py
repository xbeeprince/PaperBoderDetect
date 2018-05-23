# encoding: utf-8
import os
import yaml
import PaperBorderDetect.Help.ToolUtil as  ToolUtil

class ConfigManager():

    def __init__(self,config_path):
        try:
            pfile = open(config_path)
            self._cfgs = yaml.load(pfile)
            pfile.close()

        except Exception as err:
            self._cfgs = []
            ToolUtil.print_error('Error reading config file {}, {}'.format(config_path, err))

    @property
    def cfgs(self):
        return self._cfgs

    def setup(self,session):
        pass



def test():
    config_path = "/Users/prince/PycharmProjects/worddetect/PaperBorderDetect/Config/config.yaml"
    configManager = ConfigManager(config_path)
    cfgs = configManager.cfgs
    print "test success"


if __name__ == '__main__':
    print "test begine"
    test()
    print "test end"
