# encoding: utf-8
import os
import yaml
import Help.ToolUtil as  ToolUtil

class ConfigManager():

    def __init__(self,configfile=None):
        if configfile == None:
            cfilePath = self.configFilePath("config.yaml")
            self.configfile = cfilePath
        else:
            self.configfile = configfile

        try:
            pfile = open(self.configfile)
            self._cfgs = yaml.load(pfile)
            pfile.close()

        except Exception as err:
            self._cfgs = []
            ToolUtil.print_error('Error reading config file {}, {}'.format(self.configfile, err))

    @property
    def cfgs(self):
        return self._cfgs

    def setup(self,session):
        pass

    def configFilePath(self,config_file):
        config_dir = os.path.abspath(os.path.dirname(__file__))
        config_path = os.path.join(config_dir,config_file)
        return config_path

def test():
    configManager = ConfigManager()
    cfgs = configManager.cfgs

    configDir = ToolUtil.getConfigDir()
    dataDir = ToolUtil.getDataDir()

    print "test success"


if __name__ == '__main__':
    print "test begine"
    test()
    print "test end"
