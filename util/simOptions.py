from optparse import OptionParser
import sys

def getOptions(_args=None):
    parser = OptionParser()

    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="print status messages to stdout")
    
    
    parser.add_option("--dataDir", "--datastoreDirectory", "--dataPath",
                  action="store", dest="dataPath", default="data/simData/",
                  metavar="Directory", help="Specify the directory that files will be stored")
    
    parser.add_option("-p", "--processes", "--availableProcesses",
              action="store", dest="processes", default=1,
              metavar="INTEGER", help="The number of processes the SteerStats script can use")
    
    parser.add_option("--frameSize", 
          action="store", dest="frameSize", default="640x480",
          metavar="IntegerxInteger", help="The pixel width and height, example 640x480")
    
    parser.add_option("-c", "--commandline", "--commandLine", 
          action="store_true", dest="commandLine", default=False,
          help="TO specify if an openGL window should be created")

    parser.add_option("--startPaused", 
           action="store_true", dest="startPaused", 
          default=False, help="""Start Simulation Paused TODO""")
    
    parser.add_option("--config", 
           action="store", metavar="STRING", dest="configFile", default=None,
          help="""The json config file that many of the config settings can be parsed from""")
    
    parser.add_option("--hyperConfig", 
           action="store", metavar="STRING", dest="hyperConfigFile", default=None,
          help="""The json config file that many of the config settings can be parsed from""")
    
    parser.add_option("--randomSeed", 
           action="store", dest="randomSeed", metavar="INTEGER",
          default=10, help="""randomSeed that will be used for random scenario generation.""")
    
    if _args is None:
        (options, args) = parser.parse_args()
    else:
        (options, args) = parser.parse_args(_args)

    return options
# print getOptions()

