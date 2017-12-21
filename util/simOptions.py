from optparse import OptionParser
import sys

def getOptions(_args=None):
    parser = OptionParser()

    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="print status messages to stdout")
    
    parser.add_option("--dataDir", "--datastoreDirectory", "--dataPath",
                  action="store", dest="data_folder", default=None,
                  metavar="Directory", help="Specify the directory that files will be stored")
    
    parser.add_option("-p", "--processes", "--availableProcesses", "--num_available_threads",
                      "--p",
              action="store", dest="num_available_threads", default=None,
              type=int,
              metavar="INTEGER", help="The number of processes the SteerStats script can use")
    
    parser.add_option("--bootstrap_samples",
              action="store", dest="bootstrap_samples", default=None,
              type=int,
              metavar="INTEGER", 
              help="Number of initial actions to sample before calculating input/output scaling and starting to train.")
    
    parser.add_option("--eval_epochs",
              action="store", dest="eval_epochs", default=None,
              type=int,
              metavar="INTEGER", 
              help="Number of epoch/episode to evaluate the policy over")
    
    parser.add_option("--epochs",
              action="store", dest="epochs", default=None,
              type=int,
              metavar="INTEGER", 
              help="Number of epochs to perform per round")
    
    parser.add_option("--max_epoch_length",
              action="store", dest="max_epoch_length", default=None,
              type=int,
              metavar="INTEGER", 
              help="That max number of action that can be take before the end of an episode/epoch")
    
    
    parser.add_option("--frameSize", 
          action="store", dest="frameSize", default=None,
          metavar="IntegerxInteger", help="The pixel width and height, example 640x480")
    
    parser.add_option("--dont_visualize_learning", "--no_plot", 
          action="store_false", dest="visualize_learning", default=None,
          help="Whether or not to draw/render the simulation")
    
    parser.add_option("--shouldRender", "--render",
          action="store_true", dest="shouldRender", default=None,
          help="TO specify if an openGL window should be created")

    parser.add_option("--config", 
           action="store", metavar="STRING", dest="configFile", default=None,
          help="""The json config file that many of the config settings can be parsed from""")
    
    parser.add_option("--hyperConfig", 
           action="store", metavar="STRING", dest="hyperConfigFile", default=None,
          help="""The json config file that many of the config settings can be parsed from""")
    
    parser.add_option("--randomSeed", 
           action="store", dest="randomSeed", metavar="INTEGER", default=None,
           help="""randomSeed that will be used for random scenario generation.""")
    
    if _args is None:
        (options, args) = parser.parse_args()
    else:
        (options, args) = parser.parse_args(_args)

    return options
# print getOptions()

