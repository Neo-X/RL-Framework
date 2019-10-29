
from NNVisualize import NNVisualize
import sys
import json
import numpy as np
if __name__ == '__main__':
    
    
    file = open(sys.argv[1])
    settings = json.load(file)
    settings["save_video_to_file"] = True
    file.close()
    
    visualizeEvaluation = NNVisualize(title=str("Reward"), settings=None, nice=True)
    visualizeEvaluation.setInteractive()
    visualizeEvaluation.init(settings)
    
    rewards = settings["rewards"]
    
    for tr in rewards:
        # visualizeEvaluation.updateLoss([], np.zeros(0))
        visualizeEvaluation.setYLimit([min(tr)[0], max(tr)[0]])
        tr = np.array(tr).flatten()
        visualizeEvaluation.setXLimit([0, len(tr)])
        # visualizeEvaluation.redraw()
        print ("rewards: ", tr)
        for t in range(len(tr)):
            viz_q_values_ = tr[:t]
            print("viz_q_values_: ", viz_q_values_)
            visualizeEvaluation.updateLoss(viz_q_values_, np.zeros(len(tr)))
            visualizeEvaluation.redraw()
            

    visualizeEvaluation.finish()