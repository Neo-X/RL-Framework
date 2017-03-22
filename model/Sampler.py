import numpy as np
from model.LearningAgent import LearningAgent

class Sampler(LearningAgent):
    """
        A simple method to sample the space of actions.
    """    
    def __init__(self, settings):

        super(Sampler,self).__init__(None, None, None, None, None, settings)
        
        self._x=[]
        self._samples=[]
        # action, value
        self._bestSample=[[0],[-10000000]]
        self._settings = settings
        
    def setSimulator(self, exp_):
        self._exp=exp_
    
    def generateSamplesFromNormal(self, mean, num_samples=25, repeate=1, variance_=[0.03]):
        if np.isscalar(variance_):
            variance_ = [variance_]*len(mean)
        # print ("input Var: " + str(variance_))
        samples = np.random.multivariate_normal(mean, np.diag(variance_), num_samples)
        return samples    
        
    def generateSamples(self, action_bounds, num_samples=5, repeate=1):
        spaces =[]
        for i in range(len(action_bounds[0])*repeate):
            spaces.append(np.linspace(action_bounds[0][i%len(action_bounds[0])]
                                      , action_bounds[1][i%len(action_bounds[0])], num_samples))
        
        grid = np.meshgrid(*spaces)
        
        samples=[]
        for grid_ in grid:
            samples.append(grid_.flatten())
            
        samples = np.array(samples).T
        return samples

    def sampleModel(self, model, forwardDynamics, current_state):
        self._samples = []
        self._bestSample=[[0],[-10000000]]
        pa = model.predict(current_state)
        action = pa
        # print ("Suggested Action: " + str(action) + " for state: " + str(current_state))
        bounds = [[],[]]
        for i in range(len(action)):
            bounds[0].append(-0.35+action[i])
            bounds[1].append(0.15+action[i])
        samples = self.generateSamples(action_bounds=bounds, num_samples=10)
        for samp in samples:
            pa = samp
            prediction = forwardDynamics.predict(state=current_state, action=pa)
            y = model.q_value(prediction)
            # y = self._game._reward(self._game._computeHeight(i+current_state[1]))
            # print (i, y)
            self._samples.append([samp,[y]])
            if y > self._bestSample[1][0]:
                self._bestSample[1][0] = y
                self._bestSample[0] = pa
                            
    def getBestSample(self): 
        return self._bestSample
    
    def setBestSample(self, samp): 
        self._bestSample = samp
    
    def predict(self, state):
        """
        Returns the best action
        """
        self.sampleModel(model=self._pol, forwardDynamics=self._fd, current_state=state)
        action = self.getBestSample()
        print ("Action: " + str(action))
        action = action[0]
        return action

    def q_value(self, state):
        """
        Returns the expected value of the state
        """
        return self.getBestSample()[1]
    
    
if __name__ == '__main__':
    
    samp = Sampler()
    bounds = [[-0.2,-0.3,-0.4],[0.2,0.3,0.4]]
    s = samp.generateSamples(bounds)
    print (s)