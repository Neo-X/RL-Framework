import numpy as np

class BaseBuffer():
    '''
    Abstract buffer class
    '''

    def __init__(self):
        pass

    def add(self, obs):
        '''
        Add an observation to the buffer
        '''
        raise NotImplementedError

    def get_params(self):
        '''
        Get the sufficient statistics for the buffer
        '''
        raise NotImplementedError

    def logprob(self, obs):
        '''
        Return the logprob of an observation
        '''
        raise NotImplementedError

    def reset(self):
        '''
        Reset the buffer, clear its contents
        '''
        raise NotImplementedError

class BernoulliBuffer(BaseBuffer):
    def __init__(self, obs_dim):
        super().__init__()
        self.buffer = np.zeros(obs_dim)
        self.buffer_size = 1
        self.obs_dim = obs_dim
        
    def add(self, obs):
        self.buffer += obs
        self.buffer_size += 1

    def get_params(self):
        return np.array(self.buffer) / self.buffer_size

    def logprob(self, obs):
        obs = obs.copy()
        # ForkedPdb().set_trace()
        thetas = self.get_params()

        # For numerical stability, clip probs to not be 0 or 1
        thresh = 1e-5
        thetas = np.clip(thetas, thresh, 1 - thresh)

        # Bernoulli log prob
        probs = obs*thetas + (1-obs)*(1-thetas)
        logprob = np.sum(np.log(probs))
        return logprob

    def reset(self):
        self.buffer = np.zeros(self.obs_dim)
        self.buffer_size = 1

class GaussianBuffer(BaseBuffer):
    def __init__(self, obs_dim):
        super().__init__()
        self.buffer = np.zeros((1,obs_dim))
        self.buffer_size = 1
        self.obs_dim = obs_dim
        
    def add(self, obs):
        self.buffer = np.concatenate((self.buffer,obs.flatten()[np.newaxis,:]), axis=0)
        self.buffer_size += 1

    def get_params(self):
        #import pdb; pdb.set_trace()
        means = np.mean(self.buffer, axis=0)
        stds = np.std(self.buffer, axis=0)
        params = np.concatenate([means, stds])
        return params

    def logprob(self, obs):
        obs = obs.copy()
        means = np.mean(self.buffer, axis=0)
        stds = np.std(self.buffer, axis=0)
        
        # For numerical stability, clip stds to not be 0
        thresh = 1e-5
        thetas = np.clip(stds, thresh, None)
        #import pdb; pdb.set_trace()
        # Gaussian log prob
        logprob = -0.5*np.sum(np.log(2*np.pi*stds)) - np.sum(np.square(obs-means)/(2*np.square(stds)))
        return logprob

    def reset(self):
        self.buffer = np.zeros((1, self.obs_dim))
        self.buffer_size = 1
        
        
class GaussianCircularBuffer(BaseBuffer):
    def __init__(self, obs_dim, size):
        super().__init__()
        self.buffer = np.zeros((size,obs_dim))
        self.buffer_max_size=size
        self.buffer_size = 0
        ### Where to insert the next item
        self.buffer_pointer = 0
        self.obs_dim = obs_dim
        self.add(np.ones((1,obs_dim)))
        self.add(-np.ones((1,obs_dim)))
        
    def add(self, obs):
        self.buffer[self.buffer_pointer] = obs
        self.buffer_size += 1
        self.buffer_pointer = self.buffer_pointer + 1
        if self.buffer_pointer >= self.buffer_max_size:
            self.buffer_pointer = 0
        

    def get_params(self):
        #import pdb; pdb.set_trace()
        means = np.mean(self.buffer, axis=0)
        stds = np.std(self.buffer, axis=0)
        params = np.concatenate([means, stds])
        return params

    def logprob(self, obs):
        obs = obs.copy()
        ### Make sure to not include extra buffer samples in the beginning
        means = np.mean(self.buffer[:self.buffer_size], axis=0)
        stds = np.std(self.buffer[:self.buffer_size], axis=0)
        
        # For numerical stability, clip stds to not be 0
        thresh = 1e-5
        thetas = np.clip(stds, thresh, None)
        #import pdb; pdb.set_trace()
        # Gaussian log prob
        logprob = -0.5*np.sum(np.log(2*np.pi*stds)) - np.sum(np.square(obs-means)/(2*np.square(stds)))
        return logprob

    def reset(self):
        self.buffer = np.zeros((1, self.obs_dim))
        self.buffer_size = 1
