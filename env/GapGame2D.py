"""
A 2D bouncing ball environment

"""


import sys, os, random
from math import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
# from twisted.protocols import stateful
import copy
import math
from .GapGame1D import *

    

class GapGame2D(GapGame1D):
    def __init__(self, settings):
        """Creates a ragdoll of standard size at the given offset."""
        
        super(GapGame2D,self).__init__(settings)
        
        
    def actContinuous(self, action, bootstrapping=False):
        # print ("Action: ", action)
        pos = self._obstacle.getPosition()
        vel = self._obstacle.getLinearVel()
        # print ("Position Before action: ", pos)
        time = (action[1]/9.81)*2 # time for rise and fall
        new_vel = np.array([vel[0] + action[0], action[1]])
        new_vel = clampAction(new_vel, self._game_settings["velocity_bounds"])
        self._obstacle.setLinearVel((new_vel[0], new_vel[1], 0))
        ## Move forward along X
        self.simulateAction(new_vel)
        dist = new_vel[0] * time
        self._obstacle.setPosition(pos + np.array([dist, 0.0, 0.0]))
        # print ("Position After action: ", pos + np.array([dist, 0.0, 0.0]))

        # print (pos)

        # self._terrainData = self.generateTerrain()
        self._state_num=self._state_num+1
        # state = self.getState()
        # print ("state length: " + str(len(state)))
        # print (state)
        return dist
        # obstacle.addForce((0.0,100.0,0.0))
        
    
if __name__ == '__main__':
    import json
    settings={}
    # game = BallGame2D(settings)
    if (len(sys.argv)) > 1:
        _settings=json.load(open(sys.argv[1]))
        print (_settings)
        _settings['render']=True
        game = GapGame2D(_settings)
    else:
        _settings['render']=True
        game = GapGame2D(_settings)
    game.init()
    for j in range(100):
        # game.generateEnvironmentSample()
        game.generateValidationEnvironmentSample(j)
        print ("Starting new epoch")
        game.initEpoch()
        i=0
        while not game.endOfEpoch():
        # for i in range(50):
            # state = game.getState()
            
            # action = model.predict(state)
            _action =  np.random.random([1])[0] * 2 + 0.5
            action = [_action,4.0]
            state = game.getState()
            pos = game._obstacle.getPosition()
            # drawTerrain(state, pos[0], translateY=0.0, colour=(0.6, 0.6, 0.9, 1.0))
            # print ("State: " + str(state[-8:]))
            # print ("character State: " + str(game.getCharacterState()))
            # print ("rot Vel: " + str(game._obstacle.getQuaternion()))
            
            # print (state)
            
            game.visualizeState(state[:len(state)-1], action, state[_settings['num_terrain_samples']])
            reward = game.actContinuous(action)
            
            if (game.agentHasFallen()):
                print (" *****Agent fell in a hole")
            
            if ( reward < 0.00001 ):
                print("******Agent has 0 reward?")
            
            print ("Reward: " + str(reward) + " on action: ", action, " actions: ", i)
            # print ("Number of geoms in space: ", game._space.getNumGeoms())
            # print ("Random rotation matrix", list(np.reshape(rand_rotation_matrix(), (1,9))[0]))
            i=i+1
            game._lasttime = time.time()
            
    game.finish()
