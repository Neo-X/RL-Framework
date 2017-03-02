"""
A 2D bouncing ball environment

"""


import sys, os, random, time
from math import *
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
# from twisted.protocols import stateful
import copy
import math

    

class GapGame2D(GapGame1D):
    def __init__(self, settings):
        """Creates a ragdoll of standard size at the given offset."""
        
        super(GapGame2D,self).__init__(settings)
        
        
    def actContinuous(self, action, bootstrapping=False):
        # print ("Action: ", action)
        pos = self._obstacle.getPosition()
        # print ("Position Before action: ", pos)
        dist = action[0]
        if dist > self._game_settings["jump_bounds"][1]:
            dist = self._game_settings["jump_bounds"][1]
        elif dist < self._game_settings["jump_bounds"][0]:
            dist = self._game_settings["jump_bounds"][0]
        ## Move forward along X
        self.simulateAction([dist])
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
        
    
    def simulateAction(self, action):
        """
            Returns True if a contact was detected
        
        """
        if self._Paused:
            return
        t = self._dt - (time.time() - self._lasttime)    
        if self._game_settings['render']:
            if (t > 0):
                time.sleep(t)
        ## 
        if self._game_settings['render']:
            pos = self._obstacle.getPosition()
            steps=50
            hopTime=1.0
            x = np.array(np.linspace(-0.5, 0.5, steps))
            y = np.array(map(self._computeHeight, x))
            y = (y + math.fabs(float(np.amin(y)))) * 2.0
            x = np.array(np.linspace(0.0, 1.0, steps)) * action[0]
            # x = (x + 0.5) * action[0]
            x_ = (x + pos[0])
            for i in range(steps):
                ## Draw the ball arch
                self._obstacle.setPosition([x_[i], y[i], 0.0] )
                pos_ = self._obstacle.getPosition()
                # print ("New obstacle position: ", pos_)
                
                glutPostRedisplay()
                self.onDraw()
        return True
        
    

if __name__ == '__main__':
    import json
    settings={}
    # game = BallGame2D(settings)
    if (len(sys.argv)) > 1:
        _settings=json.load(open(sys.argv[1]))
        print (_settings)
        game = GapGame2D(_settings)
    else:
        settings['render']=True
        game = GapGame2D(settings)
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
            
            game.visualizeState(state[:len(state)-1], action, state[-1])
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
