import numpy as np
# import lasagne
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import math
import random
import json

def sum_furture_discounted_reward(rewards, discount_factor):
    discount = 0
    for i in range(len(rewards)):
        discount += rewards[i] * math.pow(discount_factor, i)
        
    return discount

if __name__ == "__main__":
    
    discount_factor = 0.9
    if (len(sys.argv) == 2):
        discount_factor = float(sys.argv[1])
    
    print ("Discount Factor: ", discount_factor)
    rewards=[0.2, 0.25, 0.2, 0.2, 0]
    rewards2 = rewards[1:]
    print ("Rewards2: ", rewards2)
    discount = sum_furture_discounted_reward(rewards, discount_factor)
    print ("Discount: ", discount)
    discount_ = sum_furture_discounted_reward(rewards2, discount_factor)
    print ("Discount2: ", discount_)
    print ("Advantage: ", discount_-discount)
    
    
    