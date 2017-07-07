import numpy as np
# import lasagne
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import math
import random
import json

def sum_furture_discounted_rewards(rewards, discount_factor):
    discounts=[]
    for k in range(len(rewards)):
        discounts.append(0)
        for i in range(len(rewards)-k):
            discounts[k] += rewards[k+i] * math.pow(discount_factor, i)
            print ("discounts: ", k, " ", i, " reward:", rewards[k+i],  " discounts", discounts)
    return discounts

def compute_advantage(discounted_rewards, rewards, discount_factor):
    discounts = []
    for i in range(len(discounted_rewards)-1):
        discounts.append(((discounted_rewards[i+1] * discount_factor) + rewards[i+1]) - discounted_rewards[i])
    return discounts

if __name__ == "__main__":
    
    discount_factor = 0.9
    if (len(sys.argv) == 2):
        discount_factor = float(sys.argv[1])
    
    print ("Discount Factor: ", discount_factor)
    rewards=[0.2, 0.25, 0.2, 0.2, 0]
    # rewards2 = rewards[1:]
    # print ("Rewards2: ", rewards2)
    discounts = sum_furture_discounted_rewards(rewards, discount_factor)
    print("Discounts: ", discounts)
    advantage = compute_advantage(discounts, rewards, discount_factor)
    print ("Advantage: ", advantage)
    
    
    