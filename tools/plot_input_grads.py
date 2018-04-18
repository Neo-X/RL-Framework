
"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt

bar_width = 0.35

opacity = 0.4

grad = [0.0024819 , 0.00379737, 0.00597974, 0.00538466, 0.00937911,
        0.00913581, 0.00948872, 0.01148854, 0.01398885, 0.01573431,
        0.02332864, 0.02498847, 0.01966272, 0.02536297, 0.02662395,
        0.02262606, 0.01559444, 0.02632327, 0.02207177, 0.01828032,
        0.01324646, 0.02582618, 0.02292261, 0.01780515, 0.02579798,
        0.01446737, 0.01477636, 0.01838213, 0.02008937, 0.02320273,
        0.01481252, 0.01907307, 0.01377775, 0.02117879, 0.01454772,
        0.01900331, 0.01962486, 0.01792726, 0.01886086, 0.02098123,
        0.01419879, 0.02679272, 0.02320086, 0.01290695, 0.02121896,
        0.0173247 , 0.01245007, 0.0260509 , 0.01574942, 0.01311136,
        0.01233317, 0.00874183, 0.00940799, 0.00931895, 0.0115082 ,
        0.00818907, 0.00532555, 0.0053509 , 0.00509559, 0.00489656,
        0.00133538, 0.        , 0.        , 0.        , 0.0062647 ,
        0.0343503 ]

print("Number of state features: ", len(grad))

index = np.arange(len(grad))

rects1 = plt.bar(index, grad, bar_width,
                 alpha=opacity,
                 color='b',
                 )

plt.xlabel('State Feature')
plt.ylabel('Average gradient')
plt.title('Comparing input feature gradients')
# plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D', 'E'))
plt.legend()

plt.tight_layout()
plt.savefig("input_grads.svg")
plt.show()

