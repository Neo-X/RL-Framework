
"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt

bar_width = 0.35

opacity = 0.4

grad = [0.0942025 , 0.02434091, 0.0228947 , 0.04949658, 0.04313976,
        0.03984146, 0.04944463, 0.07493392, 0.04407027, 0.03407219,
        0.02663133, 0.04451901, 0.07005873, 0.0713006 , 0.04167103,
        0.01631143, 0.02670271, 0.01637469, 0.01865671, 0.04340999,
        0.03781549, 0.01252457, 0.02510559, 0.01588326, 0.04396837,
        0.10386589, 0.02406434, 0.03078043, 0.00995562, 0.02293286,
        0.01245463, 0.03307493, 0.01163314, 0.03208698, 0.01564521,
        0.01402407, 0.00874195, 0.04198382, 0.02782716, 0.0256097 ,
        0.00493809, 0.00436766, 0.00820759, 0.02776992, 0.00577986,
        0.01006845, 0.01218997, 0.0444144 , 0.01349473, 0.00835981,
        0.00732449, 0.00703817, 0.00905028]

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
plt.savefig("input_grads.png")
plt.show()

