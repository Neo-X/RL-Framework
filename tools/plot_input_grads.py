
"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt

bar_width = 0.35

opacity = 0.4

grad = [0.00402124, 0.01510576, 0.02074472, 0.02722632, 0.04227189,
        0.04035283, 0.02981968, 0.04495666, 0.03329496, 0.05631122,
        0.04687028, 0.03352509, 0.04437651, 0.07889991, 0.04241401,
        0.03624049, 0.04224908, 0.0626651 , 0.04146251, 0.05721659,
        0.04752586, 0.0424463 , 0.05601852, 0.04224522, 0.0360554 ,
        0.03665033, 0.03559469, 0.05341912, 0.0343102 , 0.06448108,
        0.05597209, 0.04008995, 0.04748913, 0.06633952, 0.03951846,
        0.04274476, 0.04864419, 0.04673169, 0.03969111, 0.04242162,
        0.03450051, 0.0407175 , 0.05100257, 0.05776333, 0.03845352,
        0.0443836 , 0.07535353, 0.05323818, 0.03750236, 0.03260067,
        0.05193072, 0.04686211, 0.05040986, 0.04405671, 0.03874316,
        0.03341392, 0.04079157, 0.06039948, 0.02956135, 0.02068681,
        0.02130714, 0.02322112, 0.01420606, 0.00934107, 0.02669422,
        0.02626251]

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

