
"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt

bar_width = 0.35

opacity = 0.4

grad = [ 0.06669209,  0.06351989,  0.08471397,  0.06760015,  0.03695782,
         0.13560015,  0.03050445,  0.19822968,  0.08495502,  0.06139053,
         0.01683631,  0.07584076,  0.01913538,  0.13303461,  0.03071972,
         0.03461015,  0.04567533,  0.01492957,  0.06092968,  0.03621647,
         0.06358031,  0.01823268,  0.04462435,  0.05995731,  0.08405554,
         0.13653299,  0.04615058,  0.04742992,  0.02055108,  0.03597774,
         0.02008903,  0.02110897,  0.04000919,  0.04817809,  0.0190045 ,
         0.03409175,  0.02671746,  0.02267606,  0.01251312,  0.02348053,
         0.00790723,  0.00786928,  0.00969568,  0.02720484,  0.01058125,
         0.00859307,  0.00882772,  0.07831702,  0.03738438,  0.23550569,
         0.1481218 ,  0.19027106,  0.14136482,  0.11093478]

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
plt.show()
