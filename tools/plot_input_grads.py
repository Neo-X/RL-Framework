
"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt

bar_width = 0.35

opacity = 0.4

grad = [8.48625277e-05, 6.76283453e-05, 1.41438577e-04, 1.06085958e-04,
        2.03063304e-04, 1.67429767e-04, 4.36686503e-04, 3.71184258e-04,
        3.73028684e-04, 2.16673725e-04, 2.40558482e-04, 2.08362952e-04,
        3.89204477e-04, 2.77638552e-04, 8.92655924e-04, 3.61500192e-04,
        9.34928772e-04, 7.55625544e-04, 7.08408072e-04, 7.15131289e-04,
        6.52113929e-04, 5.73010708e-04, 3.98972479e-04, 3.49660113e-04,
        5.25782641e-04, 4.03998507e-04, 5.17697132e-04, 4.07818239e-04,
        6.77318545e-04, 4.65902936e-04, 7.53912085e-04, 3.92293587e-04,
        4.81306080e-04, 2.97889841e-04, 5.49435325e-04, 3.12289660e-04,
        6.40939281e-04, 3.50816146e-04, 3.66569817e-04, 3.51424445e-04,
        5.55817969e-04, 3.17700265e-04, 3.44725442e-04, 3.01878463e-04,
        4.38548595e-04, 2.92233715e-04, 2.56654777e-04, 1.58839743e-04,
        2.51225894e-04, 1.74880828e-04, 1.83470678e-04, 1.31692985e-04,
        1.89412880e-04, 1.17185598e-04, 9.83999562e-05, 8.50170036e-05,
        6.89301669e-05, 4.27726118e-05, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.06385334e-04, 2.20393020e-04, 1.61519507e-04, 1.78558927e-04,
        2.75459490e-04, 3.36841127e-04, 3.49287671e-04, 5.41115704e-04,
        4.20219469e-04, 1.41377124e-04, 3.56215896e-04, 2.04414173e-04,
        6.89845416e-04, 3.01449618e-04, 8.40886380e-04, 5.82821143e-04,
        1.14187202e-03, 1.18012191e-03, 1.04943651e-03, 1.22858258e-03,
        7.87668687e-04, 5.52291051e-04, 6.48878515e-04, 5.46238152e-04,
        8.40417808e-04, 5.38367778e-04, 8.71583063e-04, 7.29453226e-04,
        7.82340125e-04, 5.34188701e-04, 6.76959346e-04, 6.68419816e-04,
        5.83981629e-04, 4.79676441e-04, 5.64685150e-04, 5.11355349e-04,
        7.35630980e-04, 7.18029856e-04, 6.01451378e-04, 6.62307488e-04,
        6.27318223e-04, 7.93015526e-04, 5.24943869e-04, 7.81916373e-04,
        4.95240092e-04, 5.40163019e-04, 4.12524096e-04, 3.03731940e-04,
        4.28170111e-04, 2.18021640e-04, 2.76144652e-04, 2.30741454e-04,
        1.64779354e-04, 2.19312831e-04, 1.66785510e-04, 9.82432757e-05,
        1.22571859e-04, 6.36058976e-05, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.50671898e-04, 2.69833923e-04, 2.28833582e-04, 1.38798772e-04,
        3.72797920e-04, 4.02958598e-04, 4.22595476e-04, 8.09791905e-04,
        4.57605114e-04, 1.97234287e-04, 5.79279731e-04, 3.77832563e-04,
        7.92815001e-04, 5.96173049e-04, 8.19699431e-04, 5.43258386e-04,
        7.59965158e-04, 1.03281636e-03, 8.42541049e-04, 1.77076529e-03,
        1.12411915e-03, 1.02828466e-03, 8.65654729e-04, 1.18131551e-03,
        9.59395838e-04, 1.01630762e-03, 9.38429963e-04, 8.87287606e-04,
        7.91884377e-04, 7.21296878e-04, 6.86024781e-04, 1.10253855e-03,
        1.59716653e-03, 9.57613403e-04, 9.36860510e-04, 9.48966481e-04,
        9.17818106e-04, 1.12363789e-03, 6.49531779e-04, 8.08335375e-04,
        9.69642308e-04, 1.11186958e-03, 6.53663592e-04, 1.30298047e-03,
        5.90751064e-04, 9.10614850e-04, 5.52118756e-04, 5.79074142e-04,
        9.41041857e-04, 4.10919340e-04, 5.93547185e-04, 3.33918404e-04,
        2.88776122e-04, 2.86725612e-04, 1.96986250e-04, 1.44977195e-04,
        1.92537482e-04, 1.72806613e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        3.00801330e-04, 2.21465889e-04, 5.24468604e-04, 2.95086356e-04,
        5.83314977e-04, 3.79500911e-04, 5.35787141e-04, 9.78907803e-04,
        7.32241722e-04, 6.47665816e-04, 8.53534380e-04, 6.56192715e-04,
        9.51772905e-04, 1.66605751e-03, 9.06763249e-04, 1.36390852e-03,
        1.12833688e-03, 1.75891491e-03, 1.12236256e-03, 2.59155757e-03,
        1.38960418e-03, 1.52684795e-03, 1.42132968e-03, 1.34753610e-03,
        2.18310952e-03, 1.20411930e-03, 1.81410089e-03, 1.22440374e-03,
        1.31828978e-03, 1.57093583e-03, 1.16630411e-03, 1.41995493e-03,
        2.02781637e-03, 1.46222219e-03, 1.35029911e-03, 1.20240357e-03,
        1.40074082e-03, 1.70329364e-03, 1.36183552e-03, 9.65976622e-04,
        1.77862821e-03, 1.34311605e-03, 1.27515662e-03, 1.42899575e-03,
        7.89129292e-04, 9.04290704e-04, 5.65862516e-04, 7.11714034e-04,
        1.05709163e-03, 5.54557540e-04, 7.38011906e-04, 4.88379854e-04,
        3.82565690e-04, 4.15254966e-04, 2.89837772e-04, 1.74950037e-04,
        1.38227901e-04, 2.35326166e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.34137246e-04, 2.59546359e-04, 4.08451044e-04, 3.23111715e-04,
        3.77274206e-04, 4.64921526e-04, 7.22224999e-04, 6.65036379e-04,
        1.44646806e-03, 6.47220470e-04, 1.07821031e-03, 8.41765199e-04,
        1.22110266e-03, 1.37866812e-03, 1.20430801e-03, 1.22501515e-03,
        1.78387703e-03, 1.45989179e-03, 1.48925348e-03, 1.90210727e-03,
        1.66192022e-03, 1.68277114e-03, 1.59203087e-03, 1.77281676e-03,
        1.91833265e-03, 1.59111933e-03, 1.19292433e-03, 1.25830714e-03,
        1.00484909e-03, 1.95627287e-03, 1.08537823e-03, 1.51730969e-03,
        2.37974199e-03, 1.88134005e-03, 1.68683077e-03, 1.63254142e-03,
        1.31747383e-03, 1.86250685e-03, 1.40349416e-03, 1.19721505e-03,
        1.13354460e-03, 1.45811064e-03, 9.87907755e-04, 1.65677606e-03,
        9.17633763e-04, 1.00936950e-03, 8.45332746e-04, 8.09435325e-04,
        8.76956095e-04, 9.75276518e-04, 4.54549270e-04, 6.77346659e-04,
        3.97312717e-04, 5.42800000e-04, 2.92446988e-04, 3.90544214e-04,
        2.34945444e-04, 3.41531209e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.52523576e-04, 3.30252107e-04, 3.68202949e-04, 5.46653755e-04,
        4.95874614e-04, 4.79417911e-04, 9.49469511e-04, 7.24626530e-04,
        1.74279779e-03, 8.64146336e-04, 1.27295870e-03, 1.08807557e-03,
        9.11375508e-04, 1.16345461e-03, 9.86703672e-04, 1.02296157e-03,
        1.36418757e-03, 1.18123321e-03, 1.32964191e-03, 1.66485773e-03,
        1.37884868e-03, 1.75696181e-03, 1.54730678e-03, 1.95501838e-03,
        1.78958254e-03, 1.69647159e-03, 1.41463731e-03, 1.57594937e-03,
        1.12475583e-03, 2.79019563e-03, 1.12556061e-03, 1.62814429e-03,
        1.83365506e-03, 2.00557779e-03, 1.50811207e-03, 1.70062424e-03,
        1.42183306e-03, 1.31914171e-03, 1.50734233e-03, 1.07191456e-03,
        1.19691202e-03, 1.29112962e-03, 1.27016346e-03, 1.52324303e-03,
        1.41905702e-03, 1.09924958e-03, 1.29670557e-03, 1.13268697e-03,
        1.32774306e-03, 8.34655773e-04, 6.95364783e-04, 7.30830710e-04,
        5.71634097e-04, 5.54760511e-04, 4.50346037e-04, 5.09611389e-04,
        2.71748431e-04, 3.10219155e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        2.65955110e-04, 4.11931425e-04, 5.96965896e-04, 4.40919161e-04,
        8.46885727e-04, 6.15632220e-04, 9.41532664e-04, 6.74293842e-04,
        8.03396397e-04, 9.19934711e-04, 1.24699902e-03, 8.32270423e-04,
        1.30680483e-03, 8.79532017e-04, 1.96998380e-03, 8.93793069e-04,
        2.32804730e-03, 1.35476515e-03, 2.05461867e-03, 1.75007363e-03,
        2.11651670e-03, 1.61128968e-03, 2.05062935e-03, 2.62990594e-03,
        1.95638766e-03, 2.29414180e-03, 1.58093125e-03, 1.67654036e-03,
        1.41999125e-03, 2.60248454e-03, 1.56897563e-03, 1.49428251e-03,
        2.02215137e-03, 2.06871051e-03, 1.76716538e-03, 1.73190713e-03,
        1.82557048e-03, 1.10265554e-03, 2.26631737e-03, 1.34247961e-03,
        1.76499865e-03, 1.16519991e-03, 1.89628510e-03, 1.20858382e-03,
        2.38253758e-03, 1.32390345e-03, 1.49793539e-03, 1.25410664e-03,
        1.11630023e-03, 9.47813212e-04, 7.88342615e-04, 9.59247351e-04,
        4.82751901e-04, 5.18868852e-04, 3.81207617e-04, 5.10101556e-04,
        4.04640625e-04, 2.58337444e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        5.29067009e-04, 3.27438029e-04, 9.65044717e-04, 3.44652683e-04,
        1.31740107e-03, 5.52578771e-04, 1.16871437e-03, 8.35833547e-04,
        1.00202381e-03, 8.52760451e-04, 1.32275361e-03, 1.31205749e-03,
        2.11332180e-03, 2.34133913e-03, 4.35249880e-03, 1.48885045e-03,
        5.05692791e-03, 1.94152689e-03, 4.73974785e-03, 2.00663786e-03,
        3.68212163e-03, 1.88652158e-03, 2.54951487e-03, 2.90391454e-03,
        2.36071949e-03, 2.01957719e-03, 2.10029352e-03, 1.64491578e-03,
        2.15110648e-03, 2.73870584e-03, 2.46379268e-03, 1.45620899e-03,
        2.57312879e-03, 2.21723132e-03, 2.46983534e-03, 1.95882726e-03,
        2.38571758e-03, 1.48017239e-03, 2.34738481e-03, 1.34192128e-03,
        2.03966466e-03, 1.19568454e-03, 2.37802090e-03, 1.26776739e-03,
        3.06964922e-03, 1.28498999e-03, 1.81876111e-03, 8.15782580e-04,
        1.36167707e-03, 7.78348476e-04, 1.02023070e-03, 6.42041443e-04,
        6.15577912e-04, 5.00192808e-04, 4.55993664e-04, 5.93846780e-04,
        3.78754165e-04, 2.98164057e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        7.30208179e-04, 4.35498136e-04, 1.30928925e-03, 5.03533287e-04,
        2.08250876e-03, 6.36138837e-04, 1.46220671e-03, 7.09154992e-04,
        1.14636880e-03, 1.37479999e-03, 1.25251489e-03, 1.00004044e-03,
        1.90265023e-03, 1.31961389e-03, 3.15803685e-03, 1.74108322e-03,
        3.60944145e-03, 1.50133169e-03, 3.76826175e-03, 1.63527741e-03,
        2.71698949e-03, 2.06482457e-03, 1.74827734e-03, 2.70530744e-03,
        1.81623583e-03, 1.98072195e-03, 1.85689004e-03, 2.40804604e-03,
        1.66626321e-03, 3.33258742e-03, 1.73346663e-03, 2.26076716e-03,
        2.80817808e-03, 2.66387896e-03, 2.88823736e-03, 2.21655658e-03,
        2.68257735e-03, 1.59529829e-03, 2.81648268e-03, 1.35541183e-03,
        1.26438995e-03, 1.59183471e-03, 2.02547363e-03, 1.40573119e-03,
        2.06634402e-03, 1.52994099e-03, 1.86531607e-03, 1.03850476e-03,
        1.08437019e-03, 1.00942748e-03, 9.73228016e-04, 7.33082008e-04,
        9.69872344e-04, 7.06372550e-04, 4.18503449e-04, 5.12657221e-04,
        2.70889665e-04, 3.80139885e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        8.76915583e-04, 3.89280089e-04, 1.68117252e-03, 4.74769040e-04,
        2.55307835e-03, 6.92642352e-04, 2.04488379e-03, 7.99844682e-04,
        9.02731728e-04, 1.33479445e-03, 1.06734177e-03, 9.49823123e-04,
        1.89442292e-03, 1.75261719e-03, 3.60951782e-03, 1.46045547e-03,
        3.50042875e-03, 2.38450966e-03, 3.68855358e-03, 1.82826701e-03,
        3.12519236e-03, 2.25555664e-03, 1.87025988e-03, 2.65327748e-03,
        2.02599890e-03, 2.35363189e-03, 2.30336539e-03, 2.42500985e-03,
        1.90615398e-03, 2.27543944e-03, 1.80612272e-03, 2.00437848e-03,
        1.58445060e-03, 2.30708835e-03, 1.84550148e-03, 2.71843560e-03,
        1.98609498e-03, 1.84105791e-03, 2.10075895e-03, 1.90486095e-03,
        1.45958422e-03, 2.16242555e-03, 1.63844670e-03, 1.53600378e-03,
        2.19058059e-03, 1.43210054e-03, 2.33784108e-03, 1.27152959e-03,
        2.09572259e-03, 1.46291661e-03, 8.57029227e-04, 1.21791521e-03,
        7.59818242e-04, 1.05882564e-03, 4.89842903e-04, 5.14137850e-04,
        2.85649614e-04, 4.13392147e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        8.06511089e-04, 5.28081728e-04, 1.56424008e-03, 4.97739005e-04,
        2.93165282e-03, 1.19887129e-03, 2.08321260e-03, 9.82353813e-04,
        1.37473305e-03, 1.32776389e-03, 1.18962070e-03, 1.03204022e-03,
        1.74667861e-03, 1.72537938e-03, 2.93541094e-03, 1.45627873e-03,
        1.42380700e-03, 2.41315039e-03, 1.74275727e-03, 1.65561354e-03,
        2.62868148e-03, 1.66732958e-03, 4.75463318e-03, 1.66538078e-03,
        2.53066327e-03, 2.20666127e-03, 2.54547922e-03, 1.83246692e-03,
        2.55704229e-03, 2.66679539e-03, 3.06392717e-03, 1.97742181e-03,
        1.79395499e-03, 2.36118399e-03, 1.66871352e-03, 3.73310293e-03,
        2.20931577e-03, 2.18925788e-03, 2.16532615e-03, 2.75885966e-03,
        1.47265568e-03, 3.25125759e-03, 1.59750541e-03, 1.82211481e-03,
        1.82032073e-03, 1.57280941e-03, 1.64892955e-03, 1.27744162e-03,
        1.39040826e-03, 1.49482768e-03, 1.17625762e-03, 1.70588843e-03,
        8.15072330e-04, 1.20035978e-03, 5.22280869e-04, 5.27192140e-04,
        2.82709138e-04, 3.76723503e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        9.31839400e-04, 6.01427455e-04, 1.45781098e-03, 7.01347191e-04,
        1.84940384e-03, 2.05833977e-03, 1.07004866e-03, 1.90682476e-03,
        1.03817286e-03, 1.86623284e-03, 1.42887677e-03, 1.91109884e-03,
        1.44596840e-03, 1.33686501e-03, 1.68111525e-03, 1.32062251e-03,
        2.48299679e-03, 2.71339854e-03, 2.59890035e-03, 2.38256669e-03,
        2.98827770e-03, 2.41923286e-03, 4.24497901e-03, 1.73160539e-03,
        2.58369395e-03, 1.90302357e-03, 3.05116922e-03, 1.72287389e-03,
        2.89871683e-03, 2.44097249e-03, 3.13341967e-03, 2.42021238e-03,
        2.36254046e-03, 2.36072880e-03, 2.30828556e-03, 4.68038581e-03,
        2.53150682e-03, 4.26751934e-03, 2.34560180e-03, 3.71266552e-03,
        2.41924450e-03, 4.41082194e-03, 2.25850334e-03, 2.31647375e-03,
        1.73268514e-03, 1.54919841e-03, 1.79628353e-03, 1.72696367e-03,
        1.33824989e-03, 1.06083276e-03, 1.12294231e-03, 9.26308450e-04,
        6.68310386e-04, 7.68342928e-04, 5.63404814e-04, 4.26072656e-04,
        3.26084875e-04, 3.59864091e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.09634141e-03, 6.96279807e-04, 1.52841769e-03, 9.04389133e-04,
        1.43579836e-03, 1.87038037e-03, 1.25648582e-03, 2.35537812e-03,
        1.97330164e-03, 2.55160360e-03, 1.78116455e-03, 1.47690077e-03,
        2.40429514e-03, 1.40049949e-03, 4.02497547e-03, 2.13751104e-03,
        3.11530707e-03, 4.35727835e-03, 3.70012573e-03, 3.89953307e-03,
        3.42032267e-03, 2.87086866e-03, 3.76734789e-03, 2.26982427e-03,
        3.43718869e-03, 1.96288945e-03, 2.96981400e-03, 2.02302681e-03,
        2.82936613e-03, 2.62284558e-03, 2.83304648e-03, 2.18346133e-03,
        2.50181789e-03, 1.93024578e-03, 1.86115736e-03, 3.09269805e-03,
        1.69260975e-03, 3.81447701e-03, 2.30565760e-03, 4.14014561e-03,
        1.87474745e-03, 3.90211423e-03, 1.65887864e-03, 2.88872048e-03,
        1.41869532e-03, 1.52343535e-03, 1.18493033e-03, 1.61659683e-03,
        1.04000606e-03, 1.04725896e-03, 8.27612297e-04, 1.15644746e-03,
        8.39373213e-04, 9.04315326e-04, 5.12863451e-04, 6.82213984e-04,
        3.61563987e-04, 5.87634800e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        7.10042950e-04, 4.30676184e-04, 8.34449078e-04, 6.53029128e-04,
        1.03231252e-03, 1.37999072e-03, 1.73960708e-03, 1.82476675e-03,
        1.46518194e-03, 2.84317601e-03, 1.19820598e-03, 1.84659916e-03,
        2.29099859e-03, 1.29666144e-03, 3.57720302e-03, 2.40226695e-03,
        2.62459624e-03, 3.13049066e-03, 3.84613150e-03, 3.15484521e-03,
        3.22803622e-03, 2.73801573e-03, 2.53288564e-03, 1.98558345e-03,
        3.04694730e-03, 2.62524979e-03, 2.72569666e-03, 3.32834595e-03,
        3.20574897e-03, 3.18141421e-03, 2.77830474e-03, 2.56185257e-03,
        2.92245392e-03, 2.20616022e-03, 2.82093626e-03, 2.40284717e-03,
        1.96346408e-03, 2.85965553e-03, 2.18575122e-03, 4.20077564e-03,
        2.10326794e-03, 2.73412489e-03, 1.75405852e-03, 2.78923474e-03,
        1.80658838e-03, 1.66832365e-03, 1.30173948e-03, 1.84069690e-03,
        9.86749190e-04, 1.54065853e-03, 9.98459640e-04, 9.78558557e-04,
        9.58705728e-04, 6.29174465e-04, 5.42458089e-04, 5.05142496e-04,
        3.35685909e-04, 4.44021309e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        7.36739836e-04, 5.17008826e-04, 7.01380312e-04, 1.03814772e-03,
        1.15708436e-03, 1.44489727e-03, 1.01855560e-03, 1.87510042e-03,
        1.64737413e-03, 2.74153077e-03, 1.02447125e-03, 1.17050274e-03,
        4.20964183e-03, 1.40711630e-03, 4.96082101e-03, 2.78298627e-03,
        4.60053701e-03, 1.98435294e-03, 5.81271807e-03, 2.05866946e-03,
        3.88954277e-03, 2.10782210e-03, 2.45107291e-03, 2.14583357e-03,
        3.08932853e-03, 2.79980060e-03, 3.15243611e-03, 2.21966486e-03,
        3.73716676e-03, 2.35027983e-03, 2.81913509e-03, 2.07392615e-03,
        2.40518432e-03, 1.73076754e-03, 3.16819013e-03, 2.18907790e-03,
        1.83926593e-03, 2.11636000e-03, 3.01551516e-03, 2.30785296e-03,
        3.16692237e-03, 1.84962084e-03, 1.92976953e-03, 1.95794622e-03,
        1.85021618e-03, 1.23132567e-03, 1.29646447e-03, 1.48667896e-03,
        9.73581744e-04, 1.28916255e-03, 8.54648009e-04, 9.65696643e-04,
        7.97372137e-04, 7.60146999e-04, 4.26042097e-04, 5.06943790e-04,
        3.37385194e-04, 4.70719184e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        4.40191798e-04, 6.70160749e-04, 6.75601070e-04, 4.79611801e-04,
        7.54313136e-04, 6.81887905e-04, 1.12598063e-03, 8.27761600e-04,
        2.54651764e-03, 1.52423850e-03, 1.65213493e-03, 1.14539266e-03,
        3.48295923e-03, 1.61768764e-03, 3.91231989e-03, 2.41079973e-03,
        3.91791062e-03, 1.90254312e-03, 4.84258775e-03, 1.98091101e-03,
        3.47561971e-03, 2.44945241e-03, 2.96801538e-03, 1.99161866e-03,
        4.44400497e-03, 2.99984985e-03, 4.56757098e-03, 2.06429069e-03,
        5.03818411e-03, 1.83729001e-03, 4.12873318e-03, 2.23760610e-03,
        3.91219743e-03, 1.82307174e-03, 5.39107015e-03, 2.40692100e-03,
        4.84062778e-03, 2.00552726e-03, 5.37405815e-03, 2.57181795e-03,
        4.82999207e-03, 1.78277516e-03, 2.73185573e-03, 2.43848306e-03,
        2.35301908e-03, 1.55673001e-03, 1.61880488e-03, 1.75741198e-03,
        1.40262535e-03, 1.21003063e-03, 1.22082257e-03, 8.47642659e-04,
        9.11166542e-04, 9.18837497e-04, 6.96047675e-04, 5.55495615e-04,
        4.12523921e-04, 3.88999615e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        6.45647291e-04, 7.19025440e-04, 9.05541761e-04, 5.70190256e-04,
        9.49868234e-04, 6.77449163e-04, 1.22897048e-03, 9.39140969e-04,
        1.58593478e-03, 9.86500760e-04, 1.45207648e-03, 1.42139068e-03,
        3.01058171e-03, 2.50773947e-03, 3.02669499e-03, 2.89259548e-03,
        2.50427192e-03, 2.00630049e-03, 2.38386146e-03, 1.99984899e-03,
        1.93126383e-03, 2.12918851e-03, 2.74710124e-03, 2.72001675e-03,
        3.40819499e-03, 2.01183907e-03, 3.17340298e-03, 2.54505686e-03,
        4.35054256e-03, 2.33946368e-03, 2.09345017e-03, 3.25585180e-03,
        2.20048730e-03, 2.47114873e-03, 2.24835495e-03, 2.28018640e-03,
        1.89516239e-03, 2.65590614e-03, 2.62364815e-03, 2.53747310e-03,
        2.62398133e-03, 2.25573033e-03, 1.83319813e-03, 2.38885428e-03,
        1.53540634e-03, 1.69082067e-03, 1.25649059e-03, 1.82578084e-03,
        1.20467960e-03, 1.29849243e-03, 1.19177683e-03, 9.74968541e-04,
        7.99751782e-04, 9.23969666e-04, 6.35976961e-04, 7.45665864e-04,
        4.82601521e-04, 5.00501716e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        4.85410827e-04, 8.87201284e-04, 7.65417120e-04, 4.88018588e-04,
        1.32861163e-03, 5.97901060e-04, 1.61250320e-03, 1.03527203e-03,
        1.08583178e-03, 1.43721281e-03, 1.49615016e-03, 1.50709297e-03,
        1.70066254e-03, 1.83588732e-03, 1.89226377e-03, 1.97403342e-03,
        1.91780936e-03, 1.91398489e-03, 2.42149271e-03, 1.96998171e-03,
        2.66625686e-03, 2.89019244e-03, 4.33130004e-03, 2.02690787e-03,
        3.74772446e-03, 2.33828858e-03, 3.88418534e-03, 2.44539627e-03,
        4.13765712e-03, 2.26775673e-03, 2.06712261e-03, 3.08276666e-03,
        1.65409653e-03, 2.79473956e-03, 1.83149998e-03, 2.74398597e-03,
        1.98181602e-03, 2.73576449e-03, 2.38518254e-03, 2.90926616e-03,
        2.03275611e-03, 2.96103838e-03, 1.56833243e-03, 2.03965884e-03,
        1.28997699e-03, 1.58424571e-03, 1.44917879e-03, 1.64312299e-03,
        8.21396825e-04, 1.32622698e-03, 1.29059178e-03, 1.22219115e-03,
        9.47735505e-04, 8.24163319e-04, 7.34207628e-04, 6.58651174e-04,
        3.96717340e-04, 4.19800839e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        7.15535891e-04, 1.18714210e-03, 7.91252533e-04, 1.01830531e-03,
        1.27523718e-03, 1.17877056e-03, 1.49259833e-03, 1.82265486e-03,
        1.26272452e-03, 1.68142363e-03, 1.77534157e-03, 1.22232328e-03,
        2.17235647e-03, 1.52377202e-03, 2.44458555e-03, 1.92997232e-03,
        2.08349107e-03, 1.67343253e-03, 3.42889084e-03, 1.53323472e-03,
        1.90992607e-03, 3.79479863e-03, 2.53769499e-03, 3.18789785e-03,
        1.95931131e-03, 2.81153969e-03, 2.42609414e-03, 3.36529571e-03,
        2.92687956e-03, 2.08680611e-03, 2.23075273e-03, 3.01395892e-03,
        2.07933015e-03, 2.49213586e-03, 1.94607896e-03, 2.43587443e-03,
        1.73156441e-03, 2.64397985e-03, 1.73253892e-03, 2.59714550e-03,
        1.52737391e-03, 2.61172652e-03, 1.86475611e-03, 1.72459160e-03,
        1.36473263e-03, 1.35195081e-03, 1.43820490e-03, 1.88172015e-03,
        7.15922448e-04, 1.34941936e-03, 8.81109736e-04, 1.18248863e-03,
        7.18981551e-04, 9.24230379e-04, 4.08349151e-04, 6.59349782e-04,
        3.15602054e-04, 2.81790039e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        4.53079236e-04, 8.34495935e-04, 7.20682321e-04, 5.33109589e-04,
        1.09231297e-03, 6.87211461e-04, 1.28485064e-03, 8.04760144e-04,
        1.54134294e-03, 1.24780240e-03, 2.21423293e-03, 1.59396231e-03,
        2.57016323e-03, 1.25136808e-03, 2.67406343e-03, 1.38000038e-03,
        1.98746123e-03, 1.66079367e-03, 2.96775904e-03, 1.52834191e-03,
        1.88846176e-03, 2.93608126e-03, 2.28054333e-03, 2.63004284e-03,
        2.46448885e-03, 2.02417537e-03, 2.43426929e-03, 2.93783774e-03,
        2.04121019e-03, 2.63198046e-03, 3.36991553e-03, 2.56405841e-03,
        3.31568602e-03, 2.24604248e-03, 3.88121558e-03, 2.72712251e-03,
        2.08568457e-03, 2.72331689e-03, 1.52632257e-03, 1.78671291e-03,
        1.73396175e-03, 1.62585126e-03, 1.56277325e-03, 2.65177409e-03,
        1.24692579e-03, 1.79721415e-03, 1.19176449e-03, 1.77005248e-03,
        1.04485394e-03, 1.01288385e-03, 9.22750216e-04, 7.19698262e-04,
        6.75834133e-04, 6.60982274e-04, 4.89418686e-04, 3.45353474e-04,
        2.50551791e-04, 2.72447185e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        4.02718346e-04, 8.67082679e-04, 6.36392098e-04, 8.43859452e-04,
        1.04443752e-03, 9.33366187e-04, 1.24755770e-03, 8.08270299e-04,
        1.92969886e-03, 1.24329212e-03, 3.01029417e-03, 1.07698585e-03,
        3.80595680e-03, 1.75139483e-03, 3.75448307e-03, 1.71275972e-03,
        1.85057044e-03, 1.37439999e-03, 1.86474144e-03, 1.75608671e-03,
        2.20171083e-03, 3.19102663e-03, 2.07757787e-03, 4.05447278e-03,
        1.63637416e-03, 3.40438634e-03, 1.92596903e-03, 4.48998390e-03,
        2.32758769e-03, 3.92341148e-03, 2.10399646e-03, 3.68758873e-03,
        2.51606014e-03, 3.95467272e-03, 2.96325563e-03, 3.62719665e-03,
        1.39784522e-03, 2.42779893e-03, 1.97297591e-03, 1.83826813e-03,
        1.71689328e-03, 1.60190999e-03, 1.63957861e-03, 2.49696244e-03,
        1.23711664e-03, 1.66275294e-03, 8.43697460e-04, 1.59701961e-03,
        9.02722124e-04, 8.59008229e-04, 7.79777649e-04, 6.21564104e-04,
        8.29170458e-04, 6.85289211e-04, 4.69341176e-04, 4.64089389e-04,
        2.16763685e-04, 3.82817234e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        4.26045881e-04, 3.72100971e-04, 5.01575938e-04, 3.54751362e-04,
        1.04937016e-03, 5.47438685e-04, 1.10593322e-03, 8.46183684e-04,
        1.32310204e-03, 1.17828918e-03, 1.94574019e-03, 9.39576945e-04,
        1.79052306e-03, 1.14328437e-03, 2.16499157e-03, 1.48343504e-03,
        1.33792812e-03, 1.82550785e-03, 1.77078438e-03, 1.67667971e-03,
        2.89981579e-03, 2.26021348e-03, 3.47394124e-03, 3.41095310e-03,
        2.11332110e-03, 2.93577346e-03, 2.14569154e-03, 4.52345749e-03,
        2.73307180e-03, 4.45325393e-03, 2.05693580e-03, 4.08397987e-03,
        3.04185273e-03, 4.78630653e-03, 2.79125571e-03, 3.94374132e-03,
        1.17588823e-03, 2.39415537e-03, 1.43528613e-03, 1.93328212e-03,
        1.66631292e-03, 1.43963296e-03, 1.74669141e-03, 2.41644471e-03,
        1.88214902e-03, 1.72509730e-03, 1.46856636e-03, 1.86591281e-03,
        1.23272080e-03, 1.36000325e-03, 1.23212615e-03, 1.05411652e-03,
        1.02883042e-03, 4.75428067e-04, 6.13137323e-04, 3.90744593e-04,
        2.45072471e-04, 2.60573288e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        5.75543614e-04, 3.34662182e-04, 4.79884358e-04, 3.02038068e-04,
        8.53278849e-04, 1.01337617e-03, 9.38582234e-04, 1.34816929e-03,
        1.10551214e-03, 1.32799335e-03, 1.52588543e-03, 7.19702279e-04,
        1.09615852e-03, 1.86907733e-03, 1.49504829e-03, 3.04308347e-03,
        1.08890061e-03, 2.79639405e-03, 1.78090471e-03, 2.51661823e-03,
        1.89634529e-03, 1.57691818e-03, 2.10034498e-03, 2.07547145e-03,
        1.61211251e-03, 2.09641107e-03, 2.32767453e-03, 2.12996616e-03,
        1.77452620e-03, 2.72748899e-03, 1.30002701e-03, 2.53736996e-03,
        1.71287474e-03, 4.11557965e-03, 2.18091765e-03, 3.72322253e-03,
        1.48947467e-03, 1.81400601e-03, 1.57589884e-03, 1.38922955e-03,
        1.77815033e-03, 1.50143844e-03, 1.61445164e-03, 2.47846707e-03,
        1.48000580e-03, 1.63530465e-03, 9.19096812e-04, 1.43405900e-03,
        5.70546079e-04, 1.20567274e-03, 6.00712083e-04, 9.91539331e-04,
        6.32984098e-04, 5.81405940e-04, 6.09657145e-04, 5.00066555e-04,
        2.15908600e-04, 2.07640449e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        4.00082616e-04, 3.15083395e-04, 5.29795594e-04, 2.46066134e-04,
        6.72127586e-04, 6.70079491e-04, 9.15808487e-04, 5.85032045e-04,
        8.84736422e-04, 8.63272697e-04, 8.50090408e-04, 9.03790293e-04,
        1.51782937e-03, 5.87166345e-04, 1.46345911e-03, 9.77964723e-04,
        1.19094097e-03, 1.11330557e-03, 1.84610498e-03, 1.71232037e-03,
        1.68824126e-03, 1.11483166e-03, 2.40446324e-03, 1.73848867e-03,
        2.20033573e-03, 1.63102755e-03, 1.68226776e-03, 1.66273804e-03,
        1.55431323e-03, 2.11919355e-03, 1.88040093e-03, 1.81397854e-03,
        1.47079141e-03, 3.25144408e-03, 1.15424278e-03, 2.38525122e-03,
        1.48901343e-03, 1.77523890e-03, 1.49396085e-03, 1.34182279e-03,
        2.28743791e-03, 9.51333088e-04, 1.74456218e-03, 1.11702841e-03,
        1.29689521e-03, 7.04049948e-04, 8.24616000e-04, 7.82605086e-04,
        8.80176900e-04, 7.75584544e-04, 7.88060832e-04, 4.51328902e-04,
        4.78287286e-04, 3.37166130e-04, 3.40445462e-04, 3.68918467e-04,
        2.30307473e-04, 1.97169982e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        3.51028139e-04, 3.36803292e-04, 5.00819297e-04, 2.19325346e-04,
        4.94852429e-04, 7.42807810e-04, 8.25737312e-04, 4.14860348e-04,
        9.20193037e-04, 6.98378019e-04, 1.02625915e-03, 9.78911528e-04,
        2.22937344e-03, 1.37651863e-03, 1.97421177e-03, 1.87793432e-03,
        1.00759568e-03, 1.27715128e-03, 1.61687995e-03, 1.70618505e-03,
        9.32303839e-04, 1.16093468e-03, 1.48766267e-03, 1.21845037e-03,
        1.56297593e-03, 1.35464605e-03, 1.76875025e-03, 1.77069067e-03,
        1.59602589e-03, 1.81289157e-03, 2.14151572e-03, 2.00567581e-03,
        1.28421432e-03, 2.23403936e-03, 1.62531843e-03, 2.20845826e-03,
        1.85327046e-03, 1.70903257e-03, 1.51096226e-03, 1.26822712e-03,
        2.05493113e-03, 1.25014910e-03, 1.40347343e-03, 1.15407235e-03,
        9.26229346e-04, 7.70102197e-04, 6.97273237e-04, 7.32101384e-04,
        8.64228059e-04, 5.19790803e-04, 9.42352519e-04, 3.50895076e-04,
        7.26969447e-04, 2.70346791e-04, 2.64127360e-04, 2.74784135e-04,
        3.17833212e-04, 1.39782773e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.80319505e-04, 1.70471511e-04, 3.42830754e-04, 2.31029248e-04,
        4.10658016e-04, 3.61989398e-04, 5.98556479e-04, 6.48361165e-04,
        8.39351676e-04, 7.96477078e-04, 1.18650263e-03, 1.10261980e-03,
        1.93818205e-03, 8.69352254e-04, 2.14932021e-03, 1.01624674e-03,
        1.36972975e-03, 8.17580789e-04, 1.25816767e-03, 1.06605573e-03,
        9.43292514e-04, 1.15301460e-03, 1.29694340e-03, 9.39791498e-04,
        1.11094897e-03, 9.95770330e-04, 1.56868435e-03, 1.27752998e-03,
        1.32211752e-03, 1.17005524e-03, 1.52030739e-03, 1.34012976e-03,
        1.11566612e-03, 1.13306649e-03, 1.02483097e-03, 1.26446132e-03,
        8.98372207e-04, 1.25924184e-03, 8.99066159e-04, 1.48393027e-03,
        1.32741372e-03, 9.58968129e-04, 1.07812916e-03, 1.65732240e-03,
        6.62090839e-04, 7.20014330e-04, 1.01662695e-03, 8.66178947e-04,
        1.16473576e-03, 5.85402304e-04, 1.03131775e-03, 4.58800641e-04,
        7.13630638e-04, 3.00751068e-04, 2.60759174e-04, 2.53022474e-04,
        2.27185170e-04, 1.47517334e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        1.31951005e-04, 2.08926896e-04, 2.18534828e-04, 2.40106179e-04,
        3.01401480e-04, 2.92861165e-04, 5.38981054e-04, 5.34373219e-04,
        5.48018725e-04, 5.52041223e-04, 7.37032853e-04, 5.43597329e-04,
        1.35983352e-03, 8.25207680e-04, 1.25999982e-03, 7.96822424e-04,
        1.01304473e-03, 7.57277419e-04, 1.00288261e-03, 6.43401232e-04,
        7.19792151e-04, 9.70660069e-04, 7.44202640e-04, 8.55176942e-04,
        8.57551582e-04, 9.77131887e-04, 1.13551924e-03, 1.19962147e-03,
        1.11107144e-03, 8.04916723e-04, 8.63322872e-04, 1.07399956e-03,
        9.86664905e-04, 7.78737362e-04, 1.09329238e-03, 8.90896597e-04,
        6.06488320e-04, 7.48748251e-04, 6.65239233e-04, 8.31676647e-04,
        8.83135130e-04, 4.23915772e-04, 8.56355007e-04, 7.30604224e-04,
        4.46728169e-04, 4.57541784e-04, 3.49690381e-04, 4.97100642e-04,
        3.10848234e-04, 4.01074416e-04, 2.56101281e-04, 3.20421415e-04,
        1.78967646e-04, 1.86745703e-04, 1.36399700e-04, 1.79945579e-04,
        1.05819287e-04, 1.08884371e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        7.03477926e-05, 2.25620795e-04, 1.15143434e-04, 2.56176398e-04,
        2.01957286e-04, 2.73191865e-04, 3.14072531e-04, 3.46117653e-04,
        4.19737451e-04, 3.81244346e-04, 4.08102758e-04, 5.65330905e-04,
        7.58177892e-04, 4.30216372e-04, 7.09906279e-04, 5.16088912e-04,
        5.78072912e-04, 4.45936545e-04, 6.99628145e-04, 4.88816062e-04,
        7.57976261e-04, 9.34004900e-04, 6.23976171e-04, 7.25137012e-04,
        5.96163212e-04, 9.05494031e-04, 7.90732680e-04, 1.13470259e-03,
        5.94514655e-04, 6.16373844e-04, 7.66079698e-04, 6.83959166e-04,
        6.93426875e-04, 4.68968501e-04, 8.71337601e-04, 7.67328020e-04,
        6.22588210e-04, 5.62750152e-04, 8.05414689e-04, 5.54996193e-04,
        4.47431434e-04, 2.97350489e-04, 4.86931647e-04, 5.65084629e-04,
        4.17403440e-04, 3.18059028e-04, 5.54968312e-04, 4.97072237e-04,
        4.81245370e-04, 2.62792688e-04, 3.65950225e-04, 2.74084770e-04,
        2.10417580e-04, 2.00473587e-04, 5.28929450e-05, 1.64964251e-04,
        6.67446366e-05, 1.02727870e-04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        5.70117045e-05, 1.03350256e-04, 1.07669941e-04, 1.04570885e-04,
        1.31298220e-04, 1.93133412e-04, 1.47152823e-04, 2.91326054e-04,
        3.45337699e-04, 2.41538204e-04, 3.35136778e-04, 4.10650828e-04,
        4.94182983e-04, 3.43899010e-04, 4.98403679e-04, 3.04779736e-04,
        5.68271556e-04, 2.95756181e-04, 6.89258974e-04, 3.49702488e-04,
        8.20084533e-04, 5.07308170e-04, 6.75043208e-04, 5.12901170e-04,
        5.10557264e-04, 5.58154075e-04, 5.46442752e-04, 5.09259582e-04,
        6.18587306e-04, 3.96683201e-04, 4.60060983e-04, 4.55495028e-04,
        4.71635489e-04, 2.56436731e-04, 6.62238570e-04, 3.68845445e-04,
        4.85317723e-04, 4.33395413e-04, 8.60642816e-04, 4.66410274e-04,
        6.32412906e-04, 3.38434125e-04, 6.99045544e-04, 6.57567638e-04,
        6.56408723e-04, 3.12202639e-04, 4.71025967e-04, 5.41630201e-04,
        4.16982512e-04, 3.15680489e-04, 1.75798617e-04, 2.35976491e-04,
        1.48221472e-04, 1.82358301e-04, 7.64746001e-05, 1.06869047e-04,
        6.91597670e-05, 7.17689254e-05, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        3.41483504e-02, 3.11599411e-02, 2.64214035e-02, 3.29295099e-02,
        3.00031435e-02, 5.59781082e-02, 3.98878343e-02, 3.65305133e-02,
        3.21196355e-02, 4.47427556e-02, 2.93503553e-02, 3.04236356e-02,
        3.47419120e-02, 3.83987688e-02, 2.96322089e-02, 2.43512746e-02,
        2.38872543e-02, 3.66208851e-02, 3.85731086e-02, 2.47652400e-02,
        2.59991586e-02, 5.22401594e-02, 2.23561972e-02, 3.47999632e-02,
        2.90254131e-02, 3.13612930e-02, 2.38441452e-02, 7.19688907e-02,
        4.38614376e-02, 2.82672215e-02, 3.73858884e-02, 4.15736549e-02,
        3.51581722e-02, 3.45546752e-02, 4.00598645e-02, 4.06390689e-02,
        2.67299376e-02, 2.20765918e-02, 2.84422897e-02, 2.34373305e-02,
        5.46360575e-02, 3.21862660e-02, 2.60559972e-02, 3.38777378e-02,
        2.43584923e-02, 2.90449150e-02, 3.26094627e-02, 3.02545819e-02,
        2.85975970e-02, 3.15095708e-02, 3.00809238e-02, 2.94390600e-02,
        2.91215889e-02, 2.66443305e-02, 3.91333774e-02, 5.77704012e-02,
        4.66911308e-02, 2.92817652e-02, 3.01466193e-02, 2.64635980e-02,
        3.45760211e-02, 3.22783589e-02, 2.60318965e-02, 2.89197769e-02,
        2.56062895e-02, 2.69655101e-02, 2.73374692e-02, 2.47808732e-02,
        2.35279743e-02, 2.21402291e-02, 2.33577993e-02, 4.02690955e-02,
        2.62104999e-02, 4.53671664e-02, 6.00081421e-02, 3.96538936e-02,
        2.17345990e-02, 2.72714328e-02, 2.52770800e-02, 2.96314619e-02,
        3.98503765e-02, 4.07220274e-02, 2.67186165e-02, 4.37301435e-02,
        2.92107519e-02, 2.74733249e-02, 2.75923572e-02, 5.15994206e-02,
        3.08714863e-02, 3.05362642e-02, 3.18307057e-02, 2.43409462e-02,
        4.23508808e-02, 3.74625847e-02, 3.67398635e-02, 2.95996647e-02,
        3.88141386e-02, 2.44268849e-02, 2.40655039e-02, 2.94032581e-02,
        2.45902278e-02, 4.68546525e-02, 2.51290221e-02, 3.11869588e-02,
        2.30793264e-02, 2.87339799e-02, 4.06194255e-02]

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

