import numpy as np
import matplotlib.pyplot as plt

np.random.seed(56)

def gen_Y(pi_0):
	# H_0 : Y ~ [N(3, 1) N(-1, 1)]
	# H_1 : Y_1, Y_2 ~ P_x; p(x) = 1/2*exp(-|x|)
	Y = []
	N = int(10e6)
	ber_exp = np.random.binomial(n=1, p=pi_0, size=N)
	for i in ber_exp:
		if i == 0:
			Y.append([np.random.normal(3, 1), np.random.normal(-1, 1)])
		else:
			Y.append([px(), px()])

	return Y
	
def px(instances=1):
	X = np.random.uniform(size=(instances,1))
	Y = np.random.uniform(size=(instances,1))
	return np.log(X/Y)

# Y = gen_h1(int(10000))
Y = gen_Y(0.5)
plt.hist(Y)
plt.show()