#################
# Code by Uday Kiran and Ayonija
import numpy as np
import matplotlib.pyplot as plt


def px(instances=1):
    ##########################
    # px() is for generating Laplace random variable with pdf
    # p(x) = 1/2*exp(-|x|)
    # instances is predefined parameter it can be varied to get more random variables
    # Returns laplace random variables of size (instances, )
    ##########################
    return np.random.laplace(size=instances)

def h0():
    ##########################
    # h0() is for generating H0 hypothesis
    # Returns (2, ) array containing [N(3, 1) N(-1, 1)]
    ##########################
    return np.array([np.random.normal(3, 1), np.random.normal(-1, 1)]).reshape(2, )

def h1():
    ##########################
    # h1() is for generating H1 hypothesis
    # Returns (2, ) array containing [px(), px()]
    ##########################
    return np.array([px(), px()]).reshape(2, )

def gen_Y(pi_0):
    ##########################
    # gen_Y(pi_o) is for generating Y with prior pi_0 and 1-pi_0
    # Returns Y(shape=(2, N=10^6)), ber_exp(which is random array which generates Ber(1-pi_0))

    # If the generated Ber(1-pi_0) is 1 then we choose H1 else H0, This is how we choose hypothesis
    # N=10^6 instances are generated like this
	# H_0 : Y ~ [N(3, 1) N(-1, 1)]
	# H_1 : Y_1, Y_2 ~ P_x; p(x) = 1/2*exp(-|x|)
    ##########################
    N = int(10e6)
    Y =[]
    ber_exp = np.random.binomial(n=1, p=1-pi_0, size=N)
    for i in ber_exp:
        if i == 0:
            Y.append(h0())
        else:   
            Y.append(h1())
    Y = np.matrix(Y).T
    return Y, ber_exp

def opt_decision_rule(pi_0, Y):
    ##########################
    # opt_decision_rule(pi_0, Y) is function gives bayes decision rule with y and pior pi_0
    # Returns decision(shape=(N=10^6, ))

    # Here for a given y if we get p1(y)/p0(y) >= pi_0/1-pi_0 we choose it to belong to Gamma_1 else Gamma_0
    # The equation looks differnt because we vectorised the code inorder to minimize the time taken for the simulation and to optimize it.
    # L(Y) = (pi/2)*e^(0.5*(|Y-[3; -1]|^2) - |Y|_L1)
    # here, |Y| is L1 norm
    ##########################
    tau = pi_0/(1 - pi_0)
    a = np.array([3, -1]).reshape(2, 1)
    decision = (np.pi/2)*np.exp(0.5*(np.linalg.norm(Y - a, axis=0)**2) - np.linalg.norm(Y, axis=0, ord=1)) >= tau
    return decision.reshape(-1, )

def bayes_risk(genrated_hypothesis, decisions):
    ##########################
    # bayes_risk(genrated_hypothesis, decisions) is measure of P[True hypothesis != Decided hypothesis]
    # Returns V(pi_0) = P[True hypothesis != Decided hypothesis] which is a scalar

    # Code is vectorised here also
    ##########################
    miss_classified = np.linalg.norm(genrated_hypothesis - decisions, ord=1)
    return miss_classified/genrated_hypothesis.shape[0]


# Generating required pi_0 = [0.1, 0.2, 0.3, ....., 0.9]
pi_0 = np.linspace(0.1, 0.9, 9)

# Empty array V_pi0 to store its results
V_pi0 = []

#iteration over all pi_0's to obtain V(pi_0)
for i in pi_0: 
    Y, genrated_hypothesis = gen_Y(i)
    bayes_decisionrule = opt_decision_rule(i, Y)
    V_pi0.append(bayes_risk(genrated_hypothesis, bayes_decisionrule))

# Plotting the V(pi_0) vs pi_0
plt.plot(pi_0, V_pi0)
plt.grid()
plt.xlabel('$\pi_0$')
plt.ylabel('$V(\pi_0)$')
plt.savefig('Plot_final.pdf')
plt.show()