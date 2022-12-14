'''
Author: Hugh Shanno
Collaborated with Charlie Roslansky
'''

import numpy
import random

import qConstants as qc
import qUtilities as qu
import qGates as qg

#QMeasurement
def measurement(state):
    rand = random.uniform(0,1)
    if rand < state[0]*numpy.conj(state[0]):
        return qc.ket0
    else:
        return qc.ket1

def first(state):
    '''Assumes n >= 1. Given an n-qbit state, measures the first qbit.
    Returns a pair (tuple of two items) consisting of a classical one-qbit state (either |0> or |1>)
    and an (n - 1)-qbit state.'''
    ketOmega0 = numpy.zeros((1,int(len(state)/2)), dtype=numpy.array(0+0j).dtype)[0]
    ketOmega1 = numpy.zeros((1,int(len(state)/2)), dtype=numpy.array(0+0j).dtype)[0]
    sigma0 = 0
    for i in range(int(len(state)/2)):
        sigma0 += state[i]*numpy.conj(state[i])
        ketOmega0[i] = state[i]
    sigma0 = sigma0**(1/2)
    sigma1 = 0
    for i in range(int(len(state)/2),int(len(state))):
        sigma1 += state[i]*numpy.conj(state[i])
        ketOmega1[i-int(len(state)/2)] = state[i]
    sigma1 = sigma1**(1/2)
    
    
    if sigma0 != 0:
        ketOmega0 = ketOmega0/sigma0
    if sigma1 != 0:
        ketOmega1 = ketOmega1/sigma1
    
    ketSigma = numpy.array([sigma0, sigma1])
    ketSigma = measurement(ketSigma)
    
    if numpy.array_equal(ketSigma, qc.ket0):
        return (qc.ket0, ketOmega0)
    else:
        return (qc.ket1, ketOmega1)

def last(state):
    '''Assumes n >= 1. Given an n-qbit state, measures the last qbit.
    Returns a pair consisting of an (n - 1)-qbit state and a classical 1-qbit state (either |0> or |1>).'''
    ketOmega0 = numpy.zeros((1,int(len(state)/2)), dtype=numpy.array(0+0j).dtype)[0]
    ketOmega1 = numpy.zeros((1,int(len(state)/2)), dtype=numpy.array(0+0j).dtype)[0]
    sigma0 = 0
    sigma1 = 0
    for i in range(0,int(len(state))):
        if i%2 == 0:
            ketOmega0[int(i/2)] = state[i]
            sigma0 += state[i]*numpy.conj(state[i])
        else:
            ketOmega1[int((i-1)/2)] = state[i]
            sigma1 += state[i]*numpy.conj(state[i])
    sigma0 = sigma0**(1/2)
    sigma1 = sigma1**(1/2)
    
    
    if sigma0 != 0:
        ketOmega0 = ketOmega0/sigma0
    if sigma1 != 0:
        ketOmega1 = ketOmega1/sigma1
    
    ketSigma = numpy.array([sigma0, sigma1])
    ketSigma = measurement(ketSigma)
    
    if numpy.array_equal(ketSigma, qc.ket0):
        return (ketOmega0, qc.ket0)
    else:
        return (ketOmega1, qc.ket1)



### DEFINING SOME TESTS ###

def firstTest(n):
    # Assumes n >= 1. Constructs an unentangled (n + 1)-qbit state |0> |psi> or |1> |psi>,
    # measures the first qbit, and then reconstructs the state.
    ketPsi = qu.uniform(n)
    state = qg.tensor(qc.ket0, ketPsi)
    meas = first(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed firstTest first part")
    else:
        print("failed firstTest first part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))
    ketPsi = qu.uniform(n)
    state = qg.tensor(qc.ket1, ketPsi)
    meas = first(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed firstTest second part")
    else:
        print("failed firstTest second part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))

def firstTest345(n, m):
    # Assumes n >= 1. n + 1 is the total number of qbits. m is how many tests to run. 
    # Should return a number close to 0.64 --- at least for large m.
    psi0 = 3 / 5
    ketChi = qu.uniform(n)
    psi1 = 4 / 5
    ketPhi = qu.uniform(n)
    ketOmega = psi0 * qg.tensor(qc.ket0, ketChi) + psi1 * qg.tensor(qc.ket1, ketPhi)
    def f():
        if (first(ketOmega)[0] == qc.ket0).all():
            return 0
        else:
            return 1
    acc = 0
    for i in range(m):
        acc += f()
    print("check firstTest345 for frequency near 0.64")
    print("    frequency = ", str(acc / m))

def lastTest(n):
    # Assumes n >= 1. Constructs an unentangled (n + 1)-qbit state |psi> |0> or |psi> |1>,
    # measures the last qbit, and then reconstructs the state.
    psi = qu.uniform(n)
    state = qg.tensor(psi, qc.ket0)
    meas = last(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed lastTest first part")
    else:
        print("failed lastTest first part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))
    psi = qu.uniform(n)
    state = qg.tensor(psi, qc.ket1)
    meas = last(state)
    if qu.equal(state, qg.tensor(meas[0], meas[1]), 0.000001):
        print("passed lastTest second part")
    else:
        print("failed lastTest second part")
        print("    state = " + str(state))
        print("    meas = " + str(meas))

def lastTest345(n, m):
    # Assumes n >= 1. n + 1 is the total number of qbits. m is how many tests to run.
    # Should return a number close to 0.64 --- at least for large m.
    psi0 = 3 / 5
    ketChi = qu.uniform(n)
    psi1 = 4 / 5
    ketPhi = qu.uniform(n)
    ketOmega = psi0 * qg.tensor(ketChi, qc.ket0) + psi1 * qg.tensor(ketPhi, qc.ket1)
    def f():
        if (last(ketOmega)[1] == qc.ket0).all():
            return 0
        else:
            return 1
    acc = 0
    for i in range(m):
        acc += f()
    print("check lastTest345 for frequency near 0.64")
    print("    frequency = ", str(acc / m))


def main():
    firstTest(4)
    firstTest(3)
    firstTest345(3, 10000)
    firstTest345(1, 10000)
    lastTest(4)
    lastTest(3)
    lastTest345(3, 10000)
    lastTest345(1, 10000)

if __name__ == "__main__":
    main()