'''
Author: Hugh Shanno
Collaborated with Charlie Roslansky
'''

import numpy
import math
import random

import qConstants as qc

#QUtilities
def uniform(n):
    '''Assumes n >= 0. Returns a uniformly random n-qbit state.'''
    if n == 0:
        return qc.one
    else:
        psiNormSq = 0
        while psiNormSq == 0:
            reals = numpy.array(
                [random.normalvariate(0, 1) for i in range(2**n)])
            imags = numpy.array(
                [random.normalvariate(0, 1) for i in range(2**n)])
            psi = numpy.array([reals[i] + imags[i] * 1j for i in range(2**n)])
            psiNormSq = numpy.dot(numpy.conj(psi), psi).real
        psiNorm = math.sqrt(psiNormSq)
        return psi / psiNorm

def bitValue(state):
    '''Given a one-qbit state assumed to be exactly classical --- usually because a classical state 
    was just explicitly assigned to it --- returns the corresponding bit value 0 or 1.'''
    if (state == qc.ket0).all():
        return 0
    else:
        return 1

def powerMod(k, l, m):
    '''Given non-negative integer k, non-negative integer l, and positive integer m. 
    Computes k^l mod m. Returns an integer in {0, ..., m - 1}.'''
    kToTheL = 1
    curr = k
    while l >= 1:
        if l % 2 == 1:
            kToTheL = (kToTheL * curr) % m
        l = l // 2
        curr = (curr * curr) % m
    return kToTheL

def equal(a, b, epsilon):
    '''Assumes that n >= 0. Assumes that a and b are both n-qbit states or n-qbit gates.
    Assumes that epsilon is a positive (but usually small) real number.
    Returns whether a == b to within a tolerance of epsilon.
    Useful for doing equality comparisons in the floating-point context.
    Warning: Does not consider global phase changes; for example, two states that are global phase
    changes of each other may be judged unequal. Warning: Use this function sparingly, for inspecting output
    and running tests. Probably you should not use it to make a crucial decision in the middle of a big computation.
    In past versions of CS 358, this function has not existed. I have added it this time just to streamline the tests.'''
    diff = a - b
    if len(diff.shape) == 0:
        # n == 0. Whether they're gates or states, a and b are scalars.
        return abs(diff) < epsilon
    elif len(diff.shape) == 1:
        # a and b are states.
        return sum(abs(diff)) < epsilon
    else:
        # a and b are gates.
        return sum(sum(abs(diff))) < epsilon
    
def continuedFraction(n, m, x0):
    '''x0 is a float in [0, 1). Tries probing depths j = 0, 1, 2, ... until
    the resulting rational approximation x0 ~ c / d satisfies either d >= m or
    |x0 - c / d| <= 1 / 2^(n + 1). Returns a pair (c, d) with gcd(c, d) = 1.'''
    j = 0
    d = 1
    c = 0
    if x0 == 0:
        return (0, 1)
    while d < m and abs(x0 - c/d) > 1/(2**(n+1)):
        c,d = fraction(x0,j)
        j += 1
    return (c,d)


def lowestTerms(c,d):
    gcd = math.gcd(c,d)
    return (int(c/gcd), int(d/gcd))
    
    
def fraction(x0, j):
    if j == 0 or x0 == 0:
        return (0, 1)
    else:
        a0 = int(math.floor(1/x0))
        x1 = 1/x0 - a0
        c,d = fraction(x1,j-1)
        
        return lowestTerms(int(d),int(a0*d + c))
    
