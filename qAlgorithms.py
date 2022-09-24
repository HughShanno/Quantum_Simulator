'''
Author: Hugh Shanno
Collaborated with Charlie Roslansky
'''

import numpy
import math
import random

import qConstants as qc
import qUtilities as qu
import qGates as qg
import qMeasurement as qm
import qBitStrings as qb

#QAlgorithms

def bennett():
    '''Runs one iteration of the core algorithm of Bennett (1992). 
    Returns a tuple of three items --- |alpha>, |beta>, |gamma> --- each of which is either |0> or |1>.'''
    if random.randint(0,99) > 49:
        ketAlpha = qc.ket0
        ketPsi = qc.ket0
    else:
        ketAlpha = qc.ket1
        ketPsi = qc.ketPlus
    
    if random.randint(0,99) > 49:
        ketBeta = qc.ket0
    else:
        ketBeta = qc.ket1
    
    if numpy.array_equal(ketBeta, qc.ket0):
        ketGamma = qm.measurement(ketPsi)
    else:
        ketGamma = qm.measurement(qg.application(qc.h,ketPsi))
    
    return (ketAlpha, ketBeta, ketGamma)

def deutsch(f):
    '''Implements the algorithm of Deutsch (1985). That is, given a two-qbit gate F 
    representing a function f : {0, 1} -> {0, 1}, returns |1> if f is constant, and |0> if f is not constant.'''
    state = qg.tensor(qc.ketMinus, qc.ketMinus)
    state = qg.application(f, state)
    state = qg.application(qg.tensor(qc.h,qc.h), state)
    return qm.first(state)[0]

def bernsteinVazirani(n, f):
    '''Given n >= 1 and an (n + 1)-qbit gate F representing a function
    f : {0, 1}^n -> {0, 1} defined by mod-2 dot product with an unknown delta
    2 in {0, 1}^n, returns the list or tuple of n classical one-qbit states (each
    |0> or |1>) corresponding to delta.'''
    inputState = qg.tensor(qg.power(qc.ket0,n), qc.ket1)
    secondState = qg.application(qg.power(qc.h,n+1),inputState)
    thirdState = qg.application(f,secondState)
    fourthState = qg.application(qg.power(qc.h,n+1),thirdState)
    ketDelta = []
    state = (qc.ket0, fourthState)
    for j in range(n):
        state = qm.first(state[1])
        ketDelta.append(state[0])
    return ketDelta

def simon(n, f):
    '''The inputs are an integer n >= 2 and an (n + (n - 1))-qbit gate F
    representing a function f: {0, 1}^n -> {0, 1}^(n - 1) hiding an n-bit
    string delta as in the Simon (1994) problem. Returns a list or tuple of n
    classical one-qbit states (each |0> or |1>) corresponding to a uniformly
    random bit string gamma that is perpendicular to delta.'''
    state = qg.power(qc.ket0,2*n-1)
    state = qg.application(qg.tensor(qg.power(qc.h,n),qg.power(qc.i,n-1)),state)
    state = qg.application(f, state)
    for j in range(n-1):
        state = qm.last(state)[0]
    state = qg.application(qg.power(qc.h,n), state)
    gamma = []
    for j in range(n):
        measurement = qm.first(state)
        gamma.append(measurement[0])
        state = measurement[1]
    return gamma

def shor(n, f):
    '''Assumes n >= 1. Given an (n + n)-qbit gate F representing a function
    f: {0, 1}^n -> {0, 1}^n of the form f(l) = k^l % m, returns a list or tuple
    of n classical one-qbit states (|0> or |1>) corresponding to the output of
    Shor's quantum circuit.'''
    inputRegister = qg.power(qc.ket0, n)
    outputRegister = qg.power(qc.ket0, n)
    inputRegister = qg.application(qg.power(qc.h, n), inputRegister)
    state = qg.application(f, qg.tensor(inputRegister, outputRegister))
    for i in range(n):
        state = qm.last(state)[0]
    state = qg.application(qg.fourier(n), state)
    ketDelta = []
    state = (qc.ket0, state)
    for i in range(n):
        state = qm.first(state[1])
        ketDelta.append(state[0])
    return ketDelta

def grover(n, k, f):
    '''Assumes n >= 1, k >= 1. Assumes that k is small compared to 2^n.
    Implements the Grover core subroutine. The F parameter is an (n + 1)-qbit
    gate representing a function f : {0, 1}^n -> {0, 1} such that
    SUM_alpha f(alpha) = k. Returns a list or tuple of n classical one-qbit
    states (either |0> or |1>), such that the corresponding n-bit string delta
    usually satisfies f(delta) = 1.'''
    R = numpy.zeros((2**n,2**n), dtype=numpy.array(0+0j).dtype)
    rho = qg.power(qc.ketPlus, n)
    R = numpy.outer(rho, numpy.transpose(rho))
    R = 2*R
    R = numpy.add(R, -1 * qg.power(qc.i, n))
    t = numpy.arcsin(k*(2**(-n/2)))
    l = int(numpy.round((math.pi/(4*t))-1/2, 0))
    state = qg.power(qc.ket0, n)
    state = qg.tensor(state, qc.ket1)
    state = qg.application(qg.power(qc.h,n+1),state)
    for j in range(l):
        state = qg.application(f, state)
        state = qg.application(qg.tensor(R,qc.i), state)
    gamma = []
    for j in range(n):
        measurement = qm.first(state)
        state = measurement[1]
        gamma.append(measurement[0])
    return gamma


### DEFINING SOME TESTS ###

def bennettTest(m):
    # Runs Bennett's core algorithm m times.
    trueSucc = 0
    trueFail = 0
    falseSucc = 0
    falseFail = 0
    for i in range(m):
        result = bennett()
        if qu.equal(result[2], qc.ket1, 0.000001):
            if qu.equal(result[0], result[1], 0.000001):
                falseSucc += 1
            else:
                trueSucc += 1
        else:
            if qu.equal(result[0], result[1], 0.000001):
                trueFail += 1
            else:
                falseFail += 1
    print("check bennettTest for false success frequency exactly 0")
    print("    false success frequency = ", str(falseSucc / m))
    print("check bennettTest for true success frequency about 0.25")
    print("    true success frequency = ", str(trueSucc / m))
    print("check bennettTest for false failure frequency about 0.25")
    print("    false failure frequency = ", str(falseFail / m))
    print("check bennettTest for true failure frequency about 0.5")
    print("    true failure frequency = ", str(trueFail / m))

def deutschTest():
    def fNot(a):
        return (1 - a[0],)
    resultNot = deutsch(qg.function(1, 1, fNot))
    if qu.equal(resultNot, qc.ket0, 0.000001):
        print("passed deutschTest first part")
    else:
        print("failed deutschTest first part")
        print("    result = " + str(resultNot))
    def fId(a):
        return a
    resultId = deutsch(qg.function(1, 1, fId))
    if qu.equal(resultId, qc.ket0, 0.000001):
        print("passed deutschTest second part")
    else:
        print("failed deutschTest second part")
        print("    result = " + str(resultId))
    def fZero(a):
        return (0,)
    resultZero = deutsch(qg.function(1, 1, fZero))
    if qu.equal(resultZero, qc.ket1, 0.000001):
        print("passed deutschTest third part")
    else:
        print("failed deutschTest third part")
        print("    result = " + str(resultZero))
    def fOne(a):
        return (1,)
    resultOne = deutsch(qg.function(1, 1, fOne))
    if qu.equal(resultOne, qc.ket1, 0.000001):
        print("passed deutschTest fourth part")
    else:
        print("failed deutschTest fourth part")
        print("    result = " + str(resultOne))
        
def bernsteinVaziraniTest(n):
    delta = qb.string(n, random.randrange(0, 2**n))
    def f(s):
        return (qb.dot(s, delta),)
    gate = qg.function(n, 1, f)
    qbits = bernsteinVazirani(n, gate)
    bits = tuple(map(qu.bitValue, qbits))
    diff = qb.addition(delta, bits)
    if diff == n * (0,):
        print("passed bernsteinVaziraniTest")
    else:
        print("failed bernsteinVaziraniTest")
        print(" delta = " + str(delta))

def simonTest(n):
    # Pick a non-zero delta uniformly randomly.
    delta = qb.string(n, random.randrange(1, 2**n))
    # Build a certain matrix M.
    k = 0
    while delta[k] == 0:
        k += 1
    m = numpy.identity(n, dtype=int)
    m[:, k] = delta
    mInv = m
    # This f is a linear map with kernel {0, delta}. So it’s a valid example.
    def f(s):
        full = numpy.dot(mInv, s) % 2
        full = tuple([full[i] for i in range(len(full))])
        return full[:k] + full[k + 1:]
    gate = qg.function(n, n - 1, f)
    
    zeroRow = n*(0,)
    prediction = n*[0]
    simonMatrix = [] #This is the matrix being created by running Simon's multiple times
    while len(simonMatrix) < n-1:
        simonBitString = () #Current gamma from running Simon's
        gamma = simon(n, gate)
        for vector in gamma:
            if numpy.array_equal(vector, qc.ket0):
                simonBitString += (0,)
            else:
                simonBitString += (1,)
        simonMatrix.append(simonBitString)
        simonMatrix = qb.reduction(simonMatrix)
        if simonMatrix[-1] == zeroRow:
            simonMatrix.pop()
    zeroDeltas = n*[1] #contains the delta indices that must equal 0
    
    for row in simonMatrix:
        rowIndexes = [] #contains which delta indices are equal to 1 in each row
        for i in range(len(row)):
            if row[i] == 1:
                rowIndexes.append(i)
                zeroDeltas[i] = 0
        if len(rowIndexes) == 1:
            prediction[rowIndexes[0]] = 0
        elif len(rowIndexes) % 2 == 0:
            for index in rowIndexes:
                prediction[index] = 1
        else:
            for index in rowIndexes:
                prediction[index] = 0
    
    for i in range(len(zeroDeltas)):
        if zeroDeltas[i] == 1:
            prediction[i] = 1
    prediction = tuple(prediction)
    
    if delta == prediction:
        print("passed simonTest")
    else:
        print("failed simonTest")
        print(" delta = " + str(delta))
        print(" prediction = " + str(prediction))
        
def shorTest(n, m):
    k = random.randint(2,m)
    while math.gcd(k,m) != 1:
        k = random.randint(2,m)
    def f(l):
        int_l = qb.integer(l)
        toReturn = qu.powerMod(k, int_l, m)
        return qb.string(n, toReturn)
    gate = qg.function(n, n, f)
    def runShor():
        result = shor(n, gate)
        shorBitString = ()
        for vector in result:
            if numpy.array_equal(vector, qc.ket0):
                shorBitString += (0,)
            else:
                shorBitString += (1,)
        b = qb.integer(shorBitString)
        return b

    
    while True:
        d = m
        d1 = m
        while d >= m:
            b = 0
            while b == 0:
                b = runShor()
            c,d = qu.continuedFraction(n, m, b/(2**n))
        if qu.powerMod(k,d,m) == 1:
            p = d
            break
        while d1 >= m:
            b = 0
            while b == 0:
                b = runShor()
            c1,d1 = qu.continuedFraction(n, m, b/(2**n))
        if qu.powerMod(k,d1,m) == 1:
            p = d1
            break
        lcm = int(d*d1/math.gcd(d,d1))
        if qu.powerMod(k, lcm, m) == 1:
            p = lcm
            break
    
    p1 = 1
    while qu.powerMod(k,p1,m) != 1:
        p1 += 1
    if p == p1:
        print("passed shorTest")
    else:
        print("failed shorTest")
        print("p:", p)
        print("p1:", p1)


    #c,d = qu.continuedFractions(n, m, b/(2**n))
    #if qu.powerMod(k,d,m) == 1:
    #    p = d
    
    
    return

def groverTest(n, k):
    # Pick k distinct deltas uniformly randomly.
    deltas = []
    while len(deltas) < k:
        delta = qb.string(n, random.randrange(0, 2**n))
        if not delta in deltas:
            deltas.append(delta)
    # Prepare the F gate.
    def f(alpha):
        for delta in deltas:
            if alpha == delta:
                return (1,)
        return (0,)
    fGate = qg.function(n, 1, f)
    # Run Grover’s algorithm up to 10 times.
    qbits = grover(n, k, fGate)
    bits = tuple(map(qu.bitValue, qbits))
    j = 1
    while (not bits in deltas) and (j < 10):
        qbits = grover(n, k, fGate)
        bits = tuple(map(qu.bitValue, qbits))
        j += 1
    if bits in deltas:
        print("passed groverTest in " + str(j) + " tries")
    else:
        print("failed groverTest")
        print(" exceeded 10 tries")
        print(" prediction = " + str(bits))
        print(" deltas = " + str(deltas))


def main():
    bennettTest(100000)
    deutschTest()
    bernsteinVaziraniTest(3)
    simonTest(4)
    simonTest(5)
    simonTest(3)
    shorTest(5,5)
    shorTest(4,4)
    shorTest(4,3)
    groverTest(5,1)
    groverTest(7,3)
    groverTest(6,2)

if __name__ == "__main__":
    main()