'''
Author: Hugh Shanno
Collaborated with Charlie Roslansky
'''

import numpy
import math
import random

import qConstants as qc
import qUtilities as qu
import qBitStrings as qb

#QGates
def application(u, ketPsi):
    '''Assumes n >= 1. Applies the n-qbit gate U to the n-qbit state |psi>,
    returning the n-qbit state U |psi>.'''
    return u.dot(ketPsi)

def tensor(a, b):
    '''Assumes that n, m >= 1. Assumes that a is an n-qbit state and b is an
    m-qbit state, or that a is an n-qbit gate and b is an m-qbit gate. Returns
    the tensor product of a and b, which is an (n + m)-qbit gate or state.'''
    if len(a.shape) == 1:
        tensorProduct = numpy.zeros((a.shape[0]*b.shape[0]), dtype=numpy.array(0 + 0j).dtype)
        for i in range(int(len(tensorProduct)/b.shape[0])):
            tensorProduct[i*b.shape[0]:i*b.shape[0]+b.shape[0]] = a[i]*b
        return tensorProduct
    else:
        tensorProduct = numpy.zeros((a.shape[1]*b.shape[1],a.shape[0]*b.shape[0]), dtype=numpy.array(0+0j).dtype)
        for i in range(int(len(tensorProduct)/b.shape[1])):
            for j in range(int(len(tensorProduct[0])/b.shape[0])):
                tensorProduct[i*b.shape[1]:i*b.shape[1]+b.shape[1],j*b.shape[0]:j*b.shape[0]+b.shape[0]] = a[i][j]*b
        return tensorProduct

def function(n, m, f):
    '''Assumes n, m >= 1. Given a Python function f : {0, 1}^n -> {0, 1}^m.
    That is, f takes as input an n-bit string and produces as output an m-bit
    string, as defined in qBitStrings.py. Returns the corresponding
    (n + m)-qbit gate F.'''
    
    F = numpy.zeros(((2**n)*(2**m),(2**n)*(2**m)), dtype=numpy.array(0+0j).dtype)
    for i in range(2**n):
        alpha = qb.string(n,i)
        ketAlpha = numpy.zeros((2**n), dtype=numpy.array(0+0j).dtype)
        ketAlpha[qb.integer(alpha)] = (1+0j)
        for j in range(2**m):
            beta = qb.string(m,j)
            secondBitString = qb.addition(beta, f(alpha))
            ketSecondBitString = numpy.zeros((2**m), dtype=numpy.array(0+0j).dtype)
            ketSecondBitString[qb.integer(secondBitString)] = (1+0j)
            state = tensor(ketAlpha, ketSecondBitString)
            F[:,i*(2**m)+j] = state.T
    return F
            
    
    
def power(stateOrGate, m):
    '''Assumes n >= 1. Given an n-qbit gate or state and m >= 1, returns the
    mth tensor power, which is an (n * m)-qbit gate or state. For the sake of
    time and memory, m should be small.'''
    state = stateOrGate
    for i in range(m-1):
        state = tensor(state, stateOrGate)
    return state
    
def fourier(n):
    '''Assumes n >= 1. Returns the n-qbit quantum Fourier transform gate T.'''
    TGate = numpy.zeros((2**n,2**n), dtype=numpy.array(0+0j).dtype)
    for a in range(2**n):
        for b in range(2**n):
            pass
            TGate[a,b] = numpy.exp(1j*2*numpy.pi*a*b/(2**n))
            TGate[a,b] *= 1/(2**(n/2))
    return TGate

def fourierRecursive(n):
    '''Assumes n >= 1. Returns the n-qbit quantum Fourier transform gate T.
    Computes T recursively rather than from the definition.'''
    if n == 1:
        return qc.h
    else:
        R = tensor(qc.i, fourierRecursive(n-1))
        S = fourierS(n)
        omega = numpy.exp(1j*2*numpy.pi/4)
        D = numpy.array([[1 + 0j, 0 + 0j],[0 + 0j, omega]])
        for k in range(3,n+1):
            omega = numpy.exp(1j*2*numpy.pi/(2**k))
            newArray = numpy.array([[1 + 0j, 0 + 0j],[0 + 0j, omega]])
            D = tensor(D,newArray)
        Q = application(tensor(qc.h,power(qc.i,n-1)),directSum(power(qc.i,n-1),D))
        partial = application(Q,R)
        return application(partial,S)

def fourierS(n):
    if n == 1:
        return qc.i
    if n == 2:
        return qc.swap
    else:
        return application(tensor(qc.swap, power(qc.i, n-2)), tensor(qc.i, fourierS(n-1)))

def directSum(A, B):
    newMatrix = numpy.zeros((A[:,0].shape[0]+B[:,0].shape[0],A[0,:].shape[0]+B[0,:].shape[0]), dtype=numpy.array(0+0j).dtype)
    newMatrix[0:A[:,0].shape[0],0:A[0,:].shape[0]] = A
    newMatrix[A[:,0].shape[0]:,A[0,:].shape[0]:] = B
    return newMatrix

def distant(gate):
    '''Given an (n + 1)-qbit gate U (such as a controlled-V gate, where V is
    n-qbit), performs swaps to insert one extra wire between the first qbit and
    the other n qbits. Returns an (n + 2)-qbit gate.'''
    n = int(math.log(gate.shape[0], 2))-1
    newGate = application(tensor(qc.swap, power(qc.i, n)),tensor(qc.i, gate))
    newGate = application(newGate, tensor(qc.swap, power(qc.i, n)))
    return newGate

def ccNot():
    '''Returns the three-qbit ccNOT (i.e., Toffoli) gate. The gate is
    implemented using five specific two-qbit gates and some SWAPs.'''
    v = 1/(2**0.5)*numpy.array([[1 + 0j, 0 + 1j],[0 - 1j, -1 + 0j]])
    u = numpy.array([[1 + 0j, 0 + 0j],[0 + 0j, 0 - 1j]])
    cv = directSum(qc.i,v)
    cz = directSum(qc.i,qc.z)
    cu = directSum(qc.i, u)
    circuit = application(tensor(qc.i, cz), distant(cv))
    circuit = application(circuit, circuit)
    circuit = application(tensor(cu, qc.i), circuit)
    return circuit

def groverR3():
    '''Assumes that n = 3. Returns -R, where R is Grover's n-qbit gate for
    reflection across |rho>. Builds the gate from one- and two-qbit gates,
    rather than manually constructing the matrix.'''
    minusR = power(qc.h, 3)
    minusR = application(minusR, power(qc.x, 3))
    minusR = application(minusR, tensor(power(qc.i, 2), qc.h))
    minusR = application(minusR, ccNot())
    minusR = application(minusR, tensor(power(qc.i, 2), qc.h))
    minusR = application(minusR, power(qc.x, 3))
    minusR = application(minusR, power(qc.h, 3))
    return minusR

### DEFINING SOME TESTS ###

def applicationTest():
    # These simple tests detect type errors but not much else.
    answer = application(qc.h, qc.ketMinus)
    if qu.equal(answer, qc.ket1, 0.000001):
        print("passed applicationTest first part")
    else:
        print("FAILED applicationTest first part")
        print("    H |-> = " + str(answer))
    ketPsi = qu.uniform(2)
    answer = application(qc.swap, application(qc.swap, ketPsi))
    if qu.equal(answer, ketPsi, 0.000001):
        print("passed applicationTest second part")
    else:
        print("FAILED applicationTest second part")
        print("    |psi> = " + str(ketPsi))
        print("    answer = " + str(answer))

def tensorTest():
    # Pick two gates and two states.
    u = qc.x
    v = qc.h
    ketChi = qu.uniform(1)
    ketOmega = qu.uniform(1)
    # Compute (U tensor V) (|chi> tensor |omega>) in two ways.
    a = tensor(application(u, ketChi), application(v, ketOmega))
    b = application(tensor(u, v), tensor(ketChi, ketOmega))
    # Compare.
    if qu.equal(a, b, 0.000001):
        print("passed tensorTest")
    else:
        print("FAILED tensorTest")
        print("    a = " + str(a))
        print("    b = " + str(b))

def functionTest(n, m):
    # 2^n times, randomly pick an m-bit string.
    values = [qb.string(m, random.randrange(0, 2**m)) for k in range(2**n)]
    # Define f by using those values as a look-up table.
    def f(alpha):
        a = qb.integer(alpha)
        return values[a]
    # Build the corresponding gate F.
    ff = function(n, m, f)
    # Helper functions --- necessary because of poor planning.
    def g(gamma):
        if gamma == 0:
            return qc.ket0
        else:
            return qc.ket1
    def ketFromBitString(alpha):
        ket = g(alpha[0])
        for gamma in alpha[1:]:
            ket = tensor(ket, g(gamma))
        return ket
    # Check 2^n - 1 values somewhat randomly.
    alphaStart = qb.string(n, random.randrange(0, 2**n))
    alpha = qb.next(alphaStart)
    while alpha != alphaStart:
        # Pick a single random beta to test against this alpha.
        beta = qb.string(m, random.randrange(0, 2**m))
        # Compute |alpha> tensor |beta + f(alpha)>.
        ketCorrect = ketFromBitString(alpha + qb.addition(beta, f(alpha)))
        # Compute F * (|alpha> tensor |beta>).
        ketAlpha = ketFromBitString(alpha)
        ketBeta = ketFromBitString(beta)
        ketAlleged = application(ff, tensor(ketAlpha, ketBeta))
        # Compare.
        if not qu.equal(ketCorrect, ketAlleged, 0.000001):
            print("failed functionTest")
            print(" alpha = " + str(alpha))
            print(" beta = " + str(beta))
            print(" ketCorrect = " + str(ketCorrect))
            print(" ketAlleged = " + str(ketAlleged))
            print(" and here's F...")
            print(ff)
            return
        else:
            alpha = qb.next(alpha)
    print("passed functionTest")
    
def fourierTest(n):
    if n == 1:
    # Explicitly check the answer.
        t = fourier(1)
        if qu.equal(t, qc.h, 0.000001):
            print("passed fourierTest")
        else:
            print("failed fourierTest")
            print(" got T = ...")
            print(t)
    else:
        t = fourier(n)
        # Check the first row and column.
        const = pow(2, -n / 2) + 0j
        for j in range(2**n):
            if not qu.equal(t[0, j], const, 0.000001):
                print("failed fourierTest first part")
                print(" t = ")
                print(t)
                return
        for i in range(2**n):
            if not qu.equal(t[i, 0], const, 0.000001):
                print("failed fourierTest first part")
                print(" t = ")
                print(t)
                return
        print("passed fourierTest first part")
        # Check that T is unitary.
        tStar = numpy.conj(numpy.transpose(t))
        tStarT = numpy.matmul(tStar, t)
        id = numpy.identity(2**n, dtype=qc.one.dtype)
        if qu.equal(tStarT, id, 0.000001):
            print("passed fourierTest second part")
        else:
            print("failed fourierTest second part")
            print(" T^* T = ...")
            print(tStarT)
    
def fourierRecursiveTest(n):
    prediction = fourierRecursive(n)
    control = fourier(n)
    if qu.equal(prediction, control, 0.000001):
        print("passed fourierRecursiveTest")
    else:
        print("failed fourierRecursiveTest")
        print("control:", control)
        print("prediction:", prediction)



def main():
    applicationTest()
    applicationTest()
    tensorTest()
    tensorTest()
    functionTest(4,3)
    functionTest(5,4)
    fourierTest(3)
    fourierTest(4)
    fourierTest(5)
    fourierRecursiveTest(3)
    fourierRecursiveTest(4)
    fourierRecursiveTest(5)

if __name__ == "__main__":
    main()