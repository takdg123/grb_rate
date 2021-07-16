import numpy as np


distM = lambda x, n: -3/2.*x + n
gaus = lambda x, n, x0, sig: n*np.exp(-(x-x0)**2./sig**2.)

PL = lambda E, N0, gamma: N0*(E/50.)**(gamma)
CUTOFFPL = lambda E, N0, gamma, Ep: N0*(E/50.)**(gamma)*np.exp(-E*(2.+gamma)/Ep)

ePL = lambda E, N0, gamma: E*N0*(E/50.)**(gamma)
eCUTOFFPL = lambda E, N0, gamma, Ep: E*N0*(E/50.)**(gamma)*np.exp(-E*(2.+gamma)/Ep)
BAND1 = lambda E, N0, alpha, beta, Ep: N0*(E/100.)**alpha*np.exp(-E*(2.+alpha)/Ep)
BAND2 = lambda E, N0, alpha, beta, Ep: N0*(E/100.)**beta*((Ep/100)*(alpha-beta)/(2+alpha))**(alpha-beta)*np.exp(-(alpha-beta))

def BAND(E, N0, alpha, beta, Ep):
    engs = np.asarray(E)
    vals = []
    eb = (alpha-beta)*Ep/(2+alpha)
    val1 = BAND1(engs[engs<=eb], N0, alpha, beta, Ep)
    val2 = BAND2(engs[engs>eb], N0, alpha, beta, Ep)
    val = val1.tolist() + val2.tolist()
    return np.asarray(val)

def eBAND(E, N0, alpha, beta, Ep):
    engs = np.asarray(E)
    vals = []
    eb = (alpha-beta)*Ep/(2+alpha)
    val1 = E*BAND1(engs[engs<=eb], N0, alpha, beta, Ep)
    val2 = E*BAND2(engs[engs>eb], N0, alpha, beta, Ep)
    val = val1.tolist() + val2.tolist()
    return np.asarray(val)

def SBPL(E, N0, idx1, idx2, Eb):
    m = (idx2-idx1)/2.
    b = (idx1+idx2)/2.
    q = np.log10(E/Eb)/0.3
    q_piv = np.log10(0.1/Eb)/0.3
    a = m*0.3*np.log((np.exp(q)+np.exp(-q))/2.)
    a_piv = m*0.3*np.log((np.exp(q_piv)+np.exp(-q_piv))/2.)
    val = N0*(E/0.1)**b*10**(a-a_piv)
    return np.asarray(val)

def center_pt(x):
    return (x[1:]+x[:-1])/2.


 