import math

def sqrt(value):
    return value ** 0.5

def tarantula(ep, ef, np, nf):
    if ef == 0:
        return 0
    if ep + np == 0:
        return 1 if ef > 0 else 0
    return ef / (ef + nf) / (ef / (ef + nf) + ep / (ep + np))

def ochiai(ep, ef, np, nf):
    if ef + nf == 0 or ep + ef == 0:
        return 0
    return ef / sqrt((ef + nf) * (ep + ef))

def barinel(ep, ef, np, nf):
    if ef + ep == 0:
        return 0
    return 1 - ep / (ep + ef)

def dstar2(ep, ef, np, nf):
    if ep + nf == 0:
        return (ef + nf)**2 + 1
    return ef**2 / (ep + nf)

def jaccard(ep, ef, np, nf):
    if ep + ef + nf == 0:
        return ef
    return ef / (ep + ef + nf)

def er1a(ep, ef, np, nf):
    return -1 if nf > 0 else np

def er1b(ep, ef, np, nf):
    return ef - ep / (ep + np + 1)

def er5a(ep, ef, np, nf):
    return ef + ef / (ep + np + 1)

def er5b(ep, ef, np, nf):
    return ef / (ep + ef + np + nf)

def er5c(ep, ef, np, nf):
    return 0 if nf > 0 else 1

def gp2(ep, ef, np, nf):
    return 2 * (ef + sqrt(np)) + sqrt(ep)

def gp3(ep, ef, np, nf):
    return sqrt(abs(pow(ef, 2) - sqrt(ep)))

def gp13(ep, ef, np, nf):
    if ep + ef == 0:
        return 0
    return ef + ef / (2 * ep + ef)
    
def gp19(ep, ef, np, nf):
    return ef * sqrt(abs(ep - ef + nf - np))

FORMULAS = {
    'tarantula': tarantula,
    'ochiai': ochiai,
    'barinel': barinel,
    'dstar2': dstar2,
    'jaccard': jaccard,
    'er1a': er1a,
    'er1b': er1b,
    'er5a': er5a,
    'er5b': er5b,
    'er5c': er5c,
    'gp2': gp2,
    'gp3': gp3,
    'gp13': gp13,
    'gp19': gp19,
}
