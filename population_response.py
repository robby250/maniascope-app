"""Nuisance skill specialization in a public score-response fit.

Player/year level alone makes maps played by specialists look intrinsically
easy, and maps attempted by novices look intrinsically hard. These random slopes
are TRAIN/public-fit context, never named-map features in the chart calculator.
"""
import numpy as np


def skills(f):
    s=f.get('sk',{})
    return (min(1.,s.get('ln',0.)),min(1.,s.get('sv',0.)),
            min(1.,max(s.get('chordjack',0.),s.get('jackspeed',0.))),
            min(1.,s.get('jumptrill',0.)),min(1.,s.get('technical',0.)))


def eliminate(X, y, z, groups, precision=4.):
    """Schur complement of pooled per-player skill slopes, after demeaning.

    X/y are centered within the same player-keymode-year groups. With a finite
    prior, unsupported specialists shrink toward the population rather than
    acquiring arbitrary offsets from one score.
    """
    n=int(groups.max())+1
    counts=np.bincount(groups,minlength=n)
    means=np.stack([np.bincount(groups,z[:,j],minlength=n)/np.maximum(1,counts)
                    for j in range(z.shape[1])],axis=1)
    centered=z-means[groups];q=z.shape[1];p=X.shape[1]
    zx=np.empty((n,q,p));zy=np.empty((n,q));zz=np.empty((n,q,q))
    for i in range(q):
        zy[:,i]=np.bincount(groups,centered[:,i]*y,minlength=n)
        for j in range(q):zz[:,i,j]=np.bincount(groups,centered[:,i]*centered[:,j],minlength=n)
        for j in range(p):zx[:,i,j]=np.bincount(groups,centered[:,i]*X[:,j],minlength=n)
    zz+=np.eye(q)*precision
    ix=np.linalg.solve(zz,zx)
    iy=np.linalg.solve(zz,zy[...,None])[...,0]
    gram=np.einsum('nqi,nqj->ij',zx,ix)
    rhs=np.einsum('nqi,nq->i',zx,iy)
    def offsets(beta):
        slopes=np.linalg.solve(zz,(zy-np.einsum('nqi,i->nq',zx,beta))[...,None])[...,0]
        return np.sum(centered*slopes[groups],axis=1),np.std(slopes,axis=0).tolist()
    return gram,rhs,offsets
