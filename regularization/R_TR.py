from ot.utils import list_to_array
from ot.bregman import sinkhorn
from ot.backend import get_backend


def sinkhorn_lpl1_mm_l1temporal(a, labels_a, b, M, reg, eta=0.1, ta=0.1, numItermax=10,
                     numInnerItermax=200, stopInnerThr=1e-9, verbose=False,
                     log=False, n_states=4):
    r"""
    Solve the entropic regularization optimal transport problem with nonconvex
    group lasso regularization

    The function solves the following optimization problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg} \cdot \Omega_e(\gamma) + \eta \ \Omega_g(\gamma)

        s.t. \ \gamma \mathbf{1} = \mathbf{a}

             \gamma^T \mathbf{1} = \mathbf{b}

             \gamma \geq 0


    where :

    - :math:`\mathbf{M}` is the (`ns`, `nt`) metric cost matrix
    - :math:`\Omega_e` is the entropic regularization term :math:`\Omega_e
      (\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\Omega_g` is the group lasso  regularization term
      :math:`\Omega_g(\gamma)=\sum_{i,c} \|\gamma_{i,\mathcal{I}_c}\|^{1/2}_1`
      where  :math:`\mathcal{I}_c` are the index of samples from class `c`
      in the source domain.
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target weights (sum to 1)

    The algorithm used for solving the problem is the generalized conditional
    gradient as proposed in :ref:`[5, 7] <references-sinkhorn-lpl1-mm>`.


    Parameters
    ----------
    a : array-like (ns,)
        samples weights in the source domain
    labels_a : array-like (ns,)
        labels of samples in the source domain
    b : array-like (nt,)
        samples weights in the target domain
    M : array-like (ns,nt)
        loss matrix
    reg : float
        Regularization term for entropic regularization >0
    eta : float, optional
        Regularization term  for group lasso regularization >0
    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations (inner sinkhorn solver)
    stopInnerThr : float, optional
        Stop threshold on error (inner sinkhorn solver) (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    gamma : (ns, nt) array-like
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters


    .. _references-sinkhorn-lpl1-mm:
    References
    ----------
    .. [5] N. Courty; R. Flamary; D. Tuia; A. Rakotomamonjy,
       "Optimal Transport for Domain Adaptation," in IEEE
       Transactions on Pattern Analysis and Machine Intelligence ,
       vol.PP, no.99, pp.1-1

    .. [7] Rakotomamonjy, A., Flamary, R., & Courty, N. (2015).
       Generalized conditional gradient: analysis of convergence
       and applications. arXiv preprint arXiv:1510.06567.

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.bregman.sinkhorn : Entropic regularized OT
    ot.optim.cg : General regularized OT

    """
    a, labels_a, b, M = list_to_array(a, labels_a, b, M)
    nx = get_backend(a, labels_a, b, M)

    p = 0.5
    epsilon = 1e-3

    indices_labels = []
    classes = nx.unique(labels_a)
    for c in classes:
        idxc, = nx.where(labels_a == c)
        indices_labels.append(idxc)

    W = nx.zeros(M.shape, type_as=M)
    W_temporal = nx.zeros(M.shape, type_as=M)
    for cpt in range(numItermax):
        Mreg = M + eta * W + ta * W_temporal
        transp = sinkhorn(a, b, Mreg, reg, numItermax=numInnerItermax,
                          stopThr=stopInnerThr)
        # the transport has been computed. Check if classes are really
        # separated
        W = nx.ones(M.shape, type_as=M)
        for (i, c) in enumerate(classes):
            majs = nx.sum(transp[indices_labels[i]], axis=0)
            majs = p * ((majs + epsilon) ** (p - 1))
            W[indices_labels[i]] = majs

        W_temporal = nx.full(M.shape, 20, type_as=M) # 15.81139
        for i_s in range(0, transp.shape[0]):
            for i_t_cluster in range(0, int(transp.shape[1]/n_states)):
                sum_list = transp[i_s, i_t_cluster * n_states: (i_t_cluster+1) * n_states]
                mm = nx.sum(sum_list)
                mm = p * ((mm + epsilon) ** (p - 1))
                W_temporal[i_s, (i_s % n_states) + i_t_cluster * n_states] = mm


    return transp
