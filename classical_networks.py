import numpy as np
import sys
from numba import jit, vectorize
from time import time

@jit
def generateRealization(N,h,myDis=[]):
    """
        Given a system size (N) and disorder strength (h), generate a disorder
        realization and then compute the matrix elements for the Anderson
        orbital basis.

        Optional Param:
            - myDis: a list of length N which specifies the disorder realization
            to use.
    """
    # Compute the single particle energies and the orbitals
    myMat = -np.diag(myDis)
    myMat += np.diag(0.5 * np.ones(N - 1), 1)
    myMat += np.diag(0.5 * np.ones(N - 1), -1)
    (vals,vecs) = np.linalg.eig(myMat)
    # Vals and vecs are the energies and orbitals, respectively.

    # Sort by the position of the orbital (np.eig returns sorted by eigenvalue)
    posns = np.dot(np.arange(0,N,1.),np.abs(vecs)**2)
    ordering =  np.argsort(posns)
    vals = vals[ordering]
    vecs = vecs[:,ordering]

    psi = vecs
    # a copy of the eigenstates shifted with respect to the physical lattice index.
    psi_roll = np.array([[vecs[(j-1)%N,i] for i in range(N)] for j in range(N)])
    # antisymmetrized product of wavefunctions on adjacent sites
    psi_prod_diff = np.array([[[psi[l,n]*psi_roll[l,m] - psi[l,m]*psi_roll[l,n] for l in range(N)] for n in range(N)] for m in range(N)])
    # Density-density interaction energies
    Us = np.sum(np.conj(psi_prod_diff) * psi_prod_diff,axis=2)
    # Three-body facilitated hoppings terms.
    Vs = np.array([[[np.sum(np.conj(psi_prod_diff[j,k,:]) * psi_prod_diff[k,n,:]) for j in range(N)] for k in range(N)] for n in range(N)])

    return vals, Us, Vs

def genRandConfig(N,partNum=1):
    """
        Given a system size (N) and the number of particles (partNum), generate
        a random configuration for which sites are occupied.
    """
    node = np.array([int(i < partNum) for i in range(N)])
    np.random.shuffle(node)
    return node

@vectorize
def bathDensity(energies, tau):
    """
        Given an inverse energy window (tau) and energy differences (energies)
        for the links, return the weighted links.

        See equations 9 and 10 in the paper.
    """
    return 2. / (1. + (energies * tau)**2)


def run_sim(N, h, myDis = [], partNum = -1, tau = 1., N_steps = 100):
    """
        Initialize a network with N sites and disorder strength h.

        Optional parameters:
            - myDis -- an array of length N which specifies the disorder realization.
            - partNum -- the number of fermions in the system.
            - tau -- the inverse energy window
            - N_steps -- number of update steps
    """
    if partNum < 0:
        partNum = N // 2
    if len(myDis) != N:
        myDis = np.random.uniform(low = -h, high = +h, size = N)

    # Computing the orbital energies and interactions for the given disorder realization.
    (ep,Us,Vs) = generateRealization(N, h, myDis = myDis)
    # Repeated indices should always give vanishing U and V.
    # We enforce this below to avoid the case of some numerical error
    # (This scenario should not arise).
    for i in range(N):
        Us[i,i] = 0
        Vs[:,i,i] *= 0
        Vs[i,:,i] *= 0
        Vs[i,i,:] *= 0

    myH = np.diag(ep) + Us
    nodes = genRandConfig(N, partNum = partNum)
    energy = np.dot(np.dot(1. * nodes, myH), 1. * nodes)

    print("Initial node config: ", ''.join([str(int(elem)) for elem in nodes]))

    # Define a matrix links[j,k] which is the product V_{jl}^{(k)} * n_k,
    # i.e. the active hopping matrix elements.
    links = np.einsum('jkl,k', Vs, 1.*nodes)
    links -= np.diag(np.diag(links))

    dt = 0
    current = 0
    myTime = 0

    # Define a matrix link_energies[j,l] which is the energy difference between
    # the current state and the state reached after a particle from site j to site l
    link_energies = np.zeros((N,N))
    interact_energies = np.einsum('jk,k',Us,1.*nodes)
    for j in range(N):
        for l in range(N):
            link_energies[j,l] -= (ep[j] - ep[l])
            link_energies[j,l] -= (interact_energies[j] - interact_energies[l])

    ## Initialization over -- now run updating scheme
    node_history = ["" for _ in range(N_steps)]
    energy_history = np.zeros(N_steps)
    time_history = np.zeros(N_steps)
    occAvg = np.zeros(N)

    # Iterate over the specified number of update steps and record the relevant data.
    for i in range(N_steps):
        if i % (N_steps // 20) == 0:
            print(i)
        energy_history[i] = energy
        time_history[i] = myTime
        node_history[i] = ''.join([str(int(elem)) for elem in nodes])

        old_nodes = np.copy(nodes)

        dt, dE, l, le = update_network(N, nodes, tau, links, link_energies, Us, Vs)

        energy += dE
        myTime += dt

        links = l
        link_energies = le

        occAvg += old_nodes * dt

    # The time averaged site occupations.
    occAvg /= time_history[-1]

    return (node_history, energy_history, time_history, occAvg, myDis)

@jit
def update_network(N, nodes, tau, links, link_energies, Us, Vs):
    """
        Update scheme for the Monte Carlo.
        (1) Compute the escape rates for each particle
        (2) Randomly select a particle to escape and choose which link it takes.
        (3) Update the stored link weights and such.
    """

    # link weights as defined in Eq. 9 and 10 of our paper on the arxiv.
    link_weights = np.abs(links)**2 * bathDensity(link_energies, tau)
    # Compute the escape rates for each particle present on the chain.
    escape_rates = nodes * np.array([np.dot((1.-nodes), link_weights[j,:]) for j in range(N)])
    particle_inds = np.arange(N)[nodes == 1]
    # Randomly sample a waiting time for each particle
    escape_times = np.array([np.random.exponential(scale=1./(elem + 1e-30)) for elem in escape_rates])[particle_inds]

    # Identify the fastest particle
    my_ind = np.argmin(escape_times)
    my_wait = escape_times[my_ind]
    my_ind = particle_inds[my_ind]

    # choose a random link for the particle to hop along
    open_inds = np.arange(N)[nodes == 0]
    proposal_weights = (link_weights[my_ind,:])[open_inds]
    proposal_weights /= np.sum(proposal_weights)
    r = np.random.rand()
    cumsum_weights = np.cumsum(proposal_weights)
    link_choice = np.searchsorted(cumsum_weights, r, side="right")
    link_choice = open_inds[link_choice]

    # update the nodes and get the energy change
    nodes[my_ind] = 0
    nodes[link_choice] = 1
    energy_change = link_energies[my_ind, link_choice]

    # update the link weights
    links -= Vs[:,my_ind,:]
    links += Vs[:,link_choice,:]

    # update the link energy differences
    for j in range(N):
        for l in range(N):
            link_energies[j,l] += Us[j,my_ind] - Us[l,my_ind]
            link_energies[j,l] -= Us[j,link_choice] - Us[l,link_choice]

    assert(my_wait > 0)
    return my_wait, energy_change, links, link_energies

############# Running from terminal #############
# When running from the terminal one may enter parameters via the command line
# as python classical_networks.py [ARG1 ARG2 ... ]
# Arguments
#     Argument 1: N -- the number of sites in the chain
#     Argument 2: h -- the disorder strength
#     Argument 3: myTau -- the dephasing time
#     Argument 4: numSteps -- the number of update steps to run the Monte Carlo for.
#     Argument 5: config_str -- a string to append to all filenames when saving data.
# Output Files:
#     1) disorder_[...].txt -- the disorder realization used.
#     2) occupations_[...].txt -- the time averaged site occupations
#     3) times_[...].txt -- the physical times after each update step.
#     4) energies_[...].txt -- the energies after each update step.
#     5) nodes_[...].txt -- the configuration of occupied orbitals after each update step.
#################################################
if __name__ == "__main__":
    # Default parameters
    N = 20
    h = 4
    numSteps = 1000
    myTau = 1

    if len(sys.argv) >= 2:
        N = int(sys.argv[1])
    myPartNum = N // 2

    if len(sys.argv) >= 3:
        h = float(sys.argv[2])

    if len(sys.argv) >= 4:
        myTau = float(sys.argv[3])

    if len(sys.argv) >= 5:
        numSteps = int(sys.argv[4])

    if len(sys.argv) >= 6:
        config_str = str(sys.argv[5])
    else:
        config_str = "N_{0:d}_h_{1:.2f}_tau_{2:.3f}_jitted".format(N,h,myTau)

    print("N = {0:d}, h = {1:.2f}, tau = {2:.2f}".format(N,h,myTau))

    # Generate the disorder realization
    myDis = np.random.uniform(low=-h, high=h, size=N)
    # Run the simulation
    (node_history, energy_history, time_history, occAvg, myDis) = run_sim(N, h, partNum=myPartNum, tau=myTau, N_steps = numSteps, myDis=myDis)

    print("Final time: {0:.2f}".format(time_history[-1]))

    # Save the results
    fn_prefix = "Results/"
    np.savetxt(fn_prefix + "disorder_" + config_str + ".txt", myDis)
    np.savetxt(fn_prefix + "occupations_" + config_str + ".txt", occAvg)
    np.savetxt(fn_prefix + "times_" + config_str + ".txt", time_history)
    np.savetxt(fn_prefix + "energies_" + config_str + ".txt", energy_history)

    with open(fn_prefix + "nodes_" + config_str + ".txt", 'w') as f:
        for i in range(len(node_history)):
            f.write(node_history[i])
            f.write("\n")
