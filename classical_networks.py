import numpy as np
import sys

from collections import defaultdict
from time import time

def generateRealization(N,h,myDis=[]):
    """
        Given a system size (N) and disorder strength (h), generate a disorder
        realization and then compute the matrix elements for the Anderson
        orbital basis.

        Optional Param:
            - myDis: a list of length N which specifies the disorder realization
            to use.
    """

    # Specify the disorder profile
    if len(myDis) != N:
        myDis = h * (2 * np.random.rand(N) - 1)

    # Compute the single particle energies and the orbitals
    myMat = -np.diag(myDis)
    myMat += np.diag(0.5 * np.ones(N - 1), 1)
    myMat += np.diag(0.5 * np.ones(N - 1), -1)
    (vals,vecs) = np.linalg.eig(myMat)
    # Vals and vecs are the energies and orbitals, respectively.

    # Compute the matrix elements U_{ij} for density-density interaction
    psi = vecs
    psi_roll = np.roll(vecs, axis=1, shift=1)
    psi_prod_diff = np.array([[[psi[l,n]*psi_roll[l,m] - psi[l,m]*psi_roll[l,n] for l in range(N)] for n in range(N)] for m in range(N)])
    Us = np.sum(np.conj(psi_prod_diff) * psi_prod_diff,axis=0)
    # Vs = np.sum(np.conj(psi_prod_diff[:,j,k]) * psi_prod_diff[:,k,n],axis=0)
    Vs = np.array([[[np.sum(np.conj(psi_prod_diff[:,j,k]) * psi_prod_diff[:,k,n]) for j in range(N)] for k in range(N)] for n in range(N)])

    return vals, Us, Vs

def genRandConfig(N,partNum=1):
    """
        Given a system size (N) and the number of particles (partNum), generate
        a random configuration for which sites are occupied.
    """
    node = np.array([int(i < partNum) for i in range(N)])
    np.random.shuffle(node)
    return node

def bathDensity(energies, tau):
    """
        TODO: Specify a bath
        Given an inverse energy window (tau) and energy differences (energies)
        for the links, return the weighted links.
    """
    # return (tau / 2.) * np.exp(-tau * np.abs(energies)) # exponential bath
    # return tau * np.exp(- (energies * tau)**2 / 2.) / np.sqrt(2. * np.pi) # Gaussian bath
    return (2. / np.sqrt(np.pi)) * tau / (1. + (energies * tau)**2)

class Network:
    """
        Network class ...
    """
    def __init__(self, N, h, myDis = [], partNum = -1, tau = 1.):
        """
            Initialize a network with N sites and disorder strength h.

            Optional parameters:
                - myDis -- an array of length N which specifies the disorder realization.
                - partNum -- the number of fermions in the system.
                - tau -- the inverse energy window
        """
        self.N = N
        self.h = h
        self.tau = tau

        if partNum < 0:
            self.partNum = self.N // 2
        else:
            self.partNum = partNum

        if len(myDis) == 0:
            self.dis = np.random.uniform(low = -self.h, high = self.h, size = N)
        else:
            self.dis = myDis

        # Generate the Anderson orbital representation
        (ep,Us,Vs) = generateRealization(self.N, self.h, myDis = self.dis)
        self.ep = ep
        self.Us = Us
        self.Vs = Vs

        # Enforce which matrix elements ought to be zero in case of
        # some rounding error.
        for i in range(self.N):
            self.Us[i,i] = 0
            self.Vs[:,i,i] *= 0
            self.Vs[i,:,i] *= 0
            self.Vs[i,i,:] *= 0

        # Initialize the Hamiltonian, node configuration, and energy
        self.Hamiltonian = np.diag(self.ep) + self.Us
        self.nodes = genRandConfig(self.N, partNum = self.partNum)
        print("Initial node config: " + ''.join([str(elem) for elem in self.nodes]))
        self.energy = np.dot(np.dot(self.nodes, self.Hamiltonian), self.nodes)

        # Initialize the links V_{jl} = \sum_k V_{jl}^{(k)} n_k
        self.links = np.einsum('jkl,k', self.Vs, self.nodes)
        self.links -= np.diag(np.diag(self.links)) # Avoid possible errors

        self.current = 0
        self.dt = 0
        self.time = 0

        # Calculate the energy differences associated with
        # moving a particle from site j to site l.
        self.link_energies = np.zeros((self.N, self.N))
        interact_energies = np.einsum('jk,k', self.Us, self.nodes)
        for j in range(N):
            for l in range(N):
                self.link_energies[j,l] -= self.ep[j] - self.ep[l]
                self.link_energies[j,l] -= interact_energies[j] - interact_energies[l]

    def update(self):
        """
            Update scheme for the Monte Carlo.
            (1) Compute the escape rates for each particle
            (2) Randomly select a particle to escape and choose which link it takes.
            (3) Update the stored link weights and such.
        """

        # Calculate the link weight 2*pi * |V_{jl}|^2 * \rho(\tau, energy diff)
        link_weights = np.abs(self.links)**2 * bathDensity(self.link_energies, self.tau)
        # Calculate the escape rates \Gamma_j = \sum_l w_{jl} (1 - n_l)
        escape_rates = self.nodes * np.array([np.dot((1 - self.nodes), link_weights[j,:]) for j in range(N)])
        # escape_rates = np.sum(np.array([[ (1 - self.nodes[l]) * link_weights[j,l] for j in range(self.N)] for l in range(self.N)]),axis=1)
        particleInds = np.arange(self.N)[self.nodes == 1]
        # Calculate random waiting times for the particles to escape to another site
        escape_times = np.array([np.random.exponential(scale = 1. / (elem + 1e-30)) for elem in escape_rates])[particleInds]

        # Identify the fastest particle to move
        myInd = np.argmin(escape_times)
        myTime = escape_times[myInd]
        myInd = particleInds[myInd]
        assert(self.nodes[myInd] == 1)
        self.dt = myTime
        self.time += self.dt

        # Choose which site the particle hops to
        openInds = np.arange(self.N)[self.nodes == 0]
        proposal_weights = (link_weights[myInd,:])[openInds]
        if np.sum(proposal_weights) > 0:
            proposal_weights /= np.sum(proposal_weights)
            linkChoice = int(np.random.choice(np.arange(len(openInds)), p = proposal_weights))
        else:
            linkChoice = int(np.argmax(proposal_weights))
        linkChoice = openInds[linkChoice]

        assert(self.nodes[linkChoice] == 0)

        # Update the state and energy
        self.nodes[myInd] = 0
        self.nodes[linkChoice] = 1
        self.energy += self.link_energies[myInd,linkChoice]

        # Update the links since site myInd cannot facilitate, but linkChoice can
        self.links -= self.Vs[:,myInd,:]
        self.links += self.Vs[:,linkChoice,:]

        # Update the energy differences for links
        for j in range(self.N):
            for l in range(self.N):
                self.link_energies[j,l] += self.Us[j,myInd] - self.Us[l,myInd]
                self.link_energies[j,l] -= self.Us[j,linkChoice] - self.Us[l,linkChoice]

        assert(np.sum(self.nodes) == self.partNum)


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
        config_str = "N_{0:d}_h_{1:.2f}_tau_{2:.3f}".format(N,h,myTau)

    print("N = {0:d}, h = {1:.2f}, tau = {2:.2f}".format(N,h,myTau))

    myNet = Network(N, h, tau = myTau, partNum = myPartNum)
    print("Constructed network")

    # Saving the full configuration history, energies, and times
    node_history = []
    energy_history = np.zeros(numSteps)
    time_history = np.zeros(numSteps)
    occAvg = np.zeros(N)

    for i in range(numSteps):
        if (i % (numSteps // 20) == 0):
            print(i)

        energy_history[i] = myNet.energy
        time_history[i] = myNet.time
        node_history.append(''.join([str(int(elem)) for elem in myNet.nodes]))

        oldNodes = np.copy(myNet.nodes)

        myNet.update()

        occAvg += oldNodes * myNet.dt

    occAvg /= time_history[-1]

    print("Final time: {0:.2f}".format(time_history[-1]))

    fn_prefix = "Results/"

    np.savetxt(fn_prefix + "disorder_" + config_str + ".txt", myNet.dis)
    np.savetxt(fn_prefix + "occupations_" + config_str + ".txt", occAvg)
    np.savetxt(fn_prefix + "times_" + config_str + ".txt", time_history)
    np.savetxt(fn_prefix + "energies_" + config_str + ".txt", energy_history)

    with open(fn_prefix + "nodes_" + config_str + ".txt", 'w') as f:
        for i in range(len(node_history)):
            f.write(node_history[i])
            f.write("\n")
