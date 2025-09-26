from qiskit import QuantumCircuit
from qiskit.circuit.classical import expr
from qiskit.quantum_info import Statevector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.visualization import plot_histogram, plot_state_city

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import matplotlib.pyplot as pyplot
import numpy as np

# filters out any state where a qubit in flag_qubits is measured as 1
def filter_states(counts, flag_qubits):
    return {state: n for state, n in counts.items()
            if all(state[-(q + 1)] == '0' for q in flag_qubits)}

# compute the fidelity between a noisy and ideal results matrix
def get_fidelity(noisy, ideal):
    return np.trace((noisy.transpose() / np.sum(noisy)) @ ideal)

# sums states depending on the first and last bit
def condense_count(mat):
    result = np.array([0.0, 0.0, 0.0, 0.0])

    # note: bitstring is in reverse order (leftmost is highest qubit index)
    for bitstr in mat:
        # if '00'
        if bitstr[0] == '0' and bitstr[-1] == '0':
            result[0] += mat[bitstr]
        # if '01'
        elif bitstr[0] == '1' and bitstr[-1] == '0':
            result[1] += mat[bitstr]
        # if '10'
        elif bitstr[0] == '0' and bitstr[-1] == '1':
            result[2] += mat[bitstr]
        # if '11'
        elif bitstr[0] == '1' and bitstr[-1] == '1':
            result[3] += mat[bitstr]

    return result

# get the ideal probabilities from a circuit with its state vector
def get_ideal_prob(sim, qc):
    qc.save_statevector()
    state_vector = sim.run(qc).result().get_statevector(qc)
    return Statevector(state_vector).probabilities_dict()

# creates a quantum circuit of input size which has a ghz state
def create_ghz_state(size: int = 3):
    circuit = QuantumCircuit(size, size)

    # middle index
    mid = (size - 1) // 2

    circuit.h(mid)
    circuit.cx(mid, mid + 1)

    for i in range(mid):
        circuit.cx(mid - i, mid - (i + 1))
        circuit.cx(mid + (i + 1), mid + (i + 2))
    
    return circuit

# implements a long range CNOT protocol
def long_range_cnot(circuit: QuantumCircuit, control, target):
    distance = target - control if control < target else control - target
    mid = (distance - 1) // 2 # middle index

    # generate ghz state between control and target
    circuit.h(mid)
    circuit.cx(mid, mid + 1)

    for i in range(mid - 1):
        circuit.cx(mid - i, mid - (i + 1))
        circuit.cx(mid + (i + 1), mid + (i + 2))

    # reduce to a bell state
    for i in range(mid - 2):
        circuit.cx(mid - ((mid - 3) - i), mid - ((mid - 2) - i))
        circuit.cx(mid + ((mid - 2) - i), mid + ((mid - 1) - i))

    circuit.cx(mid, mid + 1)
    circuit.h(mid)

    # measure middle qubits
    for i in range(2, distance - 1):
        circuit.measure(i, i)

    # teleport CNOT
    circuit.cx(0, 1)
    circuit.cx(target - 1, target)

    circuit.measure(1, 1)
    circuit.h(target - 1)
    circuit.measure(target - 1, target - 1)

    with circuit.if_test((circuit.clbits[1], 1)):
        circuit.x(target)

    with circuit.if_test(expr.bit_xor(circuit.clbits[target - 1], circuit.clbits[mid])):
        circuit.z(0)


###############################################################################

n = 8
qcs = [QuantumCircuit(n, n) for i in range(4)]

# set circuit initial states
# qcs[0] is already initialized to control: 0, target: 0
# create control: 0, target: 1
qcs[1].x(n - 1)

# create control: 1, target: 0
qcs[2].x(0)

# create control: 1, target: 1
qcs[3].x(0)
qcs[3].x(n - 1)

# apply protocol to each circuit
for circuit in qcs:
    long_range_cnot(circuit, 0, n - 1)

# prepare simulator (with no noise)
aer_ideal = AerSimulator()

# get ideal probabilities and format them
ideal_probs = [get_ideal_prob(aer_ideal, circuit) for circuit in qcs]
ideal_matrix = np.array([condense_count(prob) for prob in ideal_probs])

print("Ideal probabilities:")
print(ideal_matrix)

# noisy results
backend = GenericBackendV2(num_qubits=8)
noise_model = NoiseModel.from_backend(backend)
aer_noisy = AerSimulator(noise_model=noise_model)
shots = 1000

# add a measure for target and control to get results
for circuit in qcs:
    circuit.measure(0, 0)
    circuit.measure(n - 1, n - 1)

# run each circuit and record the results
counts = aer_noisy.run(qcs, shots=shots).result().get_counts()

# discard if flag qubits are 1
flag_qubits = [2, 4, 5]

# filter and condense the results of running the simulator
noisy_matrix = np.array([condense_count(filter_states(count, flag_qubits)) for count in counts])

print("\nNoisy shot counts:")
print(noisy_matrix)

# calculate the fidelity of our circuits
fidelity = get_fidelity(noisy_matrix, ideal_matrix)

print("\nFidelity: " + str(fidelity))

pyplot.show()

# TODO:
# [DONE] make quantum circuits a list (to iter instead of repeat)
# [DONE] correct fidelity function OUTPUT ONE VALUE
# [DONE] get dinner
# [DONE] clean the code

# fix the loop to work with an arbitrary amount of qubits
# find/create toy problems for mapping
# create a pathfinding script which decides whether to go to the teleportation station or directly
# decide how to split the work

# figure out how to decide where to set up teleportation stations