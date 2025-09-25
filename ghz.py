from qiskit import QuantumCircuit
from qiskit.circuit.classical import expr
from qiskit.quantum_info import Statevector, DensityMatrix
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

# sums states depending on the first and last bit
def condense_count(mat):
    result = np.array([[0.0, 0.0], [0.0, 0.0]])

    for bitstr in mat:
        if bitstr[0] == '0' and bitstr[-1] == '0':
            result[0][0] += mat[bitstr]
        elif bitstr[0] == '0' and bitstr[-1] == '1':
            result[0][1] += mat[bitstr]
        elif bitstr[0] == '1' and bitstr[-1] == '0':
            result[1][0] += mat[bitstr]
        elif bitstr[0] == '1' and bitstr[-1] == '1':
            result[1][1] += mat[bitstr]

    return result

def get_ideal_prob(sim, qc):
    return Statevector(sim.run(qc).result().get_statevector(qc)).probabilities_dict()

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
# qc = QuantumCircuit(n,n)
qc00 = QuantumCircuit(n, n)
qc01 = QuantumCircuit(n, n)
qc10 = QuantumCircuit(n, n)
qc11 = QuantumCircuit(n, n)

# set circuit initial states

qc01.x(n - 1)

qc10.x(0)

qc11.x(0)
qc11.x(n - 1)

# long_range_cnot(qc, 0, n - 1)
long_range_cnot(qc00, 0, n - 1)
long_range_cnot(qc01, 0, n - 1)
long_range_cnot(qc10, 0, n - 1)
long_range_cnot(qc11, 0, n - 1)

# show circuit diagram
# qc.draw("mpl") # Figure 1

# prepare simulator
aer_ideal = AerSimulator()

# ideal results
# qc.save_statevector()
qc00.save_statevector()
qc01.save_statevector()
qc10.save_statevector()
qc11.save_statevector()

# ideal_result = aer_ideal.run(qc).result()
# statevector = ideal_result.get_statevector(qc)
# ideal_prob = Statevector(statevector).probabilities_dict()
ideal_prob00 = get_ideal_prob(aer_ideal, qc00)
ideal_prob01 = get_ideal_prob(aer_ideal, qc01)
ideal_prob10 = get_ideal_prob(aer_ideal, qc10)
ideal_prob11 = get_ideal_prob(aer_ideal, qc11)


# print("\nProbabilities:")
# print({x: ideal_prob[x] for x, val in ideal_prob.items()})
# plot_histogram(prob_dict) # Figure 2

# ideal_mat = condense_count(ideal_prob)
ideal_mat00 = condense_count(ideal_prob00)
ideal_mat01 = condense_count(ideal_prob01)
ideal_mat10 = condense_count(ideal_prob10)
ideal_mat11 = condense_count(ideal_prob11)

print("Ideal probabilities:")
# print(ideal_mat)
print(ideal_mat00)
print(ideal_mat01)
print(ideal_mat10)
print(ideal_mat11)


# measure our teleported CNOT control and target
# qc.measure(0, 0)
# qc.measure(7, 7)
qc00.measure(0, 0)
qc00.measure(7, 7)
qc01.measure(0, 0)
qc01.measure(7, 7)
qc10.measure(0, 0)
qc10.measure(7, 7)
qc11.measure(0, 0)
qc11.measure(7, 7)

# noisy results
backend = GenericBackendV2(num_qubits=8)
noise_model = NoiseModel.from_backend(backend)
aer_noisy = AerSimulator(noise_model=noise_model)

shots = 1000
# result = aer_noisy.run(qc, shots=1000).result()
# counts = result.get_counts()

counts00 = aer_noisy.run(qc00, shots=shots).result().get_counts()
counts01 = aer_noisy.run(qc01, shots=shots).result().get_counts()
counts10 = aer_noisy.run(qc10, shots=shots).result().get_counts()
counts11 = aer_noisy.run(qc11, shots=shots).result().get_counts()

# print("\nWith noise:")
# print(counts)
# plot_histogram(counts) # Figure 3

# discard if flag qubits are 1
flag_qubits = [2, 4, 5]

# filtered_counts = filter_states(counts, flag_qubits)
filtered_counts00 = filter_states(counts00, flag_qubits)
filtered_counts01 = filter_states(counts01, flag_qubits)
filtered_counts10 = filter_states(counts10, flag_qubits)
filtered_counts11 = filter_states(counts11, flag_qubits)

# print("\nFiltered counts:")
# print(filtered_counts)
# plot_histogram(filtered_counts) # Figure 4

# condense to only look at target and source end state
# noisy_mat = condense_count(filtered_counts)
noisy_mat00 = condense_count(filtered_counts00)
noisy_mat01 = condense_count(filtered_counts01)
noisy_mat10 = condense_count(filtered_counts10)
noisy_mat11 = condense_count(filtered_counts11)

# print(noisy_mat)
print(noisy_mat00)
print(noisy_mat01)
print(noisy_mat10)
print(noisy_mat11)


# fidelity = np.trace(density.transpose() @ ideal_density)
# fidelity = np.trace((noisy_mat.transpose() / np.sum(noisy_mat)) @ ideal_mat)
fidelity00 = np.trace((noisy_mat00.transpose() / np.sum(noisy_mat00)) @ ideal_mat00)
fidelity01 = np.trace((noisy_mat01.transpose() / np.sum(noisy_mat01)) @ ideal_mat01)
fidelity10 = np.trace((noisy_mat10.transpose() / np.sum(noisy_mat10)) @ ideal_mat10)
fidelity11 = np.trace((noisy_mat11.transpose() / np.sum(noisy_mat11)) @ ideal_mat11)

# print("Fidelity: " + str(fidelity))
print("Fidelity[00]: " + str(fidelity00))
print("Fidelity[01]: " + str(fidelity01))
print("Fidelity[10]: " + str(fidelity10))
print("Fidelity[11]: " + str(fidelity11))

pyplot.show()

# TODO:
# [DONE] add error detection (detect invalid states)
# [DONE] get the ideal matrix (matrix of the ideal probabilities)
# [DONE] get the experimental matrix (create a matrix to compare against the ideal)
# [DONE] quation 23 from ac/dc paper (measures fidelity)
# [DONE] simple truth table tomography (using |00>, |01>, |10>, and |11>)

# make quantum circuits a list (to iter instead of repeat)
# fix the loop to work with an arbitrary amount of qubits
