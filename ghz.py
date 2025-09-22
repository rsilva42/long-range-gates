from qiskit import QuantumCircuit
from qiskit.circuit.classical import expr
from qiskit.quantum_info import Statevector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.visualization import plot_histogram

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

import matplotlib.pyplot as pyplot
import numpy as np

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

    circuit.measure(0, 0)
    circuit.measure(target, target)

qc = QuantumCircuit(8, 8)
long_range_cnot(qc, 0, 7)

qc.h(0)
qc.h(7)


# # show circuit diagram
qc.draw("mpl")

# prepare simulator
###############################################################################
aer_ideal = AerSimulator()

qc.save_statevector()
statevector = aer_ideal.run(qc).result().get_statevector(qc)

# print(statevector)

prob_dict = Statevector(statevector).probabilities_dict()
print("\nProbabilities:")
print(prob_dict)

# plot_histogram(prob_dict)

backend = GenericBackendV2(num_qubits=8)
noise_model = NoiseModel.from_backend(backend)
aer_noisy = AerSimulator(noise_model=noise_model)

counts = aer_noisy.run(qc, shots=1000).result().get_counts()
print("With noise:")
print(counts)

# plot_histogram(counts)

# TODO:
# add error detection (detect invalid states)
# get the ideal matrix (matrix of the ideal probabilities)
# get the experimental matrix (create a matrix to compare against the ideal)
# eqution 23 from ac/dc paper (measures fidelity)
# simple truth table tomography (using |00>, |01>, |10>, and |11>)

# fix the loop to work with an arbitrary amount of qubits

pyplot.show()
