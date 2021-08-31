from unicodedata import name
from qiskit import *
from qiskit import Aer, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.visualization import plot_histogram, plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere, plot_bloch_vector, plot_state_city, plot_bloch_multivector
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.monitor import job_monitor
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit import assemble
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
import numpy as np
import matplotlib.pyplot as plt 
from qiskit.providers.aer.noise import NoiseModel
import qiskit.quantum_info as qi


aer_sim = Aer.get_backend('aer_simulator')

######
# Create a custum gate that is affected by error 
# (in this case an identity gate on 3 qubits that might fail and apply a x error on one of the qubits)
######
ident3 = qi.Operator([  [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]  ])
qe=QuantumCircuit(3, name='error')
qe.unitary(ident3, [0, 1, 2], label='error')




#######
# Create the quantum circuit with the initial state
#######
qc_3qx = QuantumCircuit(3)
initial_state = [1/np.sqrt(3), np.sqrt(2)/np.sqrt(3)]  
qc_3qx.initialize(initial_state, 0) # Initialize the 0th qubit in the state `initial_state`
qc_3qx.barrier()


###Encoding procedure
qc_3qx.cx(0,1)
qc_3qx.cx(0,2)
qc_3qx.barrier()
qc_3qx.append(qe,[0,1,2])
#qc_3qx.draw('mpl')
#plt.show()

#####
#Create the error model
#####
p_error = 0.1 #Error probability
bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
bit_flip1=bit_flip.tensor(bit_flip)
bit_flip2=bit_flip1.tensor(bit_flip)
print(bit_flip2)

######
#Set error basis
######
noise_model = NoiseModel(['unitary'])
noise_model.add_all_qubit_quantum_error(bit_flip2, 'error')
#######
# Measurement just after the noise, to se the effect of the noise
# uncomment these lines 
#######

# cr2=ClassicalRegister(3, 'outcome')
# qc_3qx.add_register(cr2)
# qc_3qx.measure([0,1,2],[0,1,2])
# qc_3qx.draw('mpl')
# basis_gates = noise_model.basis_gates
# print(basis_gates)
# result = execute(qc_3qx, Aer.get_backend('qasm_simulator'),
#                   noise_model=noise_model,shots=100000).result()
# counts = result.get_counts(0)
# plot_histogram(counts)
# plt.show()


#######
#Create the 3bi-flip error correction code 
#######
k=2
anc=QuantumRegister(k, 'auxiliary') ### Ancilla qubit
qc_3qx.add_register(anc)
cr=ClassicalRegister(k, 'syndrome') ### Classical register for syndrome extraction
qc_3qx.add_register(cr)
qc_3qx.cx(0,3)
qc_3qx.cx(1,3)
qc_3qx.cx(1,4)
qc_3qx.cx(2,4)
qc_3qx.barrier()
qc_3qx.measure(anc[0],cr[0])
qc_3qx.measure(anc[1],cr[1])
qc_3qx.barrier()


###
#Recovery 
###
qc_3qx.x(0).c_if(cr, 1) ###condition is in binary
qc_3qx.x(1).c_if(cr, 3)
qc_3qx.x(2).c_if(cr, 2)
qc_3qx.barrier()

###
# Simulation
###
cr2=ClassicalRegister(3, 'outcome')
qc_3qx.add_register(cr2)
qc_3qx.measure([0,1,2],[2,3,4])
qc_3qx.draw('mpl')
counts=execute(qc_3qx,backend = aer_sim, optimization_level=0, noise_model= noise_model, shots=10000).result().get_counts()
plot_histogram(counts)
plt.show()