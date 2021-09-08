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
all_qubits=[0,1,2,3,4,5,6]
qe=QuantumCircuit(7, name='error')
ident3 = qi.Operator(np.identity(2**7))
qe.unitary(ident3, all_qubits, label='error')
#qe.to_instruction()
#######
# Create the quantum circuit with the initial state
#######
qc_3qx = QuantumCircuit(7)
#initial_state = [1/np.sqrt(3), np.sqrt(2)/np.sqrt(3)] 
#initial_state = [1,0] 
#qc_3qx.initialize(initial_state, 0) # Initialize the 0th qubit in the state `initial_state`
qc_3qx.barrier()

###
#Encoding
###
def Encoding7():
    q_encoding=QuantumCircuit(7,name='Enc')
    #Hadamards
    for i in range(1, 4):
        q_encoding.h(i)
    q_encoding.cx(0,5)
    q_encoding.cx(0,6)
    q_encoding.cx(1,0)
    q_encoding.cx(1,4)
    q_encoding.cx(1,5)
    q_encoding.cx(2,0)
    q_encoding.cx(2,4)
    q_encoding.cx(2,6)
    q_encoding.cx(3,4)
    q_encoding.cx(3,5)
    q_encoding.cx(3,6)
    q_encoding.draw('mpl',  filename='Encoding_Seven')
    plt.show()
    return q_encoding
qc_3qx.append(Encoding7(),all_qubits)
qc_3qx.barrier()

qc_3qx.append(qe, all_qubits)
qc_3qx.barrier()
# qc_3qx.draw('mpl', filename='Encoding_Seven')
# plt.show()


# #####
# #Create the error model
# #####
p_error = 0.1 #Error probability
bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
bit_flip1=bit_flip.tensor(bit_flip)
bit_flip2=bit_flip.tensor(bit_flip1)
bit_flip3=bit_flip.tensor(bit_flip2)
bit_flip4=bit_flip.tensor(bit_flip3)
bit_flip5=bit_flip.tensor(bit_flip4)
bit_flip6=bit_flip.tensor(bit_flip5)
#print(bit_flip6)

# ######
# #Set error basis
# ######
noise_model = NoiseModel(['unitary'])
noise_model.add_all_qubit_quantum_error(bit_flip6, 'error')


######
# Measurement just after the noise, to se the effect of the noise
# uncomment these lines 
# #######
# qc_3qx.append(Encoding7().inverse(),all_qubits)
# cr2=ClassicalRegister(7, 'outcome')
# qc_3qx.add_register(cr2)
# qc_3qx.measure(all_qubits,all_qubits)
# qc_3qx.draw('mpl')
# basis_gates = noise_model.basis_gates
# print(basis_gates)
# result = execute(qc_3qx, Aer.get_backend('qasm_simulator'),
#                   noise_model=noise_model,shots=1000).result()
# counts = result.get_counts(0)
#plot_histogram(counts)
#plt.show()


# # # #######
# # # #Create the 3bi-flip error correction code 
# # # #######
k=6
anc=QuantumRegister(k, 'auxiliary') ### Ancilla qubit
qc_3qx.add_register(anc)
for i in range(7,13):
        qc_3qx.h(i)
cr=ClassicalRegister(k, 'syndrome') ### Classical register for syndrome extraction
qc_3qx.add_register(cr)



# # #####
# # #Create the stabilizer
# # #####
stab1Z=QuantumCircuit(5, name='M1') #IIIZZZZ
for i in range(0,stab1Z.num_qubits-1):
    stab1Z.cz(4,i)
stab2Z=QuantumCircuit(5, name='M2') #IZZIIZZ
for i in range(0,stab2Z.num_qubits-1):
    stab2Z.cz(4,i)
stab3Z=QuantumCircuit(5, name='M3') #ZIZIZIZ
for i in range(0,stab3Z.num_qubits-1):
    stab3Z.cz(4,i)
stab1X=QuantumCircuit(5,name='M4') #IIIXXXX
for i in range(0,stab1X.num_qubits-1):
    stab1X.cx(4,i)
stab2X=QuantumCircuit(5, name='M5') #IXXIIXX
for i in range(0,stab2X.num_qubits-1):
    stab2X.cx(4,i)
stab3X=QuantumCircuit(5, name='M6') #XIXIXIX
for i in range(0,stab3X.num_qubits-1):
    stab3X.cx(4,i)
qc_3qx.append(stab1Z,[3,4,5,6,7])
qc_3qx.append(stab2Z,[1,2,5,6,8])
qc_3qx.append(stab3Z,[0,2,4,6,9])
qc_3qx.append(stab1X,[3,4,5,6,10])
qc_3qx.append(stab2X,[1,2,5,6,11])
qc_3qx.append(stab3X,[0,2,4,6,12])
for i in range(7,13):
        qc_3qx.h(i)
qc_3qx.barrier()
qc_3qx.measure(anc,cr)
qc_3qx.barrier()
qc_3qx.draw('mpl',scale=0.5)
#plt.show()
#counts=execute(qc_3qx, aer_sim, shots=10000).result().get_counts()
#plot_histogram(counts)
#plt.show()
# counts=execute(qc_3qx, aer_sim, noise_model= noise_model, shots=10000).result().get_counts()
# plot_histogram(counts)
# plt.show()

# ###
# #Recovery 
# ###
qc_3qx.barrier()
qc_3qx.x(0).c_if(cr, 4) ###condition is in binary
qc_3qx.x(1).c_if(cr, 2)
qc_3qx.x(2).c_if(cr, 6)
qc_3qx.x(3).c_if(cr, 1)
qc_3qx.x(4).c_if(cr, 5)
qc_3qx.x(5).c_if(cr, 3)
qc_3qx.x(6).c_if(cr, 7)
qc_3qx.barrier()
# qc_3qx.draw('mpl')
# plt.show()
# #####
# # Decoding
# #####
qc_3qx.append(Encoding7().inverse(),all_qubits)
cr2=ClassicalRegister(7, 'outcomes')
qc_3qx.add_register(cr2)
qc_3qx.measure(all_qubits,cr2)
#qc_3qx.draw('mpl')
# # counts=execute(qc_3qx, aer_sim, shots=10000).result().get_counts()
# # plot_histogram(counts)
plt.show()

# ###
# # Simulation
# ###
# #qc_3qx.draw('mpl',scale=0.5)
counts=execute(qc_3qx,backend = aer_sim, noise_model= noise_model, shots=10000).result().get_counts()
plot_histogram(counts)
print(counts)
plt.show()
