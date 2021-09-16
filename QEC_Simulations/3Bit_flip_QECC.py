from typing import Counter
from unicodedata import name
from qiskit import *
from qiskit import Aer, transpile
from qiskit.providers.aer import AerSimulator, QasmSimulator
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.visualization import plot_histogram, plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere, plot_bloch_vector, plot_state_city, plot_bloch_multivector
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.monitor import job_monitor
from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit import assemble
from qiskit.quantum_info import Statevector,partial_trace
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
import numpy as np
import matplotlib.pyplot as plt 
from qiskit.providers.aer.noise import NoiseModel
import qiskit.quantum_info as qi
from qiskit.quantum_info import Kraus, state_fidelity, DensityMatrix,SuperOp
from qiskit import IBMQ

aer_sim = Aer.get_backend('aer_simulator')
sv_sim = Aer.get_backend('statevector_simulator')



def addErrorGate(qubit, prob, error_gates):
    n = len(error_gates)
    
    # Check to make sure n*prob<=1
    if(n*prob > 1):
        raise ValueError('The entered probability is invalid, {}*prob > 1'.format(n))
    
    rand = np.random.uniform(0,1)
    #print(rand) # Debugging line
    
    # Apply error_gate[i] if the randomly generated number is between i*prob and (i+1)*prob
    for i in range(n):
        if rand < (i+1)*prob:
            error_gates[i](qubit)
            return

#######
# Create the quantum circuit with the initial state
#######
def Circuit(p_error, err=0):
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
    if err==0:
        qc_3qx = QuantumCircuit(3)
        k=2
        anc=QuantumRegister(k, 'auxiliary') ### Ancilla qubit
        qc_3qx.add_register(anc)
        cr=ClassicalRegister(k, 'syndrome') ### Classical register for syndrome extraction
        qc_3qx.add_register(cr)
        sv = execute(qc_3qx,sv_sim).result().get_statevector()
        ###Encoding procedure
        qc_3qx.cx(0,1)
        qc_3qx.cx(0,2)
        qc_3qx.barrier()

        #qc_3qx.append(qe,[0,1,2])
        #qc_3qx.draw('mpl')
        #plt.show()

        #####
        #Create the error model
        #####
        # bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
        # bit_flip1=bit_flip.tensor(bit_flip)
        # bit_flip2=bit_flip1.tensor(bit_flip)
        # #print(bit_flip2)
        # bit_flip_kraus = Kraus(bit_flip2)
        # #print("hey kraus",bit_flip_kraus)
        # ######
        # #Set error basis
        # ######
        # noise_model = NoiseModel(['unitary'])
        # noise_model.add_all_qubit_quantum_error(bit_flip2, 'error')

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
        # plot_histogram(counts, sort="hamming", target_string="000")
        # plt.show()
        for i in range(0, 3):
            addErrorGate(i,p_error,[qc_3qx.x])
        #qc_3qx.draw('mpl')
        #plt.show()
        # count=0
        # rand = np.random.uniform(0,1)
        # if rand < p_error:
        #     qc_3qx.x(0)
        # if rand < p_error**2:
        #     qc_3qx.x(1)
        # if rand < p_error**3:
        #     qc_3qx.x(2)
        #######
        #Create the 3bi-flip error correction code 
        #######
        # sv_sim = Aer.get_backend('statevector_simulator')
        # result = execute(qc_3qx, sv_sim, noise_model=noise_model).result()
        # sv = result.get_statevector()

        qc_3qx.barrier()
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
        qc_3qx.x(0).c_if(cr, 1)

         ###condition is in binary
        qc_3qx.x(3).c_if(cr, 1)
        qc_3qx.x(1).c_if(cr, 3)
        qc_3qx.x(3).c_if(cr, 3)
        qc_3qx.x(4).c_if(cr, 3)
        qc_3qx.x(2).c_if(cr, 2)
        qc_3qx.x(4).c_if(cr, 2)
        qc_3qx.barrier()
        ###
        #Decoding
        ###
        qc_3qx.cx(0,2)
        qc_3qx.cx(0,1)
        qc_3qx.barrier()



        # ###
        # # Simulation
        # ###
        #qc_3qx.draw('mpl', scale=0.8, filename='3bitflipcode.png')
        cr2=ClassicalRegister(3, 'outcome')
        qc_3qx.add_register(cr2)
        qc_3qx.measure([0,1,2],[2,3,4])
        sv2= execute(qc_3qx,sv_sim).result().get_statevector()
    else:
        qc_3qx=QuantumCircuit(3)
        sv = execute(qc_3qx,sv_sim).result().get_statevector()
        rand = np.random.uniform(0,1)
        if rand < p_error:
             qc_3qx.x(0)
        sv2= execute(qc_3qx,sv_sim).result().get_statevector()

    return qc_3qx , sv , sv2
# counts=execute(qc_3qx, backend = sv_sim, optimization_level=0, noise_model= noise_model, shots=10000).result().get_statevector()
#counts=execute(Circuit(0.1)[0], backend = sv_sim, optimization_level=0, shots=10000).result().get_statevector()
#print(counts)
N=300
lp=20
probs = np.linspace(0,1,lp)
phi = [1, 0]


counter_on= np.zeros(lp)
counter_off= np.zeros(lp)
for n , prob in enumerate(probs):
    print("n:", n +1, "prob:", prob)
    for i in range(N):
        qc_err_on = Circuit(prob,0)
        qc_err_off = Circuit(prob,1)
        if np.array_equal(qc_err_on[1],qc_err_on[2]):
            counter_on[n] += 1 
        if np.array_equal(qc_err_off[1],qc_err_off[2]):
            counter_off[n] += 1 
    counter_on[n] = counter_on[n]/N
    counter_off[n] = counter_off[n]/N


expected = 1 - 3*(probs*probs) + 2*(probs*probs*probs)
expected2 = 1-probs
fig = plt.figure(figsize=(12,6))

plt.title('Fidelity for Circuit')
#plt.plot(probs,F,label='Measured')
plt.plot(probs,counter_on, label='Simulated with QECC')
plt.plot(probs,expected,label='Exptected with QECC')
plt.plot(probs,counter_off, label='Measured with no error correction')
plt.plot(probs,expected2, label = 'Expected with no error correction')
plt.xlabel('p')
plt.ylabel('Fidelity')
plt.ylim([0,1.05])

plt.legend()
plt.show()

###Good plots
fig = plt.figure(figsize=(12,6))
plt.title('Fidelity without error correction')
plt.plot(probs,counter_off, label='Simulated with no error correction')
plt.plot(probs,expected2, label = 'Expected with no error correction')
plt.xlabel('p')
plt.ylabel('Fidelity')
plt.ylim([0,1.05])
plt.legend()
plt.savefig('Fidelity_no_correction.png')
plt.show()


fig = plt.figure(figsize=(12,6))
plt.title('Fidelity with error correction')
plt.plot(probs,counter_on, label='Simulated with QECC')
plt.plot(probs,expected,label='Exptected with QECC')
plt.plot(probs,expected2, label = 'Expected with no error correction')
plt.xlabel('p')
plt.ylabel('Fidelity')
plt.ylim([0,1.05])
plt.legend()
plt.savefig('Fidelity_yes_correction.png')
plt.show()


