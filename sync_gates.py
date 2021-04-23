import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.qip.circuit import *

# Unitary Evolution


def Uevo(args):
    delta, epsilon, dt = args
    Rz = (1j * delta * sigmaz() * dt).expm()
    Ry = (1j * epsilon * sigmay() * dt / 2).expm()
    return Rz * Ry * Rz


# Dispitive Gates


def CU(args, N=2, control=0, target=1, control_value=1):
    return controlled_gate(qasmu_gate(args), N, control, target, control_value)


# Other Operations


def reset(reg, bit, new):
    """Reset the qubit at position bit of reg with dm new"""
    traces = [reg.ptrace(i) for i in range(int(np.log2(reg.shape[0])))]

    traces[bit] = ket2dm(new) if new.type == "ket" else new
    return tensor(traces)


# Circuit


def unitary_circuit(detuning, signal_strength, dt):
    qc = QubitCircuit(2, user_gates={"Uevo": Uevo}, num_cbits=1)
    qc.add_gate("Uevo", arg_value=[detuning, signal_strength, dt])

    return qc


def unitary_and_damping_circuit(detuning, signal_strength, dt, theta):
    def CUc0t1(args):
        return CU(args)

    qc = QubitCircuit(2, user_gates={"CU": CUc0t1, "Uevo": Uevo}, num_cbits=1)

    qc.add_gate("Uevo", arg_value=[detuning, signal_strength, dt])

    qc.add_gate("CU", arg_value=[theta, 0, 0])
    qc.add_gate("CNOT", targets=[0], controls=[1])
    qc.add_measurement("M", [1], classical_store=0)

    return qc


def run_simu(qc, interations, initial_state=basis(2, 0)):
    for _ in range(interations):
        result = qc.run(state=initial_state)
        reset_result = reset(result, 1, basis(2, 0))

        initial_state = reset_result

        yield initial_state


# sim = CircuitSimulator(qc, mode="density_matrix_simulator")

# final = sim.run(initial_state).get_final_states()[0]
# b.add_states(final.ptrace(0))

# for _ in range(10):
#     final = sim.run(final).get_final_states()[0]
#     b.add_states(final.ptrace(0))
