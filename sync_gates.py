import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.qip.circuit import *

from copy import deepcopy


def U(args):
    """Custom Implementation of QASM U Gate since QuTip's is different
    See QASM Docs / Koppenhofer Thesis
    """
    theta, phi, gamma = args
    return Qobj(
        [
            [np.cos(theta / 2), -np.exp(1j * gamma) * np.sin(theta / 2)],
            [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + gamma)) * np.cos(theta / 2)],
        ]
    )


# Unitary Evolution


def Uevo(args):
    detuning, signal_strength, dt = args
    Rz = (1j * detuning / 2 * sigmaz() / 2 * dt).expm()
    Ry = (1j * 2 * signal_strength * sigmay() / 2 * dt / 2).expm()
    return Rz * Ry * Rz


# Dispitive Gates


def CU(args, N=2, control=0, target=1, control_value=1):
    return controlled_gate(U(args), N, control, target, control_value)


# Other Operations


def reset(reg, bit, new):
    if reg.type == "ket":
        reg = ket2dm(reg)
    """Reset the qubit at position bit of reg with dm new"""
    traces = [reg.ptrace(i) for i in range(int(np.log2(reg.shape[0])))]

    traces[bit] = ket2dm(new) if new.type == "ket" else new
    return tensor(traces)


# Circuit


def add_unitary_evo_gates(qc, detuning, signal_strength, dt):
    # qc.add_gate("RZ", arg_value=-detuning * dt / 2, targets=0)
    # qc.add_gate("RY", arg_value=-signal_strength * dt, targets=0)
    # qc.add_gate("RZ", arg_value=-detuning * dt / 2, targets=0)

    phi = gamma = -detuning * dt / 2
    theta = -signal_strength * dt
    qc.add_gate("U", arg_value=[theta, phi, gamma], targets=0)

    return qc


def add_loss_gates(qc, theta_d):
    # qc.add_gate("CU", arg_value=[theta_d, 0, 0])
    # qc.add_gate("CNOT", targets=[0], controls=[1])

    qc.add_gate("U", targets=0, arg_value=[np.pi / 2, -np.pi, 0])
    qc.add_gate("U", targets=1, arg_value=[-theta_d / 2, -np.pi / 2, np.pi])
    qc.add_gate("CNOT", controls=1, targets=0)
    qc.add_gate("U", targets=0, arg_value=[np.pi / 2, -np.pi / 2, 0])
    qc.add_gate("U", targets=1, arg_value=[-theta_d / 2, np.pi, np.pi / 2])
    qc.add_gate("CNOT", controls=1, targets=0)
    qc.add_gate("U", targets=0, arg_value=[0, 0, -np.pi / 2])
    qc.add_gate("U", targets=1, arg_value=[0, 0, -np.pi / 2])

    return qc


def add_gain_gates(qc, theta_g):
    qc.add_gate("U", targets=0, arg_value=[-np.pi, 0, 0])
    qc.add_gate("CNOT", targets=[1], controls=[0])
    qc.add_gate("CU", arg_value=[theta_g, 0, 0])
    qc.add_gate("CNOT", targets=[1], controls=[0])
    qc.add_gate("U", targets=0, arg_value=[np.pi, 0, 0])

    return qc


def unitary_circuit(detuning, signal_strength, dt):
    qc = QubitCircuit(2, user_gates={"U": U})
    add_unitary_evo_gates(qc, detuning, signal_strength, dt)

    return qc


def unitary_and_damping_circuit(detuning, signal_strength, dt, theta):
    def CUc0t1(args):
        return CU(args)

    qc = QubitCircuit(2, user_gates={"U": U, "CU": CUc0t1}, num_cbits=1)

    add_unitary_evo_gates(qc, detuning, signal_strength, dt)

    qc.add_gate("CU", arg_value=[theta, 0, 0])
    qc.add_gate("CNOT", targets=[0], controls=[1])
    qc.add_measurement("M", [1], classical_store=0)

    return qc


def dissipative_circuits(theta_d, theta_g):
    """Create QuTip quantum circuits for unitary evolution and dissipation
    Split into two circuits to allow for resets."""

    def CUc0t1(args):
        return CU(args)

    # Loss
    loss_qc = QubitCircuit(2, user_gates={"U": U, "CU": CUc0t1}, num_cbits=1)
    add_loss_gates(loss_qc, theta_d)
    loss_qc.add_measurement("M1", [1], classical_store=0)

    # Gain
    gain_qc = QubitCircuit(2, user_gates={"U": U, "CU": CUc0t1}, num_cbits=1)
    add_gain_gates(gain_qc, theta_g)
    gain_qc.add_measurement("M2", [1], classical_store=0)

    return loss_qc, gain_qc


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


def add_tomography(circuit, nqubits, main, ancilla):
    """Qiskit: Add tomography circuits with measurments on existing classical register"""
    # This function add gates to the end of circuits but the measurments add to a new classical register.
    # This additional register messes with noise mitigation so the next function adds the same gates but
    # the classical register is reused.

    for i in range(3 if nqubits == 1 else 9):
        c = deepcopy(circuit)
        qreg = c.qregs[0]
        creg = c.cregs[0]

        c.barrier()
        if i == 0:
            c.h(qreg[main])
            c.name = "('X',)"  # Need to addjust names for StateTomographyFitter
            if nqubits == 2:
                c.h(qreg[ancilla])
                c.name = "('X', 'X')"
        elif i == 1:
            if nqubits == 1:
                c.sdg(qreg[main])
                c.h(qreg[main])
                c.name = "('Y',)"
            elif nqubits == 2:
                c.h(qreg[main])
                c.sdg(qreg[ancilla])
                c.measure(qreg[main], creg[0])
                c.h(qreg[ancilla])
                c.name = "('X', 'Y')"
        elif i == 2:
            c.name = "('Z',)"
            if nqubits == 2:
                c.h(qreg[main])
                c.name = "('X', 'Z')"
        elif i == 3:
            c.name = "('Y', 'X')"
            c.sdg(qreg[main])
            c.h(qreg[main])
            c.h(qreg[ancilla])
        elif i == 4:
            c.name = "('Y', 'Y')"
            c.sdg(qreg[main])
            c.sdg(qreg[ancilla])
            c.h(qreg[main])
            c.h(qreg[ancilla])
        elif i == 5:
            c.name = "('Y', 'Z')"
            c.sdg(qreg[main])
            c.h(qreg[main])
        elif i == 6:
            c.name = "('Z', 'X')"
            c.measure(qreg[main], creg[0])
            c.h(qreg[ancilla])
        elif i == 7:
            c.name = "('Z', 'Y')"
            c.measure(qreg[main], creg[0])
            c.sdg(qreg[ancilla])
            c.h(qreg[ancilla])
        elif i == 8:
            c.name = "('Z', 'Z')"

        if i != 1 and i != 6 and i != 7:
            c.measure(qreg[main], creg[0])
        if nqubits == 2:
            c.measure(qreg[ancilla], creg[1])

        yield c
