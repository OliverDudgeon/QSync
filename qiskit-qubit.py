from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(2, "q")
creg_c = ClassicalRegister(2, "c")
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.u(pi / 2, 0, 0, qreg_q[1])
circuit.cx(qreg_q[1], qreg_q[0])
circuit.measure(qreg_q[1], creg_c[1])
circuit.reset(qreg_q[1])

circuit.draw()
