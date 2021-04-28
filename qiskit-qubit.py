from qiskit import *
from numpy import pi

qreg_q = QuantumRegister(2, "q")
creg_c = ClassicalRegister(2, "c")
qc = QuantumCircuit(qreg_q, creg_c)

qc.u(pi / 2, 0, 0, qreg_q[1])
qc.cx(qreg_q[1], qreg_q[0])
qc.measure(qreg_q[1], creg_c[1])
qc.reset(qreg_q[1])

qc.draw()
