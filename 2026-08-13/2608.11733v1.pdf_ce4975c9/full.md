# A 12-CNOT Double Qubit Excitation Gate

Irfansha Shaik # Kvantify Aps, Copenhagen, Denmark

## Abstract

Efective implementation of high-level quantum gates is essential for practical quantum computing. To the best of our knowledge, we present the first reported 12-CNOT decomposition of the double qubit excitation operator, improving upon state-of-the-art (SOTA) implementations with 13 CNOTs. Our new circuit has the lowest CNOT count (12), lowest CNOT depth (10), and lowest total circuit depth (16) among all the previous SOTA circuits. Further, we only added 2 extra one-qubit gates compared to the lowest one-qubit gate count (11) among the previous SOTA circuits.

## 1 Introduction

Efective implementation of important building blocks (such as high-level quantum gates) in quantum algorithms is essential for practical quantum computing. In this work, we look at one such high-level gate, the Double Qubit Excitation Operator, which implements the equation (1) (Eq. 20 of [21]):

$$
\begin{array} { r l } & { U _ { k l i j } ( \theta ) = \exp \Big [ - \frac { i \theta } { 8 } \big ( X _ { i } Y _ { j } X _ { k } X _ { l } + Y _ { i } X _ { j } X _ { k } X _ { l } + Y _ { i } Y _ { j } Y _ { k } X _ { l } + Y _ { i } Y _ { j } X _ { k } Y _ { l } } \\ & { \qquad - X _ { i } X _ { j } Y _ { k } X _ { l } - X _ { i } X _ { j } X _ { k } Y _ { l } - Y _ { i } X _ { j } Y _ { k } Y _ { l } - X _ { i } Y _ { j } Y _ { k } Y _ { l } \big ) \Big ] . } \end{array}\tag{1}
$$

Using the computational-basis ordering $| q _ { i } q _ { j } q _ { k } q _ { l } \rangle$ in a 4-qubit system, the operator implements a continuous rotation between the two states |0011⟩ and |1100⟩, as shown in (2):

$$
U ( \theta ) \left| x \right. = \left\{ \begin{array} { l l } { \cos \theta \left| 0 0 1 1 \right. + \sin \theta \left| 1 1 0 0 \right. , } & { \left| x \right. = \left| 0 0 1 1 \right. , } \\ { \cos \theta \left| 1 1 0 0 \right. - \sin \theta \left| 0 0 1 1 \right. , } & { \left| x \right. = \left| 1 1 0 0 \right. , } \\ { \left| x \right. , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{2}
$$

The double qubit excitation gate is used as a building block for several quantum algorithms, both near-term and fault-tolerant. In near-term algorithms, for example, it is used as a CNOT eficient alternative to the fermionic double excitation operator in variational ground-state ansätze such as unitary coupled cluster (UCCSD) [9, 10] in the Variational Quantum Eigensolver (VQE) and its adaptive variants such as QEB-ADAPT-VQE [20] and FAST-VQE [5]. In fault-tolerant algorithms, it is used in Trotterized Hamiltonian simulation and time evolution of the electronic-structure Hamiltonian [18]. Further, one can use the operator in state preparation, such as a UCC-type ansatz, for subsequent ground-state energy estimation in fault-tolerant algorithms like quantum phase estimation (QPE) [2].

![](images/9ff1bf815462e06aacf19a99ff0873408503da25212bd14eddb5d3b2bf458b60.jpg)  
Figure 1 High-level double-excitation operator.

One can implement the double excitation gate simply using a triple controlled $R _ { y }$ rotation and some CNOT gates. Figure 1 shows such a high-level 4-qubit circuit with triple controlled $R _ { y } ( 2 \theta )$ rotation with $q _ { i }$ as the target qubit, sandwiched between L and R CNOT circuits. We refer to Yordanov et al. [21] for extended explanation on double qubit excitation operators, which is beyond the scope of this work. In this paper, we present, to the best of our knowledge, the first reported decomposition of the double-excitation operator with 12 CNOTs, improving on the previously reported SOTA implementations with 13 CNOTs.

The structure of the rest of the paper is as follows. In Subsection 1.1, we will provide more details on the construction of the high-level circuit, including the role of L and R CNOT circuits. We also provide a simple 14-CNOT gate implementation using so-called Gray-code expansion [6, 15] of the $C ^ { 3 } R _ { y } ( 2 \theta )$ rotation. In Subsection 1.2, we will present the current SOTA implementations of the double excitation operator with 13-CNOTs. Finally in Section 2, we present a new 12-CNOT circuit for the double excitation operator. Our new circuit is better in 3 diferent metrics compared to existing SOTA circuits, i.e., in CNOT count (12), CNOT depth (10), and circuit depth (16).

## 1.1 A 14-CNOT baseline decomposition using Gray-code expansion

Recall that the double-excitation operator implements 8 Pauli strings, as shown in (1). One can naively implement the double-excitation operator of 48 CNOTs, implementing each of the 8 Pauli strings separately using 6 CNOT gates per string (see §4.7.3 and Fig. 4.19 of [8]). As discussed earlier, one can also implement the double-excitation operator using a single $C ^ { 3 } R _ { y } ( 2 \theta )$ rotation and some CNOT gates as shown in Figure 2. Intuitively, we want the controlled rotation to trigger only when the input state is either |0011⟩ or |1100⟩. Precisely, the L CNOT circuit performs this required transformation, as is defined in (3):

$$
L \left| x \right. = { \left\{ \begin{array} { l l } { | 0 1 1 1 \rangle , } & { | x \rangle = | 0 0 1 1 \rangle , } \\ { | 1 1 1 1 \rangle , } & { | x \rangle = | 1 1 0 0 \rangle , } \\ { | x ^ { \prime } \rangle { \mathrm { ~ w i t h ~ } } q _ { j } q _ { k } q _ { l } \neq 1 1 1 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }\tag{3}
$$

The R CNOT circuit, on the other hand, performs the inverse transformation of L, i.e., $L R = I .$ Depending on the θ, controlled rotation $C ^ { 3 } R _ { y } ( 2 \theta )$ either flips the input basis states |0011⟩ and |1100⟩ or leaves them unchanged. Thus, all the input states excluding |0011⟩ and |1100⟩ stay untouched in the output, implementing our desired double-excitation operation. Decomposition of the triple-controlled $R _ { y }$ rotation is well studied, and multiple 8-CNOT constructions are known. Here, we use the Gray-code expansion of the $C ^ { 3 } R _ { y } ( 2 \theta )$ rotation [6, 15], with 8 CNOTs and 8 $R _ { y }$ rotations, resulting in a baseline 14-CNOT doubleexcitation gate as shown in Figure 2.

![](images/c6b29ffe728cba1bec8c61db8c31b5fb115688953c635e82747ae46d590b7e63.jpg)  
Figure 2 A 14-CNOT decomposed double-excitation circuit with $C ^ { 3 } R _ { y } ( 2 \theta )$ Gray-code expansion.

## 1.2 Previous SOTA 13-CNOT double-excitation gate implementations

In the literature, several 13-CNOT decompositions of the double-excitation operator have been proposed. The usual approach is to find rewrite rules to absorb some of the CNOTs in the L and R circuits into the various decompositions of the $C ^ { 3 } R _ { y } ( 2 \theta )$ rotation, resulting in 13-CNOT circuits. Later in Table 1, we refer to some well-known examples of these 13-CNOT circuits of the double-excitation operator, and their metrics. In this Subsection, we present two of the best 13-CNOT circuits from the literature. The 13-CNOT circuit by Yordanov

![](images/dc55e14f63a3efc0c164ddbe91931db285936ccfabf8e06de743d6f6fa2e3649.jpg)  
Figure 3 A 13-CNOT double-excitation gate by Yordanov et al. [21, 20] with lowest CNOT depth (11) cf. Table 1. The two highlighted gates on q<sub>k</sub> are added to correct the original circuit.

![](images/22caf0dfd279524739ec040790c6882b5936db8ef0b227a6f0d4896bee591448.jpg)  
Figure 4 Nam [7] double-excitation circuit with 13 CNOTs. Among the 13-CNOT reference circuits it has the lowest one-qubit gate count (11). Cf. Table 1.

et al. [21, 20] as shown in Figure 3 has the lowest 11 CNOT depth but at the cost of 16 one-qubit gates. Figure 4 on the other hand shows a 13-CNOT circuit by Nam [7] with the lowest one-qubit gate count (11) but at the cost of 13 CNOT depth. Another 13-CNOT circuit by Wang [18] is reported in the Table 1 with 15 one-qubit gates and 13 CNOT depth.

## 2 A 12-CNOT decomposition of the double-excitation operator

Although several 13-CNOT decompositions of the double-excitation operator have been proposed in the literature, our literature search found no previously reported implementation with fewer than 13 CNOTs. In this section, we present the first 12-CNOT decomposition of the double-excitation operator, as shown in Figure 5. We have used various circuit synthesis and optimization tools such as Q-Synth [11], Qiskit Transpiler [3], Tket [16] to explore diferent decompositions of the double-excitation operator. Various synthesis techniques such as Cliford Synthesis [13, 14], CNOT+Rz Synthesis [12, 4], KAK decomposition [17] etc. have been useful to extensively explore diferent decompositions (mainly for excluding dead ends). Table 1 compares our new 12-CNOT circuit with the previous SOTA 13-CNOT circuits, in 4 diferent metrics. Our new circuit has the lowest CNOT count (12), lowest CNOT depth (10), and lowest total circuit depth (16) among all the previous SOTA circuits.

![](images/38866a3339da26cb5d2d721432e9baa6f56f452697e389fdff21e22fff28e7a9.jpg)  
Figure 5 Our best double-excitation circuit: 12 CNOTs (same-control CNOTs drawn compactly as fan-outs, but expanded when computing depth), CX-depth 10, 13 one-qubit gates. Cf. Table 1.

Table 1 CNOT count, CNOT depth, single-qubit gate count, and total circuit depth for doubleexcitation circuit implementations. Single-qubit gates are counted as universal one-qubit (u3) gates, i.e., each maximal run of consecutive single-qubit gates on a qubit is merged into one u3 gate.
<table><tr><td>Circuit</td><td>CX count CX depth</td><td>1q gates</td><td>Depth</td></tr><tr><td>Naive Pauli decomposition [8] 48</td><td>48</td><td>38</td><td>64</td></tr><tr><td>Baseline: with Gray-code (Fig. 2) [6, 15] 14</td><td>14</td><td>8</td><td>22</td></tr><tr><td>PennyLane 2021 [1, 19] 14</td><td>12</td><td>14</td><td>20</td></tr><tr><td>Wang 2021 [18] 13</td><td>13</td><td>15</td><td>22</td></tr><tr><td>Nam 2020 [7] 13</td><td>13</td><td>11</td><td>22</td></tr><tr><td>Yordanov 2020 [21, 20] 13</td><td>11</td><td>16</td><td>20</td></tr><tr><td>This work (Fig. 5) 12</td><td>10</td><td>13</td><td>16</td></tr></table>

## 3 Conclusion

In this work, we presented, to the best of our knowledge, the first reported 12-CNOT decomposition of the double qubit excitation operator. We compared our new circuit with the previous SOTA 13-CNOT circuits in 4 diferent metrics. Our new circuit has the lowest CNOT count (12), lowest CNOT depth (10), and lowest total circuit depth (16) among all the previous SOTA circuits. Further, we only added 2 extra one-qubit gates compared to the lowest one-qubit gate count (11) among the previous SOTA circuits.

## Acknowledgements

Author would like to thank Søren Fuglede Jørgensen for the introduction to the problem and the discussions on the topic. Further, author would also like to thank Jaco van de Pol for the corrections on Yordanov’s original circuit in Figure 3. LLMs have been used, mainly Opus 4.8, for setting up experiments, brainstorming, generating Latex figures and Tables, and minor editorial tasks. Finally, author would like to thank Patrick Ettenhuber and Asbjørn Frost Teilmann for feedback and correctness checks.

## References

1 Gian-Luca R. Anselmetti, David Wierichs, Christian Gogolin, and Robert M. Parrish. Local, expressive, quantum-number-preserving VQE ansätze for fermionic systems. New Journal of Physics, 23(11):113010, 2021. arXiv:2104.05695, doi:10.1088/1367-2630/ac2cb3.

2 Stepan Fomichev, Kasra Hejazi, Modjtaba Shokrian Zini, Matthew Kiser, Joana Fraxanet, Pablo Antonio Moreno Casares, Alain Delgado, Joonsuk Huh, Arne-Christian Voigt, Jonathan E. Mueller, and Juan Miguel Arrazola. Initial state preparation for quantum

chemistry on quantum computers. PRX Quantum, 5(4):040339, 2024. arXiv:2310.18410, doi:10.1103/PRXQuantum.5.040339.

3 Ali Javadi-Abhari, Matthew Treinish, Kevin Krsulich, Christopher J. Wood, Jake Lishman, Julien Gacon, Simon Martiel, Paul D. Nation, Lev S. Bishop, Andrew W. Cross, Blake R. Johnson, and Jay M. Gambetta. Quantum computing with Qiskit, 2024. arXiv:2405.08810, doi:10.48550/arXiv.2405.08810.

4 Xinpeng Li, Ji Liu, Shuai Xu, Paul Hovland, and Vipin Chaudhary. HOPPS: Hardware-aware optimal phase polynomial synthesis with blockwise optimization for quantum circuits. In 2025 IEEE 32nd International Conference on High Performance Computing, Data, and Analytics (HiPC), 2025. arXiv:2511.18770, doi:10.1109/HIPC66333.2025.00010.

5 Marco Majland, Patrick Ettenhuber, and Nikolaj Thomas Zinner. Fermionic adaptive sampling theory for variational quantum eigensolvers. Physical Review A, 108(5):052422, 2023. arXiv: 2303.07417, doi:10.1103/PhysRevA.108.052422.

6 Mikko Möttönen, Juha J. Vartiainen, Ville Bergholm, and Martti M. Salomaa. Quantum circuits for general multiqubit gates. Physical Review Letters, 93(13):130502, 2004. arXiv: quant-ph/0404089, doi:10.1103/PhysRevLett.93.130502.

7 Yunseong Nam, Jwo-Sy Chen, Neal C. Pisenti, Kenneth Wright, Conor Delaney, Dmitri Maslov, Kenneth R. Brown, Stewart Allen, Jason M. Amini, Joel Apisdorf, Kristin M. Beck, Aleksey Blinov, Vandiver Chaplin, Mika Chmielewski, Coleman Collins, Shantanu Debnath, Kai M. Hudek, Andrew M. Ducore, Matthew Keesan, Sarah M. Kreikemeier, Jonathan Mizrahi, Phil Solomon, Mike Williams, Jaime David Wong-Campos, David Moehring, Christopher Monroe, and Jungsang Kim. Ground-state energy estimation of the water molecule on a trapped-ion quantum computer. npj Quantum Information, 6(1):33, 2020. arXiv:1902.10171, doi:10.1038/s41534-020-0259-3.

8 Michael A. Nielsen and Isaac L. Chuang. Quantum Computation and Quantum Information: 10th Anniversary Edition. Cambridge University Press, 10th anniversary edition, 2010. doi: 10.1017/CBO9780511976667.

9 Alberto Peruzzo, Jarrod McClean, Peter Shadbolt, Man-Hong Yung, Xiao-Qi Zhou, Peter J. Love, Alán Aspuru-Guzik, and Jeremy L. O’Brien. A variational eigenvalue solver on a photonic quantum processor. Nature Communications, 5:4213, 2014. arXiv:1304.3061, doi:10.1038/ncomms5213.

10 Jonathan Romero, Ryan Babbush, Jarrod R. McClean, Cornelius Hempel, Peter J. Love, and Alán Aspuru-Guzik. Strategies for quantum computing molecular energies using the unitary coupled cluster ansatz. Quantum Science and Technology, 4(1):014008, 2018. arXiv: 1701.02691, doi:10.1088/2058-9565/aad3e4.

11 Irfansha Shaik and Jaco van de Pol. Optimal layout synthesis for quantum circuits as classical planning. In 2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD), San Francisco, California, USA, 2023. IEEE/ACM. doi:10.1109/ICCAD57390.2023.10323924.

12 Irfansha Shaik and Jaco van de Pol. Optimal layout-aware CNOT circuit synthesis with qubit permutation. In 27th European Conference on Artificial Intelligence (ECAI 2024), Santiago de Compostela, Spain, 2024. IOS Press. doi:10.3233/FAIA240748.

13 Irfansha Shaik and Jaco van de Pol. CNOT-optimal cliford synthesis as SAT. In 28th International Conference on Theory and Applications of Satisfiability Testing (SAT 2025), volume 341 of LIPIcs, pages 28:1–28:21, Glasgow, Scotland, UK, 2025. Schloss Dagstuhl – Leibniz-Zentrum für Informatik. doi:10.4230/LIPIcs.SAT.2025.28.

14 Irfansha Shaik and Jaco van de Pol. Optimal cliford synthesis as planning. Proceedings of the International Conference on Automated Planning and Scheduling, 36(1):266–274, 2026. doi:10.1609/icaps.v36i1.42836.

15 Vivek V. Shende, Stephen S. Bullock, and Igor L. Markov. Synthesis of quantum-logic circuits. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 25(6):1000–1010, 2006. arXiv:quant-ph/0406176, doi:10.1109/TCAD.2005.855930.

16 Seyon Sivarajah, Silas Dilkes, Alexander Cowtan, Will Simmons, Alec Edgington, and Ross Duncan. t|ket⟩: a retargetable compiler for NISQ devices. Quantum Science and Technology, 6(1):014003, 2021. arXiv:2003.10611, doi:10.1088/2058-9565/ab8e92.

17 Robert R. Tucci. An introduction to Cartan’s KAK decomposition for QC programmers, 2005. arXiv:quant-ph/0507171, doi:10.48550/arXiv.quant-ph/0507171.

18 Qingfeng Wang, Ming Li, Christopher Monroe, and Yunseong Nam. Resource-optimized fermionic local-hamiltonian simulation on a quantum computer for quantum chemistry. Quantum, 5:509, 2021. arXiv:2004.04151, doi:10.22331/q-2021-07-26-509.

19 Xanadu. PennyLane qml.doubleexcitation operation. https://docs.pennylane.ai/en/ stable/code/api/pennylane.DoubleExcitation.html, 2024. Accessed 2026-08-06.

20 Yordan S. Yordanov, V. Armaos, Crispin H. W. Barnes, and David R. M. Arvidsson-Shukur. Qubit-excitation-based adaptive variational quantum eigensolver. Communications Physics, 4(1):228, 2021. doi:10.1038/s42005-021-00730-0.

21 Yordan S. Yordanov, David R. M. Arvidsson-Shukur, and Crispin H. W. Barnes. Eficient quantum circuits for quantum computational chemistry. Physical Review A, 102(6):062612, 2020. arXiv:2005.14475, doi:10.1103/PhysRevA.102.062612.