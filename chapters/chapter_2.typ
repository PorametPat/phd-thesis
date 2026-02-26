#import "@preview/physica:0.9.7": *
#import "@preview/glossarium:0.5.6": gls
#import "@preview/drafting:0.2.2": inline-note
#import "../utils.typ": style, todoneedcite, todoneedwrite
#import "@preview/gentle-clues:1.2.0": *
#import "@preview/drafting:0.2.2": inline-note, margin-note

= Before the Journey Begins #emoji.face.wink <sec:before-thejourney-begins>

Proceeding to the unknown territory without the necessary knowledge is an unwise decision. Thus, we would like to review the foundations we will walk on. First, the basic mathematical formalism used in quantum information is reviewed in @sec:quantum-information, along with its connection to the practical use of current quantum computers. Second, we will discuss how to make a quantum computer operate properly in practice in @sec:cc.

== Quantum Information <sec:quantum-information>

=== Quantum State

In a nutshell, quantum information is stored within a *quantum state*. In a digital quantum computer, a qubit is a two-level quantum system that acts as the smallest unit of memory of a quantum computer. Mathematically, the single-qubit state is defined as

$ ket(psi) = mat(delim: "[", a; b) = a ket(0) + b ket(1), $
<eq:quantum-state>

where $abs(a)^2 + abs(b)^2 = 1$. The ket notation used, however, can only describe a pure quantum state within a closed system, i.e., one with no interaction with an environment. In an open system, the quantum state interacts with the environment, and a more general description is a density matrix,

$ rho = sum_i p_i ketbra(psi_i) , $

where $p_(i)$ is a classical probability of pure state $ketbra(psi_i)$.

At a high level, quantum circuit abstraction is a way to reason about the operations of a quantum computer. This abstraction is a direct analogy to a classical circuit, which is a building block of operation in a classical computer. The quantum circuit is composed of quantum logic gates, as is the classical circuit. The quantum logic gate can be mathematically represented by a Unitary matrix. To perform an arbitrary computation on a quantum computer, a universal gateset is needed. There are several choices of the universal gateset. The elementary operations required for a universal gate are typically chosen from single and two-qubit gates.

The examples of single gates are
$
  X = mat(
    delim: "[", 0, 1;
    1, 0
  ),
  Y = mat(
    delim: "[", 0, - i;
    i, 0
  ),
  Z = mat(
    delim: "[", 1, 0;
    0, - 1
  ),
  H = mat(
    delim: "[", 1, 1;
    1, - 1
  ),
$
which are 3 Pauli gates and a Hadamard gate, respectively. For the two-qubit gate, the prime example would be the controlled-X gate,
$
  upright(C X) = mat(
    delim: "[", 1, 0, 0, 0;
    0, 1, 0, 0;
    0, 0, 0, 1;
    0, 0, 1, 0
  ).
$

The transformation of the quantum state by the quantum gate $hat(U)$ can be calculated by matrix multiplication as follows,
$
  ket(psi_f) = hat(U) ket(psi_i),
$
for the pure state case. In the case of density matrix formalism, the transformation can be calculated as,
$
  rho_f = hat(U) rho_i hat(U)^dagger .
$

However, in the presence of an environment, a Unitary operator cannot fully describe a possible transformation caused by noise. Instead of limiting ourselves to ideal operation, we can also incorporate noise in the calculation. In that case, we refer to this transformation as a quantum channel. Mathematically, a quantum channel could be noisy. In a simple representation, we introduce a set of Kraus operators ${K_0, ..., K_i }$ that satisfy $sum_i K_i^* K_i <= I$, the quantum channel transforming state $rho$ is
$
  rho_f -> Phi(rho) = sum_(i) K_(i) rho K_(i)^(*).
$
To extract the information from the quantum state, we need to measure the quantum state with the selected observable $hat(O)$. Typically, we want to extract information stored in the quantum state $rho$ in the form of the expectation value of a quantum observable
$
  expval(hat(O)) = Tr [ hat(O) rho ].
$ <eq:ideal-expval>

=== From Hamiltonian to Expectation value <sec:hamiltonian-to-unitary>
In the lower level than the quantum circuit model, the dynamics of the quantum state are governed by a differential equation (DE). In the closed system, given a Hamiltonian describing the system, $hat(H)(t)$, the quantum state $ket(psi(t))$ can be calculated by solving the Schrödinger equation in the form,
$
  i hbar partial / (partial t) ket(psi(t)) = hat(H)(t) ket(psi(t)).
$
Alternatively, in this thesis, the Unitary operator $hat(U)(t)$ is preferred and can be solved by solving,
$
  i hbar partial / (partial t)hat(U)(t) = hat(H)(t)hat(U)(t).
$
Generally, the system's Hamiltonian is time-dependent. Thus, the solution is given as a time-ordered operator
$
  hat(U)(t) = cal(T)_(+) exp { -i integral_(0)^(t) hat(H)_("total")(s) d s }
$
An analytical solution is hard to solve for, we typically need to use numerical method to solve for the solution.

In an open system, a different differential equation is needed to describe its dynamics. For instance, a Lindblad master equation is needed to solve for the quantum state in the form of a density matrix $rho(t)$. The master equation @lambertQuTiP5Quantum2024 @manzanoShortIntroductionLindblad2020 is a differential equation of the following form,
$
  partial / (partial t) rho(t) = -(i / hbar) [H_"ideal" (t), rho(t)] + sum_i 1/2 ( C_i rho(t) C_i^dagger - { C_i^dagger C_i, rho(t) } ).
$ <eq:master-equation>
The collapse operator $C_i$ describes noise, such as amplitude damping and dephasing.

At this level, a time-dependent Hamiltonian can be a function that is controlled by an experiment. For instance, Hamiltonian is controlled by an external microwave pulse in the case of a superconducting qubit. The Hamiltonian generates a quantum logic gate.

Finally, the expectation value of observable $hat(O)$ can be calculated following the @eq:ideal-expval.

=== Noise in the quantum system

// #task[
//   Reviews this @hashimPracticalIntroductionBenchmarking2025
// ]

Noise refers to the undesired factors that affect the dynamics of a system. In the quantum regime, there are several ways to quantify system noise. Regardless of the actual physical process that generates the noise, we can model it in the form we are interested in, which might be easier to deal with.

One of the most fundamental types of noise is Pauli noise. The noise in this system is *characterized* in terms of a quantum channel where the Kraus operators $hat(P)$ are a set of Pauli matrices and the Identity matrix. More formally, a Pauli channel is as follows,
$
  rho_f -> Phi (rho) = p_0 rho + p_1 hat(X) rho hat(X)^dagger + p_2 hat(Y) rho hat(Y)^dagger + p_3 hat(Z) rho hat(Z)^dagger
$

where ${ p_0, p_1, p_2, p_3 }$ is a probability that operator ${ hat(I), hat(X), hat(Y), hat(Z) }$ applied on the quantum state $rho$.

Modeling the noise as a Pauli channel is simple, yet powerful. Multiple error management techniques assume that system noise can be modeled as a Pauli channel—for example, quantum error correction and entanglement purification. The analysis using this model often performs at the circuit level. For instance, the noise is modeled as applying a Pauli gate randomly with some probability, allowing for a simulation of a large-scale quantum system with noise. This modelling approach does not specify the underlying physical process responsible for the Pauli noise, but models the system's effective noise as a Pauli channel.

One can model the noise at a lower level closer to physical implementation. Opposed to the Pauli channel that can be used regardless of the physical realization of the qubit. At this level, detailed implementation of the experiment can be considered. From the mathematical description of the quantum system, i.e., Hamiltonian, the noise can be inserted and modeled at that level. Consequently, solving for the solution would yield a dependence on the noise model, which can be used for further analysis, e.g., an optimal control problem. However, the analytical solution of an arbitrary noise model is generally hard, if not impossible, to solve for. Thus, there is a need for a numerical method.

Nonetheless, *detuning* is the noise model for which the solution of the system can be solved analytically in the simple case. Detuning usually refers to a deviation coefficient $delta$ from the ideal value. Consider the following transmon qubit with energy splitting $omega_q$, control strength $Omega_d$, and control signal $s(t)$, which explains the classical field of the form
$
  hat(H) = omega_q / 2 sigma_z + Omega_d s(bold(Theta), t) sigma_x,
$
In `qiskit` @javadi-abhariQuantumComputingQiskit2024, the signal is defined as a functional of a function of control envelope $h(bold(Theta), t)$ and drive frequency $omega_d$ and phase $phi$ as,

The detuning can be in both the qubit frequency, $omega_q -> omega_q + delta$, and the detuning, $omega_d -> omega_d + delta$. Note that the detune values can be arbitrary and do not have to be equal. Detuning can affect the control of the qubit, thereby deviating the system's dynamics in a non-trivial way. The noise can be mitigated using control calibration techniques such as DRAG. Furthermore, one can use a specialized optimal control solution to mitigate the noise @laforgueOptimalQuantumControl2022.

In the quantum open system, the noise is not limited to the unwanted Unitary transformation, i.e., unwanted energy transfer within the system, but also the system-environment energy transfer @hashimPracticalIntroductionBenchmarking2025 @krantzQuantumEngineersGuide2019 @guntherQuandaryOpensourcePackage2021. In the Bloch-Redfield model, the noise may be characterized by two quantities: (1) a longitudinal relaxation rate,
$
  Gamma_1 = 1/T_1,
$
and (2) a transverse relaxation rate,
$
  Gamma_2 = 1/T_2 = Gamma_1/2 + Gamma_phi,
$
where $Gamma_phi$ is the pure depahsing rate. The effect of the noise on the quantum state in @eq:quantum-state can be modeled in the Bloch-Redfield density matrix as,
$
  rho_("BR") = mat(1 + (|a|^2 -1 ) e^(-Gamma_1 t), a b^* e^(i delta omega t) e^(-Gamma_2 t); a^* b e^(-i delta omega t) e^(-Gamma_2 t), |b|^2 e^(-Gamma_1 t)).
$

The system can also be affected by *stochastic* noise, such as colored noise, which is represented by a Power Spectral Density (PSD). This type of noise is common in the solid-state qubit, specifically 1/f noise. This type of noise can be characterized and mitigated by specialized control such as Dynamical Decoupling (DD) pulse sequence @alvarezMeasuringSpectrumColored2011 @krantzQuantumEngineersGuide2019 @yanSuppressionDissipationLaserdriven2015.

In the qubit system realized from the energy levels of an atom, the state of the qubit is defined by two of its energy levels. The physical realization of control relies on the precise specific frequency. However, in the case that the frequency of the control field is deviated from the perfect value, the quantum state might transfer to the undesired level. This noise is called *leakage*. Specialized technique that can mitigate leakage noise is DRAG pulse @motzoiSimplePulsesElimination2009 @theisCounteractingSystemsDiabaticities2018 @hyyppaReducingLeakageSingleQubit2024.

*Crosstalk* is the noise that the quantum information stored in the qubit leaks to another qubit instead. This type of noise occurs in the system in which qubits are connected to each other via some interaction @majumderRealtimeCalibrationSpectator2020 @ash-sakiExperimentalCharacterizationModeling2020.

In the experimental setup, the measurement result from the quantum system is binary. However, a misclassification can occur, flipping the bit $0 -> 1$ and $1 -> 0$. This can be modeled as a probability of classical bit-flip error. Several error-mitigation techniques @tannuMitigatingMeasurementErrors2019 @gellerRigorousMeasurementError2020 @funckeMeasurementErrorMitigation2022 are proposed to mitigate measurement errors.

=== Fidelity

In quantum information, we are often interested in how close a quantum state is to another. The measure of the overlap between two quantum states is state fidelity. The state fidelity between a quantum state $rho$ and $sigma$ is
$
  F(rho, sigma) = upright(T r) [sqrt(sqrt(rho) sigma sqrt(rho))]^2 .
$

In the case where $rho = sigma$, the state fidelity $F(rho, sigma)= 1$, indicating that these two states are the same. In the case where $rho = ketbra(psi)$ and $sigma = ketbra(sigma)$ are pure state, then state fidelity reduces to
$
  F(ket(psi_rho), ket(psi_sigma)) =|braket(psi, sigma)|^2 .
$

Fidelity is not limited to the quantum state; we can also measure the closeness between quantum channels. The process fidelity measuring the closeness of quantum channel $hat(U)$ and $cal(E)$ is
$
  F_("process") (hat(U), cal(E)) = Tr ([S_hat(U)^dagger S_cal(E) ])/d^2,
$ <eq:process-fidelity>
where $S_O$ is a Superoperater representation of operator $hat(O)$ and $d$ is the number of basis states of the channel. Process fidelity is especially useful when one needs to measure the quality of a quantum channel in an experiment.

In the context of control calibration, we might be interested in the low-level control of a quantum device that maximizes an  *#gls("agf", long: true)* of target quantum gate and quantum channel. Let $hat(U)$ be target Unitary gate, the quantum channel $cal(E)$ realized in experiment, AGF is defined as
$
  macron(F)(cal(E), hat(U)) = integral d psi chevron.l psi|hat(U)^dagger cal(E)(|psi chevron.r chevron.l psi|) hat(U)|psi chevron.r .
$

From @nielsenSimpleFormulaAverage2002, we can write AGF in a form that is easier to collect in the real experiment. For a single qubit case, AGF can be written as
$
  macron(F)(cal(E), hat(U)) = 1/2 + 1/12 sum_(j = 1)^3 sum_(k = 0)^1 sum_(m = 1)^3 alpha_(j, k) b_(m, j) Tr [hat(P)_m cal(E)(rho_(j, k))],
$ <eq:agf>

where $hat(P)_j, hat(P)_m in {hat(X), hat(Y), hat(Z) }$. $ketbra(psi)_(j,k)$ are the basis states, $ket(+), ket(-), ket(i), ket(-i), ket(0), ket(1)$. The index $k in {0, 1}$ corresponds ${+1, -1}$ eigenvalues. The coefficients are $alpha_(j,0) = 1, alpha_(j, 1) = -1$ for any $j$ Pauli observable and $b_(m,j) = 1/2 Tr [ hat(U) hat(P)^dagger_j hat(U)^dagger hat(P)_m ]$.

== Characterization & Calibration (C&C) <sec:cc>

=== Calibration <sec:calibration>

Multiple physical realizations of quantum devices are available nowadays. Currently, there is no clear winner in building a quantum computer, just as a transistor is the main way to build a bit in a traditional computer. Regardless of the implementation choice, the physical operation of that platform used to realize quantum operation is most likely to be imperfect due to noise. For example, in the superconducting qubit platform @krantzQuantumEngineersGuide2019, the quantum state was controlled by a microwave pulse aimed at the qubit. Not just any control pulse can control the qubit, but it must be a pulse at a specific frequency. Furthermore, to rotate the state from the initial to the final position, control parameters such as duration, amplitude, and pulse axis play an important role.

In theory, the control parameters can be analytically calculated. But in practice, due to the noise, the theoretical ideal set of control parameters might not work perfectly as expected. In general, errors can be compensated for with careful adjustment of the control parameters. This process of adjusting the control to achieve some objective is referred to as control *calibration*. This process can be categorized by the need to interact with the target device to be calibrated @kochQuantumOptimalControl2022. First, a closed-loop approach in which the algorithm decides based on feedback from the target device, eliminating the need for interaction. Second is the open-loop approach, where the interaction is not needed. Instead, the algorithm will query a mathematical representation of the target device. The advantages and disadvantages of these two approach will be discussed in sec:closed-loop-calibration and @sec:open-loop-calibration. Previous studies had also attempted to get the best of both worlds, which we called the hybrid approach and discussed in @sec:hybrid-calibration. In any case, the objective of the calibration is to minimize or maximize some cost function. Typically, those cost functions are defined as a function of fidelity, which is the quantity that indicates the closeness to the target quantum state or operation. Further extension might include minimizing the duration of the control as an additional objective.

As mentioned, the open-loop approach did not require a direct interaction with the target device, but the mathematical representation, the construction of such a representation raises further complications. A mathematical *model* of the target device is not easily acquired, especially an accurate one. One of the rather straightforward ways to create the model is to come up with some parametrized function that represents the dynamics of the system, including noise. Then perform some experiments to estimate *system parameters*. This process of estimating the system parameters is referred to as *characterization* to be discussed in @sec:characterization.

==== Closed-loop Calibration <sec:closed-loop-calibration>

The most alluring feature of the closed-loop approach is the lack of need for extensive prior knowledge of the system. To put it simply, the algorithm requires little knowledge of the system. We refer to this type of algorithm as a model-free algorithm, such as reinforcement learning (RL)@bukovReinforcementLearningDifferent2018 @zhangWhenDoesReinforcement2019 @baumExperimentalDeepReinforcement2021 @niuUniversalQuantumControl2019 Bayesian optimization @lazinHighdimensionalMultifidelityBayesian2023 @renganathanLookaheadAcquisitionFunctions2021 @sauvageOptimalQuantumControl2020, Nelder-Mead (standard Blackbox optimization algorithm). These algorithms rely on different techniques; however, the goal remains, i.e., trying to optimize the parameters that maximize or minimize some cost function. We will now discuss the mainstream approaches in closed-loop calibration.

*Reinforcement learning* is one of the types of machine learning that is learn based on trial and error. The algorithm basically lets an agent perform a sequence of actions based on a policy and observe the environment's state and the reward for each action. At the end of the sequence, the policy is updated to maximize reward in the next sequence. RL has been widely used in robotic applications.

In the context of control calibration of a quantum device, each element of RL may be defined as follows. The environment is a target device, including noisy conditions. The action is a possible control of the target device; for example, Baum et al. @baumExperimentalDeepReinforcement2021 defined the action as the amplitude at each time step of the control sequence. And the reward is the value of fidelity, which may be penalized by the duration of the control. In other words, the agent is trained to perform control to maximize the fidelity as fast as possible.

The model-free feature of RL might become a disadvantage eventually if care is not taken. As the space of action grows, the number of interactions with the target device would become an unbearable cost. This is when the concepts of exploration and exploitation become crucial, i.e., whether to explore an unknown territory or fine-tune the known to the fullest. The right balance must be struck, otherwise, the cost of exploration will go further up than the risk of finding a better region. On the other hand, the right answer might reside right next to the explored space, awaiting just a tiny tweak to discover it. In the calibration task, we have a choice to choose a simple control with a tiny, limited search space or a far larger space that allows for fine-grained control. Constraints can be made to limit the possibility, mostly due to the limitations of the target device, making it possibly easier for the RL agent to find the right answer faster.

*Bayesian Optimization* typically employs the problem that the cost function is costly to evaluate. The concept is to model the given cost function as a Gaussian process. Upon observing new data, the model is updated. Then, the acquisition function decides based on a metric, such as expected improvement evaluated using a Gaussian model, the next point to evaluate. The steps are repeated until the budget runs out or terminal conditions are met. The simplified concept of Bayesian optimization is to evaluate the cost function at a point with some uncertainty, which may contain a critical point to be optimized; thus, it is well-suited for a task with a limited budget. At a high level, the acquisition function decides where to evaluate; this somewhat aligns with the concepts of exploration and exploitation in RL. One of the biggest disadvantages of Bayesian optimization is that it is not good at a large number of parameters.

Other approaches in closed-loop optimization are *Blackbox optimization*. We would like to refer to them as Blackbox because, other than feeding it the input and observing the output, we had no other access. A common example of this approach is the Nelder-Mead optimization algorithm. This approach is typically used to benchmark other optimization approaches.

==== Open-loop Calibration <sec:open-loop-calibration>
One of the greatest strengths of calibration via an open-loop manner is the ability to leverage the gradient direction toward optimal control parameters. This is possible because the target system is modeled as a parametrized function. Different techniques can then be used to calculate the gradient of the function represented by the target device. One of the first gradient-based algorithms for optimal control problems for quantum devices is Gradient Ascent Pulse Engineering (GRAPE) @khanejaOptimalControlCoupled2005. Many of its variations had been developed and reviewed in @porottiGradientAscentPulseEngineering2023.

Another gradient-based approach is Gradient Optimization of Analytic conTrols (GOAT) @machnesTunableFlexibleEfficient2018. These two approaches differ from each other in the method of gradient computation of system evolution with respect to some parameters.

With recent developments of deep neural networks, the computation of gradients on neural networks can be done efficiently. Hence, there is a study that uses a recently proposed architecture, Physics-Informed Neural Network (PINN), to study the problem of quantum optimal control @gungorduRobustQuantumGates2022.

Nonetheless, the open-loop approach required characterization of the quantum device, as system parameters are essential. Furthermore, the performance of the open-loop approach heavily depends on the accuracy of the model. Because calibrating using a mismatch system parameter is equivalent to calibrating a different device. Thus, careful system characterization is crucial for the open-loop approach.

==== Hybrid Approach Calibration <sec:hybrid-calibration>
Recent studies have explored the hybridization of open and closed loops. One of the approaches is to begin with open-loop optimization to produce a sub-optimal solution, then employ closed-loop optimization to enhance the solution further @goerzHybridOptimizationSchemes2015 @eggerAdaptiveHybridOptimal2014 @machnesTunableFlexibleEfficient2018. Another approach is to repurpose the data from characterization experiments to get a relatively good initial state, and iteratively update with data from experiments, such as those done in @evansFastBayesianTomography2022, by using the concept of Bayesian Optimization. It is also worth mentioning that  #cite(<wuDatadrivenGradientAlgorithm2018>, form: "prose") uses experimental data to help in the calculation of gradient-based optimization. However, doing so requires that the experiment be executed upon request from the optimizer. Again, these studies did not focus on signaling the recalibration.

=== Characterization <sec:characterization>

Characterization is the process that determines the system parameters. Thus, to characterize a device is to first identify the model or the form of a parametrized function that could describe the behavior of the system. Then, experiments have to be done to determine the parameters of the model. Another way to describe characterization is the process of estimating a mathematical object, i.e., "quantum channel", that describes how the system evolves.

We will categorize characterization into two main types depending on the targeting properties. First is channel characterization, where the entire quantum channel is fully described by a mathematical object. Second is parameter characterization, which targets a particular parameter of the aforementioned parametrized model of the system.

For the channel characterization, the most straightforward one is process tomography, which is also the most resource-intensive to perform as the size of the system grows larger. However, the method provides complete information of the targeted quantum channel, though distinguishing state-preparation, measurement error (SPAM) from the channel is not considered. The successive tomography is Gateset tomography @nielsenGateSetTomography2021, which provides a characterization of gate-set and is self-consistent (i.e., taking into account the SPAM error). The former method is also one of the ingredients for the closed-loop optimization, as fidelity can be computed from the quantum channel reconstructed from the process tomography. It is not necessary to perform process tomography if only fidelity is required @nielsenSimpleFormulaAverage2002 @flammiaDirectFidelityEstimation2011. Another recent approach models the system such that it takes into account the environment interaction with the system, then encodes the noise into an abstract mathematical function that can be estimated using machine learning techniques @youssryCharacterizationControlOpen2020, the _Graybox_ method. I refer to this class of characterization approaches as _parametrized channel characterization_. Note that the parameters here are not the parameters defined in the system model, but rather parameters that describe its corresponding quantum channel.

Parameter characterization assumes an explicit mathematical form of the system model. Thus, the system parameters have to be estimated directly with specialized methods. For the superconducting qubit that is subject to leakage noise, Wood et al. @woodQuantificationCharacterizationLeakage2018 and Chen et al. @chenMeasuringSuppressingQuantum2016 proposed methods to characterize the parameters. In a system where interactions between memories are not well controlled, crosstalk can occur. Crosstalk is the process where another memory affects another unintentionally in a destructive way and is being studied in @ash-sakiExperimentalCharacterizationModeling2020. One of the noises in a charged system is colored noise that is described by the power spectrum of the noise. This can be characterized using methods proposed in @huangRandomPulseSequences2025 @chalermpusitarakFrameBasedFilterFunctionFormalism2021. Tornow et al. @tornowMinimumQuantumRunTime2022 show that with restless measurement, the characterization process can be faster.

There are several studies about adaptive experiment characterization of quantum devices. Sarra et al. @sarraDeepBayesianExperimental2023 use Bayesian Optimal Experiment Design to estimate system Hamiltonians. Lennon et al. @lennonEfficientlyMeasuringQuantum2019 use neural networks and information gain approximation techniques for adaptive measurement, selecting points of maximal information gain to characterize quantum dots, improving measurement efficiency by 4 times compared to grid search. Fiderer et al. @fidererNeuralNetworkHeuristicsAdaptive2021 use reinforcement learning to minimize Bayes risk for qubit frequency estimation tasks. Note that these studies focus on the characterization task only.

== C&C Framework
With regard to the existing C&C framework, `C3` @wittlerIntegratedToolSet2021 is an integrated tool-set to perform control calibration and device characterization using both open-loop and closed-loop techniques. However, it uses a standard optimizer for its algorithms, and it does not focus on being a data-efficient framework. One recent framework, `qibocal` @pasqualeOpensourceFrameworkPerform2024, focuses on being *platform-agnostic*, i.e., API remains the same regardless of the target quantum device platform. However, it again does not focus on data utilization efficiency. While QCtrl's Boulder Opal @BoulderOpal is production-ready software with a study about a data-efficient characterization approach @staceOptimizedBayesianSystem2024, it is a closed-source package.

== The notion of "Data Efficiency" <sec:data-efficient-notion>

// #task[
//   - Review from Modern Bayesian Optimal Design @rainforthModernBayesianExperimental2024 Done?
//   - Reviews about the data-efficient approach in quantum. Done?
// ]

// #todoneedwrite[data-efficient in chapter 2]
// - Drift of noise, when to re-calibrate
// - The duration of calibration
// - "sample-efficiency"

// #show table.cell.where(x:0): set text(style: "italic")


The notion of "data-efficient" may be used differently in other literatures. In our context, "data-efficient" is an efficient usage of experimental data, i.e., from current data at hand, we want to determine the next batch of experiment data that will help us characterize the device faster. In other words, we want to create a dataset with a minimal sample size that can be used to construct an accurate predictive model.

The concept is indeed possible; that is, we want to perform the experiment where the model is the most uncertain. *@boed* @fosterVariationalBayesianOptimal2019 @liExpectedInformationGain2025 is a framework that allows us to bring the concept to reality. Mathematically, we are interested in the posterior model $p(y | theta, d)$ construction, where $y$ is observed data, $theta$ is the latent variable that we seek to infer with as little data as possible, and $d$ is the design of the experiment. The *@eig* of $theta$ with experiment with design $d$ can be defined as the expected difference between the entropy of the prior and the posterior
$
  "EIG"(d) eq.delta EE_(p(y|d))[H[p(theta)] - H[p(theta|y, d)]]
$
However, it is in general intractable to obtain a closed-form expression for @eig. There are a number of @eig estimators, each with its own tradeoff. We will estimate @eig by using a variational marginal (VM) estimator to estimate @eig @fosterVariationalBayesianOptimal2019. This method is preferred, since $theta$ may be high-dimensional. However, it is a biased estimator if the variational distribution used in the VM estimator does not contain the target distribution. However, it provides a fast convergence. The variational marginal @eig is given by,
$
  "EIG"(d) lt.eq.slant cal(U)_("marg")(d) & eq.delta EE_(p(y, theta | d)) [log p(y|theta, d)/(q_(m)(y|d))] \
                                          & approx hat(mu) (d) \
                                          & eq.delta 1/N sum_(n=1 )^N log p(y_n | theta_n, d) / (q_m (y_n | d)).
$ <eq:vm-eig>
Consequently, in order to estimate @eig using approach propose by #cite(<fosterVariationalBayesianOptimal2019>, form: "prose"), we have to be able to compute the *probabilistic* predictive model (posterior distribution) $p(y|theta, d)$ and the marginal distribution $q(y|d)$. We can estimate the posterior distribution in a number of ways. For instance, using @bnn, normalizing flow, and particle filter.

After we estimate the @eig for the given experiment design, we have to decide which experiments to perform to gather data from the device. One simple thing we can do is to perform an experiment where the @eig is maximized. This approach is one-step optimal from an information-theoretic point of view @fosterVariationalBayesianOptimal2019. That is, it does not take into account the future experiment. This simple strategy might not maximize the total @eig of the sequential experiments. Furthermore, this approach requires two-step optimization, which are (1) estimation of the @eig and (2) searching for an experimental design that maximizes @eig. To reduce the computational challenge, #cite(<fosterUnifiedStochasticGradient2020>, form: "prose") proposes a single-step approach. Later, #cite(<sarraDeepBayesianExperimental2023>, form: "prose") uses the single-step approach to achieve @boed in the context of quantum device characterization. They use normalizing flow to estimate the posterior distribution of the system. Interestingly, some normalizing flow can be interpreted as a @bnn @wehenkelYouSayNormalizing2020.

As discussed earlier, greedily experimenting with a design that maximizes the @eig at the current step does not necessarily maximize the total @eig of the sequential experiments setting. To tackle the problem of sequential experiment design @rainforthModernBayesianExperimental2024, recent studies have proposed the use of @dnn to predict the next experimental design @fosterDeepAdaptiveDesign2021 @ivanovaImplicitDeepAdaptive2021. The @dnn is learn by minimizing the lower bound of the total @eig. However, this approach requires significant computational power during @dnn training. While at the deployment time, the next design can be efficiently determined. The concept of using @dnn to make a decision on what is the next experiment design to perform in the context of quantum characterization is explored in @fidererNeuralNetworkHeuristicsAdaptive2021, but they did not use the result presented in @fosterDeepAdaptiveDesign2021 @ivanovaImplicitDeepAdaptive2021; instead, they optimize the @dnn using Bayes risk (their source code is not easy to use.)

Alternatively to the @eig, #cite(<staceOptimizedBayesianSystem2024>, form: "prose") proposes the OBSID algorithm for performing a sequential experiment. They used a particle filter to sample from the posterior distribution. They experiment with designs that minimize cost functions based on anticipated posterior covariance and modified Fisher information. There are some disadvantages in using Fisher information compared to @eig @rainforthModernBayesianExperimental2024. For example, since the Fisher Information Matrix is a matrix, one has to optimize the summary statistics rather than fully characterize the statistics as @eig.


