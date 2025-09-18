#import "@preview/physica:0.9.5": *
#import "@preview/glossarium:0.5.6": gls
#import "@preview/drafting:0.2.2": inline-note
#import "../utils.typ": style, todoneedcite, todoneedwrite
#import "@preview/gentle-clues:1.2.0": *
#import "@preview/drafting:0.2.2": inline-note, margin-note

= Before the Journey Begin #emoji.face.wink <sec:before-thejourney-begins>

Proceeding to the unknown territory without necessary knowledges is an unwise decision. Thus, I would like to reviews the foundations that we are going to walk forward on. First, basic mathematics formalism used in quantum information is reviewed in @sec:quantum-information, along with the connection to practical usage of current quantum computer. Second, we will discuss how to make quantum computer operate properly in practice in @sec:cc.

== Quantum Information <sec:quantum-information>

=== Quantum State

In a nutshell, quantum information is stored within a *quantum state*. In a digital quantum computer, qubit is a two-level quantum system that act a smallest unit of memory of quantum computer. Mathematically, the single qubit state is defined as

$ ket(psi) = mat(delim: "[", a; b), $

where $abs(a)^2 + abs(b)^2 = 1$. The ket notation used however can only describe a pure quantum state, contained within closed system, i.e. no interaction with an environment. In an open system, the quantum state interacts with the environment, a more general description is a density matrix,

$ rho = sum_i p_i ketbra(psi_i) , $

where $p_(i)$ is a classical probability of pure state $ketbra(psi_i)$.

At the high level, quantum circuit abstraction is a way to reason about operations of the quantum computer. This is a direct analogy to classical circuit that is a building block of operation in a classical computer. Quantum circuit also compose of quantum logic gates as the classical circuit also compose of logic gates. The quantum logic gate can be mathematically represented by Unitary matrix. To perform an arbitrary computation on quantum computer, an universal gateset is needed. There are several choice of the universal gateset. The basis operation required for universal gate typically chosen from single and two qubit gates.

The example of single gates are
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
which are 3 Pauli gates and Hadamard gate respectively. For the two qubit gate, the prime example would be controlled-X gate,
$
  upright(C X) = mat(
    delim: "[", 1, 0, 0, 0;
    0, 1, 0, 0;
    0, 0, 0, 1;
    0, 0, 1, 0
  ).
$

The transformation of quantum state by the quantum gate $hat(U)$ can be calculate by matrix multiplication as follow,
$
  ket(psi_f) = hat(U) ket(psi_i),
$
for the pure state case. In the case of density matrix formalism, the transformation can be calculated as,
$
  rho_f = hat(U) rho_i hat(U)^dagger .
$

However, in a presence of environment, Unitary operator cannot fully describe a possible transformation cause by noise. Instead of limiting ourself with ideal operation, we can also incorporate noise in the calculation. In that case, we refer to this transformation as a quantum channel. Mathematically, quantum channel could be noisy. In a simple representation, we introduce a set of Kraus operators ${K_0, ..., K_i }$ that satisfying $sum_i K_i^* K_i <= I$, the quantum channel transforming state $rho$ is
$
  rho_f -> Phi(rho) = sum_(i) K_(i) rho K_(i)^(*).
$
To extract the information from the quantum state, we need to measure the quantum state with selected observable $hat(O)$. Typically, we want to extract information stored in the quantum state $rho$ in the form of expectation value of quantum observable
$
  expval(hat(O)) = Tr [ hat(O) rho ].
$ <eq:ideal-expval>

=== From Hamiltonian to Expectation value <sec:hamiltonian-to-unitary>
In the lower level than the quantum circuit model, the dynamics of quantum state is governed by differential equation (DE). In the closed system, given a Hamiltonian describing the system, $hat(H)(t)$, the quantum state $ket(psi(t))$ can be calculated by solving Schrödinger equation of a form,
$
  i hbar partial / (partial t) ket(psi(t)) = hat(H)(t) ket(psi(t)).
$
Alternatively, in this thesis, the Unitary operator $hat(U)(t)$ is preferred and can be solved by solving,
$
  i hbar partial / (partial t)hat(U)(t) = hat(H)(t)hat(U)(t).
$
Generally, the Hamiltonian of the system is time-dependent. Thus, the solution is given as time-ordered operator
$
  hat(U)(t) = cal(T)_(+) exp { -i integral_(0)^(t) hat(H)_("total")(s) d s }
$
An analytical solution is hard to solve for, we typically need to use numerical method to solve for the solution.

In the open loop system, a different differential equation is needed to describe the dynamics of the system. For example, a Master equation is need to solve for the quantum state in the form of density matrix.

At this level, a time-dependent Hamiltonian can be a function that is controlled by an experiment. For instance, Hamiltonian is controlled by external microwave pulse in the case of superconducting qubit. The Hamiltonian is responsible for generation of quantum logic gate.

Finally, the expectation value of observable $hat(O)$ can be calculated following the @eq:ideal-expval.

=== Noise

#task[
  Reviews this @hashimPracticalIntroductionBenchmarking2025
]

Noise is a term that describe the undesired factors that affect the dynamics of the system. In quantum regime, there are several ways to quantify noise in the system. Regardless or the actual physical process that generate noise, we could model the noise in the form that we are interested, and might easier to deal with.

One of the most fundamental type of noise is Pauli noise. The noise in this system is *characterized* in term of quantum channel that the Kraus operators $hat(P)$ is a set of Pauli matrices and Identity matrix. More formally, a Pauli channel as follows,
$
  rho_f -> Phi (rho) = p_0 rho + p_1 hat(X) rho hat(X)^dagger + p_2 hat(Y) rho hat(Y)^dagger + p_3 hat(Z) rho hat(Z)^dagger
$

where ${ p_0, p_1, p_2, p_3 }$ is a probability that operator ${ hat(I), hat(X), hat(Y), hat(Z) }$ applied on the quantum state $rho$.

Modeling the noise Pauli channel is simple, yet powerful. Error management techniques are built by assuming that noise in the system can be model as Pauli channel. For example, quantum error correction, and entanglement purification. The analysis using this model often perform at the level of circuit level. For instance, the noise is model as applying Pauli gate randomly with some probability, allowing for a simulation of large scale quantum system with noise. This approach does not restrict an underlying physical process that responsible for the Pauli noises, but model the effective noise of the system as Pauli channel.

One can model the noise at lower level closer to physical implementation. Oppose to Pauli channel that can be used regardless of the physical realization of qubit. At this level, details experiment implementation can be take into account. From the mathematical description of the quantum system, i.e. Hamiltonian, the noise can be insert and model at that level. Consequently solving for the solution would yield a dependent on noise model, which can be used for further analysis, e.g. optimal control problem. However, analytical solution of arbitrary noise model is generally hard, if not impossible to solve for. Thus, the need for a numerical method.

Nonetheless, *detuning* is the noise model that the solution of the system can be solve analytically in the simple case. Detuning usually refer to a deviation coefficient $delta$ to the ideal value. Consider a following transmon qubit with energy splitting $omega_q$ and control strength $Omega_d$ and control signal $s(t)$ which explain the classical field of the form
$
  hat(H) = omega_q / 2 sigma_z + Omega_d s(bold(Theta), t) sigma_x,
$
In `qiskit` @javadi-abhariQuantumComputingQiskit2024, the signal is defined a functional of a function of control envelope $h(bold(Theta), t)$ and drive frequency $omega_d$ and phase $phi$ as,

The detuning can be in both qubit frequency $omega_q -> omega_q + delta$ and $omega_d -> omega_d + delta$. Note the the detune values can be arbitrary and do not have to be the same value. Detuning can affect the control of the qubit, which can deviate the dynamics of the system in the non-trivial way. The noise can be solve by careful characterization of system parameters in the case of calibrating the control by open-loop approach. Furthermore, one can use specialize optimal control solution to mitigate the noise @laforgueOptimalQuantumControl2022.

In the quantum open system, #todoneedwrite[T1 and T2 noise] (see @krantzQuantumEngineersGuide2019 for great reference) T1 and T2 noise

The system can also be affected by a *stochastic* noise, such as noise produced by colored noise, which represented by a Power Spectrum Density (PSD) . This type of noise is common in the solid-state qubit, specifically 1/f noise. #todoneedcite This type of noise can be mitigated by specialize control such as Dynamical Decoupling (DD) pulse sequence.

In the qubit system that realized from energy level of an atom, the state of qubit is then defined as two of its energy levels. The physical realization of control relies the precise specific frequency. However, in the case that the frequency of the control field is deviated from the perfect value, the quantum state might transfer to the undesired level. This noise is referred to as *leakage*. Specialize technique that can mitigate leakage noise is DRAG pulse. #todoneedcite

*Crosstalk* is the noise that the quantum information stored the qubit leak to another qubit instead. This type of noise happens to the system that qubits are connected with each other via some interaction. #todoneedcite

In the experimental setup, measurement result from quantum system is a binary result. However, a mis-classification can occur which flip the bit $0 -> 1$ and $1 -> 0$. This can be model as a probability of classical bit-flip error. Several error mitigations techniques #todoneedcite are proposed to mitigate the measurement error.

=== Fidelity

In quantum information, we often interested in how close is a quantum state to another. The measure of the overlap between two quantum states is state fidelity. The state fidelity between a quantum state $rho$ and $sigma$ is
$
  F(rho, sigma) = upright(T r) [sqrt(sqrt(rho) sigma sqrt(rho))]^2 .
$

In the case where $rho = sigma$, the state fidelity $F(rho, sigma)= 1$, indicating that these two states are the same. In the case where $rho = ketbra(psi)$ and $sigma = ketbra(sigma)$ are pure state, then state fidelity reduces to
$
  F(ket(psi_rho), ket(psi_sigma)) =|braket(psi, sigma)|^2 .
$

Fidelity is not limited to the quantum state, we can also measure the closeness between quantum channel. The process fidelity measuring the closeness of quantum channel $hat(U)$ and $cal(E)$ is
$
  F_("process") (hat(U), cal(E)) = Tr ([S_hat(U)^dagger S_cal(E) ])/d^2,
$ <eq:process-fidelity>
where $S_O$ is a Superoperater representation of operator $hat(O)$ and $d$ is the number of basis states of the channel. Process fidelity is especially useful when one needs to measure the quality of quantum channel in experiment.

In the context of control calibration, we might interested in the low-level control of quantum device that maximize an  *#gls("agf", long: true)* of target quantum gate and quantum channel. Let $hat(U)$ be target Unitary gate, the quantum channel $cal(E)$ realized in experiment, AGF is defined as
$
  macron(F)(cal(E), hat(U)) = integral d psi angle.l psi|hat(U)^dagger cal(E)(|psi angle.r angle.l psi|) hat(U)|psi angle.r .
$

From @nielsenSimpleFormulaAverage2002, we can write AGF in the form that is easier to perform experiment. For a single qubit case, AGF can be written as
$
  macron(F)(cal(E), hat(U)) = 1/2 + 1/12 sum_(j = 1)^3 sum_(k = 0)^1 sum_(m = 1)^3 alpha_(j, k) b_(m, j) Tr [hat(P)_m cal(E)(rho_(j, k))],
$ <eq:agf>

where $hat(P)_j, hat(P)_m in {hat(X), hat(Y), hat(Z) }$. $ketbra(psi)_(j,k)$ are the basis states, $ket(+), ket(-), ket(i), ket(-i), ket(0), ket(1)$. The index $k in {0, 1}$ corresponds ${+1, -1}$ eigenvalues. The coefficients are $alpha_(j,0) = 1, alpha_(j, 1) = -1$ for any $j$ Pauli observable and $b_(m,j) = 1/2 Tr [ hat(U) hat(P)^dagger_j hat(U)^dagger hat(P)_m ]$.

== Characterization & Calibration (C&C) <sec:cc>

=== Calibration <sec:calibration>

Multiple choices of physical realization of quantum device are available nowadays. Currently, there is no clear winner in building quantum computer, like transistor is the main way to build a bit in traditional computer. Regardless of the implementation choice, physical operation of that platform that used to realize quantum operation most likely to be imperfect due to noise. For example, in superconducting qubit platform @krantzQuantumEngineersGuide2019, the quantum state was controlled by microwave pulse aims at qubit. Not just any control pulse can control the qubit, but it must be pulse at specific frequency. Furthermore, to rotate the state from initial position to final position, control parameter such as duration, amplitude, and axis of pulse plays an important role.

In theory, the control parameters can be analytically calculated. But in practice, due to the noise, theoretical ideal set of control parameter might not work perfectly as expected. In general, error can be compensated with careful adjustment of control parameter. This process of adjusting the control to achieve some objective is referred to as control *calibration*. This process can be categorized by the needed to interact with target device to be calibrated @kochQuantumOptimalControl2022. First is closed-loop approach, where the algorithm decides based on the feedback from target device, thus the need of interaction. Second is open-loop approach, where the interaction is not needed. Instead, the algorithm will query at a mathematical representation of the target device. The advantages and disadvantages of these two approach will be discussed in sec:closed-loop-calibration and @sec:open-loop-calibration. Previous studies had also attempted to get the best of both worlds, where I called the combined approach as hybrid approach to be discussed in @sec:hybrid-calibration . In any case, the objective of the calibration is to minimize or maximize some cost function. Typically, those cost functions are defined as a function of fidelity, which is the quantity indicate the closeness to the target quantum state or operation. Further extension might include minimizing duration of the control as an additional objective.

As mentioned, open-loop approach did not required a direct interaction with target device but the mathematical representation, the construction of such a representation raise further complications. Mathematical *model* of target device is not easily acquired, especially the accurate one. One of the rather straightforward way to create the model is to come up with some parametrized function that represent a dynamics of the system including noise. Then perform some experiments to estimate *system parameters*. This process of estimate the system parameters is referred to as *characterization* to be discuss in @sec:characterization.

==== Closed-loop Calibration <sec:closed-loop-calibration>

The most alluring feature of closed-loop approach is perhaps, the lack of needs of extensive prior knowledge of the system. To put it in simpler words, the algorithm does not require much knowledge about the system. I referred to this type of algorithm as model-free algorithm such as reinforcement learning (RL)@bukovReinforcementLearningDifferent2018 @zhangWhenDoesReinforcement2019 @baumExperimentalDeepReinforcement2021 @niuUniversalQuantumControl2019 Bayesian optimization @lazinHighdimensionalMultifidelityBayesian2023 @renganathanLookaheadAcquisitionFunctions2021 @sauvageOptimalQuantumControl2020, Nelder-Mead (standard Blackbox optimization algorithm). These algorithms rely on difference techniques, however, the goal remains, i.e. trying to optimize the parameters that maximize or minimize some cost function. We will now discuss the main stream approaches in closed-loop calibration.

*Reinforcement learning* is one of the type of machine learning that is learn based on trial and error. The algorithm basically let an agent(s) perform a sequence of action based on policy and observe the state of the environment along with the reward of the action. At the end of the sequence, the policy is updated to maximize reward in the next sequence. RL has been widely used in robotic applications.

In the context of control calibration of quantum device, each element of RL may defined as follows. The environment is target device including noisy condition. The action is a possible control of target device, for example, Baum et al @baumExperimentalDeepReinforcement2021, defined action as a value of amplitude of each time step of the control sequence. And the reward is the value of fidelity and might be penalized by the duration of the control. In other words, the agent is trained to perform control to maximize the fidelity as fast as possible.

The model-free feature of RL might become disadvantage eventually if care is not taken. As the space of action grow, the amount of interactions with the target device would become unbearable cost. This is when the concept of exploration and exploitation become crucial, i.e. whether to explore an unknown territory or fine-tune the known to the fullest. The right balance must be strike, otherwise, the cost of exploration will go further up from the bet of finding a better region. On the other hand, the right answer might resign right next to the explored space, awaiting for just a tiny tweak to discover it. In the calibration task, we have a choice to chose a simple control with tiny limited search space or a far larger space that allows for fine-grain control. Constraints can be made to limited the possibility, mostly due to the limitation of the target device, possibly easier for RL agent to find the right answer faster.

*Bayesian Optimization* typically employs the problem that the cost function is costly to evaluate. The concept is to model the given cost function as a Gaussian process. Upon observation of new data, the model is updated with the data. Then, the acquisition function decides based on a metric, such as expected improvement evaluated using a Gaussian model, the next point to evaluate. The steps are repeated until the budget runs out, or terminal conditions are met. The simplified concept of Bayesian optimization is to evaluate the cost function at some uncertain point that might contain a critical point to be optimized for, thus it is well-suited for a task with a limited budget. On the high level, acquisition function act as a decider of where to evaluate, this is somewhat align with the concept of exploration and exploitation in RL. One of the biggest disadvantage of Bayesian optimization is that it is not good at large number of parameters.

Other approaches in closed-loop optimization is *Blackbox optimization*. I would like to referred to them as Blackbox because other than feeding it the input and observing the output, we simply did not have other access. The common example of this approach is Nelder-Mead optimization algorithm. This approach typically used to benchmark other optimization approaches.

==== Open-loop Calibration <sec:open-loop-calibration>
One of the greatest strength of calibration via open-loop manner is the ability to leverage the gradient direction toward optimal control parameter. This is possible because the target system is modeled as a parametrized function. Difference techniques can then be used to calculate the gradient of the function represented target device. One of the first gradient-based algorithm for optimal control problem for quantum device is GRadient Ascent Pulse Engineering (GRAPE) @khanejaOptimalControlCoupled2005. Many of its variations had been developed and reviewed in @porottiGradientAscentPulseEngineering2023.

Another gradient-based approach is Gradient Optimization of Analytic conTrols (GOAT) @machnesTunableFlexibleEfficient2018. These two approaches differentiate from each other by the method of gradient computation of system evolution with respect to some parameters.

With recent developments of deep neural networks, the computation of gradients on neural networks can be done efficiently. Hence, there is a study that uses a recently proposed architecture, Physics-Informed Neural Network (PINN) to study the problem of quantum optimal control @gungorduRobustQuantumGates2022.

Nonetheless, open-loop approach required characterization of quantum device, as system parameters are essential. Furthermore, performance of open-loop approach heavily depends on the accuracy of the model. Because, calibrate using a mismatch system parameter is equivalent to calibrate a different device. Thus, careful system characterization is crucial for open-loop approach.

==== Hybrid Approach Calibration <sec:hybrid-calibration>
Recent studies have explored hybridization of open and closed-loops. One of the approaches is to begin with open-loop optimization to produce a sub-optimal solution, then employ closed-loop optimization to enhance the solution further @goerzHybridOptimizationSchemes2015 @eggerAdaptiveHybridOptimal2014 @machnesTunableFlexibleEfficient2018. Another approach is to repurpose the data from characterization experiments to get a relatively good initial state, and iteratively update with data from experiments such as done in @evansFastBayesianTomography2022 by using the concept of Bayesian Optimization. It is also worth mentioning that  #cite(<wuDatadrivenGradientAlgorithm2018>, form: "prose") uses experimental data to help in the calculation of gradient-based optimization. However, doing so requires that the experiment has to be executed upon request from the optimizer. Again, these studies did not focus on signaling the re-calibration.

=== Characterization <sec:characterization>

Characterization is the process that determines the system parameters. Thus, to characterize a device is to first identify the model or the form of parametrized function that could describe the behavior of the system. Then, experiments have to be done to determine the parameters of the model. Another way to describe characterization is the process of estimating a mathematical object, i.e. "quantum channel", that describes how does the system evolve.

I will categorize characterization into two main types depending on the targeting properties. First is channel characterization where the entire quantum channel is fully described by a mathematical object. Second is parameter characterization, which target a particular parameter of the aforementioned parametrized model of the system.

For the channel characterization, the most straightforward one is process tomography, which is also the most resource-intensive to perform as the size of the system grows larger. However, the method provides complete information of the targeted quantum channel, though distinguishing state-preparation, measurement error (SPAM) from the channel is not considered. The successive tomography is Gateset tomography @nielsenGateSetTomography2021 which provides a characterization of gate-set and is self-consistent (i.e. taking into account the SPAM error). The former method is also one of the ingredients for the closed-loop optimization as fidelity can be computed from the quantum channel reconstructed from the process tomography. It is not necessary to perform process tomography if only fidelity is required @nielsenSimpleFormulaAverage2002 @flammiaDirectFidelityEstimation2011. Another recent approach models the system such that it takes into account the environment interaction with the system, then encodes the noise into an abstract mathematical function that can be estimated using machine learning techniques @youssryCharacterizationControlOpen2020, the _Graybox_ method. I refer to this class of characterization approaches as _parametrized channel characterization_. Note that the parameters here are not the parameters defined in the system model, but rather parameters that describe its corresponding quantum channel.

Parameter characterization assumes an explicit mathematical form of the system model. Thus, the system parameters have to be estimated directly with specialized methods. For the superconducting qubit that is subject to leakage noise, Wood et al. @woodQuantificationCharacterizationLeakage2018 and Chen et al. @chenMeasuringSuppressingQuantum2016 proposed methods to characterize the parameters. In a system where interactions between memories are not well controlled, crosstalk can occur. Crosstalk is the process where another memory affects another unintentionally in a destructive way and is being studied in @ash-sakiExperimentalCharacterizationModeling2020. One of the noises in a charged system is colored noise that is described by the power spectrum of the noise. This can be characterized using methods proposed in @huangRandomPulseSequences2025 @chalermpusitarakFrameBasedFilterFunctionFormalism2021. Tornow et al. @tornowMinimumQuantumRunTime2022 show that with restless measurement, characterization process can be faster.

There are several studies about adaptive experiment characterization of quantum devices. Sarra et al. @sarraDeepBayesianExperimental2023 use Bayesian Optimal Experiment Design to estimate system Hamiltonians. Lennon et al. @lennonEfficientlyMeasuringQuantum2019 use neural networks and information gain approximation techniques for adaptive measurement, selecting points of maximal information gain to characterize quantum dots, improving measurement efficiency by 4 times compared to grid search. Fiderer et al. @fidererNeuralNetworkHeuristicsAdaptive2021 use reinforcement learning to minimize Bayes risk for qubit frequency estimation tasks. Note that these studies focus on the characterization task only.

== C&C Framework
With regard to existing C&C framework, `C3` @wittlerIntegratedToolSet2021 is an integrated tool-set to perform control calibration, and device characterization using both open-loop and closed-loop techniques. However, it uses standard optimizer for its algorithms, and it does not focus on being data-efficient framework. One recent framework, `qibocal` @pasqualeOpensourceFrameworkPerform2024, focuses on being *platform-agnostic*, i.e. API remains the same regarding of the target quantum device platform. However, again, it does not focus on the data utilization efficiency aspect. While QCtrl's Boulder Opal @BoulderOpal is production ready software with study about data-efficient characterization appraoch @staceOptimizedBayesianSystem2024a, it is a close soruce package.

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


The notion of "data-efficient" may be used differently in other literatures. In our context, "data-efficient" is an efficient usage of experimental data, i.e. from current data at hand, we want to determine the next batch of experiment data that will help us characterize the device faster. In other words, we want to create the dataset with minimal sample size, that can be used to construct an accurate predictive model.

The concept is indeed possible, that is we want to perform the experiment where the model is uncertain the most. *@boed* @fosterVariationalBayesianOptimal2019 @liExpectedInformationGain2025 is a framework that allow us to bring the concept to reality. Mathematically, we interested in posterior model $p(y | theta, d)$ construction, where $y$ is an observed data, $theta$ is latent variables that we seek to infer for with less data as possible, and $d$ is a design of the experiment. The *@eig* of $theta$ with experiment with design $d$ can be defined as the expected different between entropy of prior and posterior
$
  "EIG"(d) eq.delta EE_(p(y|d))[H[p(theta)] - H[p(theta|y, d)]]
$
However, it is intractable in general to obtain closed-form of the @eig. There are numbers of @eig estimator, each with its own tradeoff. We will estimate @eig by using variational marginal (VM) estimator to estimate @eig @fosterVariationalBayesianOptimal2019. This method is prefer, since $theta$ may be high dimensional. Though, it is a biased estimator if the variational distribution used to in VM estimator does not contain target distribution. However, it provides a fast convergence. The variational marginal @eig is given by,
$
  "EIG"(d) lt.eq.slant cal(U)_("marg")(d) & eq.delta EE_(p(y, theta | d)) [log p(y|theta, d)/(q_(m)(y|d))] \
                                          & approx hat(mu) (d) \
                                          & eq.delta 1/N sum_(n=1 )^N log p(y_n | theta_n, d) / (q_m (y_n | d)).
$ <eq:vm-eig>
Consequently, in order to estimate @eig using approach propose by #cite(<fosterVariationalBayesianOptimal2019>, form: "prose"), we have to be able to compute the *probabilistic* predictive model (posterior distribution) $p(y|theta, d)$ and the marginal distribution $q(y|d)$. We can estimate the posterior distribution in a number of ways. For instance, using @bnn, normalizing flow, and particle filter.

After we estimate the @eig given experiment design, we have to decide the experiments to perform to gather data from the device. One simple things we can do is to perform experiment where the @eig is maximized. This approach is one-step optimal in information-theoretic point of view @fosterVariationalBayesianOptimal2019. That is, it does not take in to account the future experiment. This simple strategy might not maximize total @eig of the sequential experiments. Furthermore, this approach requires two steps optimization, which are (1) estimation of the @eig and (2) the searching of experimental design that maximize @eig. To reduce the computation challenge, #cite(<fosterUnifiedStochasticGradient2020>, form: "prose") propose a single step approach. Later, #cite(<sarraDeepBayesianExperimental2023>, form: "prose") use the single step approach to achieve @boed in quantum device characterization context. They use normalizing flow to estimate the posterior distribution of the system. Interestingly, some normalizing flow can be interpreted as a @bnn @wehenkelYouSayNormalizing2020.

As discuss earlier, to greedy perform experiment with a design that maximize the @eig at the current step does not necessary maximize the total @eig of the sequential experiments setting. To tackle the problem of sequentail experiment design @rainforthModernBayesianExperimental2024, recent studies have propose the use of @dnn to predict the next experimental design @fosterDeepAdaptiveDesign2021 @ivanovaImplicitDeepAdaptive2021. The @dnn is learn by minimizing the lower bound of the total @eig. However, this approach require significant computation power during @dnn training. While,  at the deployment time, the next design can be efficiently determined. The concept of using @dnn to make decision of what is the next experiment design to perform in context of quantum characterization is explored in @fidererNeuralNetworkHeuristicsAdaptive2021, but they did not use the result presented in @fosterDeepAdaptiveDesign2021 @ivanovaImplicitDeepAdaptive2021, instead they optimize the @dnn using Bayes risk (their source code is not easy to use.)

Alternatively to the @eig, #cite(<staceOptimizedBayesianSystem2024a>, form: "prose") propose OBSID algorithm to perform sequentail experiment. They used particle filter to sample from the posterior distribution. They perform the experiment at the design that minimize cost functions based on anticipated posterior covariance and modified fisher information. There are some disadvantages in using Fisher information compare to @eig @rainforthModernBayesianExperimental2024. For example, since Fisher Information Matrix is a matrix, one has to optimize the summary statistics instead of fully characterize statistics as @eig.


