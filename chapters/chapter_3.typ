#import "../utils.typ": control, todocontinue, todoneedwrite
#import "@preview/physica:0.9.7": *
#import "@preview/drafting:0.2.2": inline-note, margin-note
#import "../utils.typ": expval, tldr
#import "@preview/gentle-clues:1.2.0": *


#let argmax = math.op("arg max", limits: true)

#let estexpval = $EE lr([hat(O)], size: #50%)_(rho_0)$
#let idealexpval = $expval(hat(O))_(rho_0)$
#let mseeloss = $cal(L)_("MSE[E]")$
#let aefloss = $cal(L)_("AE[F]")$
#let var = math.op("Var")
#let vx = $upright(bold(x))$

// Debug section
// == inspeqtor <sec:inspeqtor>
// == boed <sec:boed-experiment>

= The Ways

In a nutshell, it is resource-intensive to achieve an optimal control over the target device, either by an open-loop or a closed-loop approach. With closed-loop approach, the balance of exploration and exploitation has to be chose and keep in mind that this is happen on the real device. With the open-loop approach, the accuracy of the characterized model is prominent. Thus, the needs of the experiment to characterize the target device are still crucial and potentially costly. A characterized model can be used beyond the optimal control problem, further information can be extracted, and the interaction cost with the target device can be reduced by interacting with the model first in the early stage of development.

#goal[Hence, reducing the interaction with device as much as possible yet, still accurately characterized device, i.e. "*data-efficient*" is of a great interest.]

Through this section, we will discuss how we can approach the problem and the ways forward to data efficiency, starting from the characterization of a quantum device using statistical machine learning and then moving forward to the probabilistic machine learning approach. Building upon the solid foundation of the statistical version of the model, then equip it with uncertainty quantification capability with the probabilistic version. Finally, the model can be used with @boed, the principled framework for efficient utilization of data @fosterVariationalBayesianOptimal2019

Machine learning is a field of study to model a system of interest without explicitly programming it to perform prediction by inferring from observed data. In this thesis, we would like to categorize the field into statistical and probabilistic approaches. In our context, we are interested in a regression program since we are dealing with the problem of quantum device characterization to construct a *predictive model*. For the @sml, we consider the model that learns by minimizing an objective function computed from the observed data. While the @pml, we infer the predictive model from the observed data by using Bayesian inference.

@sml is a mature field that finds its success in many fields. Compare to the @pml, it is a faster and more efficient way to construct a predictive model. However, it does not explicitly quantify the prediction uncertainty, which is crucial in many applications such as self-driving, medical applications, etc. On the other hand, the probabilistic approach is a mathematical formulation of uncertainty @henningProbabilisticMachineLearning2023. It is also a common language that we can use to connect with other fields @p.murphyProbabilisticMachineLearning2022. Moreover, mpirical Risk Minimization (ERM) is equivalent to Maximum a posterior (MAP) which give a natural connection between @sml and @pml. Nonetheless, we can learn from @sml's approach and use the insights gained from @sml to implement the @pml approach, which we might view as a natural way to use @sml as a prior for the @pml. For example, we can use @sml as a baseline reference for the @pml implementation.

// #task[
//   From @henningProbabilisticMachineLearning2023
//   + Probability is a mathematical formulation of uncertainty
//   + Empirical Risk Minimization (ERM) is equivalent to Maximum a posterior (MAP)
//   + Gaussian Process (GP) is an infinitely wide single-layer neural network
//   + Deep networks is GPs
//   + We can convert any DNN to GPs with laplace approximation of covariance.
//   From @p.murphyProbabilisticMachineLearning2022
//   + Using probabilistic approach since it is the optimal approach to decision making under uncertainty
//   + It is common language that we can use to connect with other fields.
// ]

== The Way of Statistical Machine Learning <sec:sml-way>

By itself, @sml does not provide a tool that can be used with @boed out of the box, but it is certain to pave the way toward an elegant tool. Also, it is simple, easy to understand, and computationally inexpensive in the early days of characterizing quantum devices. Thus, from the development point of view, starting our journal here would be considered a solid step along this path.

In literature @p.murphyProbabilisticMachineLearning2023, a term *predictive model* is referred to a function that given $bold(x)$, predict output $y$, formally $y = f(bold(x))$ that is constructed from some dataset. In our context, characterizing quantum device is to construct a *predictive model* from the data obtain from experiment. In the information point of view, we extract the information of the system via experiment, encode it to a model, then we can predict the output of the target device using the model instead of perform new experiment on the target device. I will refer a procedure of estimating predictive model from dataset by the word train / fit / estimate / construct. Given large enough dataset, we can fit the model with high accuracy. However, if we have to spend much of a resource to characterize the device, it would be impractical. Thus, characterizing quantum device by exploit the knowledge of quantum mechanics and system we use to minimize the resource needed to construct an expensive model is preferred.

One of the simple way of characterize quantum device is to *assume* functional form of the device to characterize. The *functional form* of the quantum device may consists of multiple functions. In the context of modeling the system in the quantum mechanics, it is convenient to model the system in term of a Hamiltonian. In quantum mechanics, Hamiltonian is the mathematical object that describe a dynamics of the quantum system and its interaction with the environment in the case of open system, if not then it is closed system. In our context of characterization, we model the system with functional form of Hamiltonian. That is the mathematical form consisting of Hermitian matrices, and the the functions that indicate the finer detail. The function may be a time dependent, $H_(1)(t)$ or independent $H_(0)$ or both. Thus, a general form of Hamiltonian functional form $H(t)$ is,
$
  H(t) = H_(0) (t) + H_(1) (t)
$
The functional form above left the dependent on the system parameters implicit. In practice, the function may depend on a lot of system parameters, thus, in this thesis, I would like to be explicit when the parameters are the focus to avoid symbols cluttering.

Now, I would to go into the some of the detail of the functions combined into the total Hamiltonian. Starting by the most important function, consider control parameters $#control$, the control function $s(#control, t)$ is a time-dependent function. For example, in superconducting platform, the control function may be a function of the signal which is a function of pulse envelope itself. The next one is a function of noise, that describe how noise interact with the system, the noise function itself might be a functional of the control function. The time-independent noise function may be modeled as a constant shift of sort. Such as a detuning, that model a shift of qubit frequency to its original value. The function can be chose with freedom, given that the total Hamiltonian still remain Hermitian. For example, the time-dependent Hamiltonian might be a control act on the Pauli-X axis $H_1(t) = s(#control, t) hat(sigma)_X$.

Model the device at the level of Hamiltonian is one of the possible way. However, Hamiltonian is not the observed object in the experimental setting. Hamiltonian has to be translated to the Unitary operator by solving the differential equation, either Schrödinger equation in the case of closed system or beyond like the Master equation in the case of open system. We have two choices of the final product to solve for, (1) the final quantum state $rho_f$ from $rho_0$ and (2) the Propagator / Unitary $hat(U)$. Furthermore, since quantum information required extraction via measurement, the observable $hat(O)$ has to be chosen. The expectation value of the observable is then,
$
  expval(hat(O)) = Tr [ hat(O) rho_f] = Tr [ hat(O) hat(U) rho_0 hat(U)^dagger ].
$
The choice between solving for final state or Propagator typically decided by the object of interest. For instance, if the expectation value of observable of multiple possible choice of initial state given control is the main focus, then computing Propagator would be more efficient as we have to solve the differential equation just once instead of multiple time for each initial state.

The expectation value expression allows us to be one step closer to the realistic setting. The dependent on the control parameter is implicit for Propagator and final state in the expression. However, it is also implies that the expectation value is a function of the control parameters. Thus, open the path for us to model the target device at the level of higher order functional form, i.e. a function of control parameters that return the expectation value. In general, we can write the predictive model $caron(f)$ of expectation value $expval(hat(O))$ given control parameter $#control$ as
$
  expval(hat(O)) approx caron(f)(#control, rho_0, hat(O)).
$ <eq:predictive-model>
The next step can be either directly implement predictive model as a fully Blackbox such as using Deep Neural Network or impose the prior knowledge about the system.

=== How to train your SML

Training predictive model required the input $bold(x)$ and the output $y$, or referred as feature and label in ML literature respectively. Each pair of input and output is called sample. Dataset form of $N$ samples. The input may be a vector of real number. In the case of complex input, the real and imaginary part can be separated and stored as a real number. The sample can have multiple output too. This is common, for example in the image generation task.

We can infer from the form of the predictive model of quantum device for what our dataset should look like. The input is the control parameters $#control$, where the it can be a vector, i.e. multiple input features. The output label is the expectation value calculated from the measurement readout of the quantum device. Since measuring quantum device only yield binary outcome as a result each time, multiple execution of the same input have to be carry out to obtain an ensemble of binary outcomes. Those binary values are then used to calculate expectation value. Formally, $n$ binary outcomes $b in { 0,1 }^n$ is mapped to its eigenvalue values corresponded $s in { 1, -1}^n$ and expectation value of observable $O$ for given initial state $rho_0$ is calculated by,
$
  estexpval = 1 / n sum_i^n s_i
$
So far, I have not specify the size of the quantum system just yet. Note that the observable and initial state are of arbitrary size. Here the size is referred to the dimension of Hilbert space, not to be confuse with sample size. For the sake of clarity, I will refer to the size of Hilbert space in term of the number of qubit, otherwise will be specified explicitly. The number of qubit of the initial state and observable depends on the characterization method and device. For example, to characterize single qubit system, initial state and observable will have size of $2 times 2$ matrix. For two qubits system, we dealing with $4 times 4$ matrix. The size of system grow exponentially with the number of the qubit. However, the characterization and calibration relied on classical computer, that their memory cannot keep up with the quantum system. Thus, efficient characterization method is necessary. It should capture the interplay between qubits interaction and system-environment interaction is essential without having to compute un-realistic large size of matrix.

The dataset defined above provides a complete information about the system. In other words, a quantum channel can be reconstructed using each sample, given that the initial state and observable pairs are chosen sufficiently, e.g. process tomography. It is not necessary for every characterization method to collect aforementioned dataset. For example, a predictive model that have explicit mathematical form required specialized experiments to estimate system parameters, for example clear-box approach @fyrillasScalableMachineLearningassisted2024. The explicit model characterization approach has an advantage of being able to characterized with less data consumption, but the flexibility is limited.

There are varieties of regression model that can be used. The go-to choice for the task with less priors knowledge is the family of Deep Neural Network model. Intuition and nature of data then can be acquired by researchers to enforce more informed model. Particularly in our case, quantum mechanics is known. The dynamics of system can be solved explicitly given that the Hamiltonian of the system is fully known. The solution of the differential equation can be solved by using numerical solver of estimated by using other technique such as Physics-Informed Neural Network (PINN). One can also model the system with known part and parametrize the unknown part. One of the example, Graybox characterization method @youssryExperimentalGrayboxQuantum2024 @youssryCharacterizationControlOpen2020 will be the primary case study to be discussed in @sec:graybox-characterization.

In the case that the quantity of interest is derived expectation value of observable, one might higher order model in term of expectation value outputted model. For example, fidelity can be calculated by expectation value, whether it is state fidelity, process fidelity, or average gate fidelity. In the case of model that predicting mathematical object prior to expectation value of observable such as final stare, or unitary, one may model higher model outputting the expectation value in term of lower order model. The central pivotal role of expectation value is because it is the quantity that is estimated directly from the binary measurement results. While binary outcome is closer to the low level implementation, it is random variable by nature. However, modeling a deterministic outcome would be more straightforward. While a probability of observing each outcome is invertible map of expectation value of Pauli observable. Thus, expectation value as an output label is reasonable choice.

In statistical machine learning paradigm, there are several methods to fit the predictive model. However, the most prominent one is Empirical Risk Minimization (ERM), that is to find the model parameters $theta^*$ that minimize a loss function $l$, averaging over the dataset of size $N$. Formally, the loss function is,
$
  cal(L) (theta) = sum_i^N cal(l) (x_i, y_i; theta)
$
and we want to solve for,
$
  theta^* = argmax_(theta in Theta) cal(L) (theta)
$
The word *empirical* indicate that this is a value calculated from the experimental dataset.

The most generic loss function for the regression model training is perhaps a mean squared error (MSE) of the output label function. In the case of expectation value of observable $EE[ hat(O) ]_(rho_0)$, the loss function $cal(l)(#control, expval(hat(O))_(rho_0) ; theta)$ can be defined as,
$
  cal(l)(#control, expval(hat(O))_(rho_0) ; theta) = ( estexpval - caron(f)(#control, rho_0, hat(O); theta) )^2
$
MSE-based loss function is the function that output the value in the unit of square of the output label. The loss function is then, form from the average over the experimental settings of initial state $rho_(0)$ and observable $hat(O)$ that provide sufficient information for system characterization,
$
  #mseeloss = 1 / K sum_(hat(O), rho_0) (estexpval - caron(f)(#control, rho_0, hat(O)) )^2
$
While another one is a absolute error based function of the form,
$
  cal(l) (#control, estexpval; theta) = abs(estexpval - caron(f)(#control, rho_0, hat(O))).
$
The AE-based loss function output the value of the same unit as the label. In fact, the choice of loss function is rather arbitrary. It is depended on the use case of the predictive model. For example, if only the fidelity is the quantity of interest, one may define loss function in term of fidelity directly.

=== Graybox Characterization Method <sec:graybox-characterization>

Nonetheless, choosing MSE-based loss function does has advantage of analysis. I will discuss the benefits in @sec:learn-from-sml.

After loss function selecting, the model can be fitted using any suitable optimization algorithm. Implementation details will be discussed in @sec:inspeqtor.

After the model training, assessing its performance is the next step. This is also including the finding the model that perform best. As different initial model parameter typically yield different model. Furthermore, the complexity of the model can also affect the performance as well as speed and resource needed for training. For example of Deep Neural Network model, regardless of model architecture, the number of trainable parameters contribute to the resource needed for model training. Also during the prediction phase, large model would required more time to evaluate. Usually, the problem of select the most suitable model that perform well with reasonable size is tackled by the mean of *hyperparameter tuning*, a.k.a. *model selection*. This is the blackbox optimization over space of possible model that can predict given dataset. More detail will be discussed in @sec:inspeqtor. But the discussion here is a prelude of @sec:learn-from-sml.



In this section, I will discuss the main characterization method that will be used throughout this thesis, the Graybox characterization method. First, the motivations behind the choice of the approach will be listed. Second, the mathematical formulation of the method will be reviewed. The architecture of model that can be used with the method will be discussed.

Graybox characterization method is chosen for several reasons. (1) The method allows for partial implicit model of the system from its mathematical formulation. The ideal Hamiltonian must be explicit. However, the noise can be implicit and represented by regression model instead. This is very powerful method as the explicit noise model can be hard to identify in experimental setting. (2) The control function can be chosen. The freedom of control allows for model feature engineering, and applying specialized control methodologies, exploit the prior knowledge of the platform. Moreover, only the control related to the application of interest can be selected, reducing the control space to be explored. For example, for Quantum Noise Spectroscopy (QNS), only CPMG pulse is the logical choice. (3) It is a great example of characterization method that combines both explicit and implicit solution together. The hybridization shows the potential and flexibility of the studies done in this thesis for other approach that may be fully mathematical implicit or explicit.

Before proceeding to the mathematical formulation of the Graybox method, I would like to remark on the fundamental concept of the Graybox first.

#quote[Graybox model consist of known explicit part, namely *Whitebox*, and unknown implicit part, namely *Blackbox*. The former can be calculated by using knowledge from physics, where the later can be learn by statistical learning method such as Deep Neural Network.]

Now we are going to review the foundation that we will build upon to in @youssryCharacterizationControlOpen2020. Note that, however, this is not a only way to model a Graybox characterization method. Let us first consider a total Hamiltonian $hat(H)_("total")$ of the system that is the sum of an Ideal Hamiltonian $hat(H)_("ideal")$ and a noise Hamiltonian $hat(H)_("noise")$ of the form,
$
  hat(H)_("total") (#control, t) = hat(H)_("ideal") (#control, t) + hat(H)_("noise") (#control, t)
$
The time and control dependent form is not a requirement, as it can be time and control independent. But I kept the time-dependent form for the generality propose, and drop the explicit dependent on control for the sake of cleanness. The Propagator of the total Hamiltonian is a time-ordered Propagator obtained by solving Schrodinger equation $hat(U)(t) = cal(T)_(+) exp { -i integral_(0)^(t) hat(H)_("total")(s) d s }$

The first step of Graybox approach is to transform the total Hamiltonian to interaction picture by using the time-ordered Propagator $hat(U)_("ideal")(t) = cal(T)_(+) exp { -i integral_(0)^(t) hat(H)_("ideal")(s) d s }$. The total Hamiltonian in the new frame becomes,
$
  hat(H)_I (t) = hat(U)^dagger_("ideal")(t) hat(H)_("noise")hat(U)_("ideal")(t).
$
Similarly, the time-ordered Propagator is $hat(U)_I(t)$. Total Propagator in the Schrodinger equation is $hat(U)(T) = hat(U)_("ideal")(T)hat(U)_(I)(T)$. However, the initial state is operated first by the interaction Propagator, resulting the dependent on the initial state. The Graybox method rearrange the expression by mathematics manipulation, where the interaction Propagator is rotated by ideal Propagator as,
$
  hat(U)_(I, R)(T) = cal(T)_(+) exp { -i integral_(0)^(T) hat(U)_("ideal")(T) hat(H)_I(s) hat(U)^dagger_("ideal")(T) d s }.
$ <eq:toggling-unitary>
Thus, the total Propagator become,
$
  hat(U)(T) = hat(U)_(I, R)(T) hat(U)_("ideal")(T).
$ <eq:noisy-unitary>
The expectation value become,
$
  idealexpval = Tr [ hat(U)_(I, R)(T) hat(U)_("ideal")(T) rho_0 hat(U)^dagger_("ideal")(T) hat(U)^dagger_(I, R)(T) hat(O) ],
$
using the property of trace, we can define an operator containing the noise as,
$
  hat(V)_(O) (T) = expval(hat(O)^(-1) hat(U)^dagger_(I, R)(T) hat(O) hat(U)_(I, R)(T))_c.
$ <eq:vo>
Resulting in the expectation value of observable of a following form,
$
  idealexpval = Tr [ hat(V)_(O) (T) hat(U)_("ideal")(T) rho_0 hat(U)^dagger_("ideal")(T) hat(O) ]
$
Observe that the expectation value above isolate the noise operator from the ideal operator. We can further simplify the expression by define a noisy observable $hat(W)_(O)$ by notice that the Pauli observable is constant over the classical expectation value,
$
  hat(W)_(O)(T) = expval(hat(U)^dagger_(I, R)(T) hat(O) hat(U)_(I, R)(T)).
$ <eq:wo-operator>
The noisy observable is a Hermitian matrix that is dependent on both ideal and noisy operators. With the noisy observable, the expectation value of observable can be written as
$
  idealexpval = Tr [ hat(W)_(O) (T) hat(U)_("ideal")(T) rho_0 hat(U)^dagger_("ideal")(T) ].
$ <eq:exp-wo>
The Graybox model is then defined over the noise and ideal operators. The ideal evolution which is explicitly known is a *Whitebox*. While, the implicit unknown part is a *Blackbox*.

As discussed in @sec:hamiltonian-to-unitary, the closed-form of solution to the Schrödinger equation is not generally solvable via analytical method, except for simple cases. Thus, numerical methods are needed for the calculation of the solution. The implementation of the ODE solver will be discussed later in @sec:inspeqtor. The main ways to obtain solution including numerical ODE solver, Trotterization, or Physics Informed Neural Network (PINN). Regardless of the numerical method used, accuracy of the Whitebox is also important as the accuracy of Whitebox plays affect the accuracy of the Graybox predictive model.

Formally, *Whitebox* is the component in the *Graybox* that has the explicit mathematical procedure. For example, the procedure of calculating Propagator and then transform it to measurable quantity can also be considered *Whitebox* @youssryModelingControlReconfigurable2020 @youssryExperimentalGrayboxQuantum2024.

For the *Blackbox* component, this is the implicit part that estimate the unknown part. The implicit functional form to be estimated must be identified first to be able to chose the appropriate statistical learning method. The function should be part of the overall functional form of the Graybox. From the mathematical formulated above, the Blackbox is the noisy observable as a function of control parameters $#control$,
$
  hat(W)_(O)(#control ;T) & = expval(hat(U)^dagger_(I, R)(#control ; T) hat(O) hat(U)_(I, R)(#control ; T)) \
                          & = tilde(cal(b)) (#control)
$ <eq:wo>
Here, the dependence on control parameters come from the fact that $hat(U)_(I, R)(#control ; T)$ depends on $hat(U)_("ideal") (#control)$ which is a function of ideal Hamiltonian. Thus, the output of the Blackbox is Hermitian matrix of the same size as the Hilbert space of the system, e.g. $2 times 2$ matrix for a qubit system. If the statistical learning method of choice is Neural Network, one can define the output layer of the model to has $2k$ number of neuron, where $k$ is the number of element of Hermitian matrix. The factor of $2$ is for real and imaginary components of each matrix entry. However, one can also model the output layer to be parameters that parametrized Hermitian matrix by using its decomposition, the number of output neuron can be decreased in this manner.

In this thesis, Deep Neural Network as Blackbox is the main focus as it can be easily promote to Bayesian Neural Network. The details will be discussed in @sec:pml-way. In previous works, the architecture of Blackbox is identified based on the priors knowledge of the problem. For instance, if the input is a sequence-like object, Recurrent Neural Network (RNN) based is used, for its capability to deal with time series information. One of the example is the control that apply in sequence, the accumulate consequence is expected to be captured by RNN architecture @youssryCharacterizationControlOpen2020 @youssryModelingControlReconfigurable2020.
In Ref. @youssryCharacterizationControlOpen2020, they model a simulated device by using the LSTM (one variation of RNN) as the Blackbox to map control parameters to the noise operator defined in @eq:vo. Then the Whitebox map the results and the system information to the expectation values.
In Ref. @youssryModelingControlReconfigurable2020, they model the reconfigurable photonic circuit by using GRU (one variation of RNN architecture) as a Blackbox to to model the map between time series of control voltage to the interaction Hamiltonian. Then the Whitebox transform the interaction Hamiltonian to power loss during the measurement.
In the case of simpler control, #cite(<youssryExperimentalGrayboxQuantum2024>, form: "author") model the quantum photonic circuit by using a simple @mlp as the Blackbox to map the control to the static Hamiltonian. Then Whitebox transforms the Hamiltonian to the probability of observing measurement results.

There are multiple possible choices of Blackbox and Whitebox depends on how we model the physical system. Each choice may has its own advantage. For example, Blackbox can model the operator defined in @eq:wo as well instead of @eq:vo. Modeling @eq:wo resulting in a simpler computation procedure than @eq:vo. Another example is to model the @eq:toggling-unitary, which is even more simpler. Moreoever, modeling the @eq:toggling-unitary has additional advantage such that we can simultanousely characterize SPAM noise. Furthermore, to completely eliminate the heavy calculation of solving Schrödinger equaiton, we can also model the @eq:noisy-unitary which is the total unitray directly.

Combining both boxes, the Graybox model is successfully constructed. The Whitebox does not need the training phase as it can calculate the output from input directly from the mathematical procedure. While, the Blackbox needed the training phase by minimizing the loss. The performance or the accuracy of the Graybox model can be evaluated by testing on the unseen experimental data. In ML practice, dataset would be spilt to training, validating, and testing subset. Usually, the testing dataset is used for model to evaluate the test loss after the model selection and represent the expected accuracy of the model against the actual evaluation time.

=== What can we learn from SML approach <sec:learn-from-sml>

Loss function basically told us what did this model trained to do. For the loss function averaging over the dataset based on MSE of the expectation value of observables $#mseeloss$, the model is trained to predict expectation value of observable as close as possible to the actual value produced from real device. However, the value of $#mseeloss$ after the model training does not has self-contained meaningful physical interpretation. For example, in the image classification task, the accuracy can be easily interpret by counting the correct predictions and divide the number by the total number of test dataset, the model will have associate performance quantity, e.g. $73.4\%$. We know that the lower $#mseeloss$, the better, but at what particular value that the model can be used in practical usage?

One of the practical application of predictive model is to be used in control calibration by open-loop approach. The usage of predictive model to produce high performance physical gate is important. Because, quantum error correction protocol require physical gate to perform with error rate below some noise threshold, otherwise the logical error rate cannot be improved. Thus, the predictive model expected to predict at high accuracy and precision. Thus, the prediction error in term of gate performance from actual experiment serve as a better performance metric.

One can define loss function that is defined based on fidelity, which can be interpreted directly as a closeness of prediction and experiment as a scale of fidelity from zero to one. For instance, the expectation values can be used to calculate @agf. Thus, the loss function defined based on the @agf predicted by the model $macron(F)_("model")$ and the experimental @agf $macron(F)_("exp")$ with respect to $hat(U)_("ideal") (#control)$ can be defined as,
$
  aefloss = abs(macron(F)_("model") - macron(F)_("exp")).
$
This loss function allows us to interpret the model as it trained to predict the expectation values that produces the @agf that match of value calculated from expectation values from experiment.

==== Decomposition of MSE

One of the challenge of training machine learning is determining when to stop model selection, i.e., determining if this model with this model parameters is good enough for practical application.
In the following, we will analyze the model trainability. Let us consider quantum experiments composed of qubits where only binary measurement outcomes are observed, e.g., $+1$ and $-1$ for qubit measurements. The expectation value given an initial state and a set of control actions, $expval(hat(O))_control$, can be estimated by taking an average of the measurement outcomes over $n$ executions of the same circuit. We refer to $n$ as the number of shots or simply shots and denote the $n$-shot estimate of the expectation value by $bb(E)_n [hat(O)]_control$. The finite number of shots is one of the sources of uncertainty in the experimental setting, causing the model in @eq:predictive-model to be inaccurate. Typically, one would need to minimize a loss function as low as possible; however, with the finite-shot training data, there might be limits on the accuracy of the trained model and its predictive performance. In this section, we will answer the question of what one can expect in model training given a finite-shot dataset.

In order to search for an optimal predictive model for @eq:predictive-model, we consider a standard choice for the loss function in machine learning for regression tasks, which is the square error (SE). The model $caron(f)(control) = expval(hat(O))^"model"_(rho_0, control)$ (noting that we have dropped the dependency on the initial state $rho_0$ and observable $hat(O)$ for notational simplicity) should predict the expectation value as close as possible to the exact (infinite-shot)
expectation value $expval(hat(O))^"exp"_(rho_0, control)$ by learning from the experimentally obtained (finite-shot) expectation value, $bb(E)_n [hat(O)]^"exp"_(rho_0, control)$, for a given control $control$. For the SE loss function, we write the loss as a squared difference between the two quantities,
$
  cal(L)_"SE[E]" equiv 1/M sum_(k, rho_0) ( expval(hat(O))_"model"_(rho_0, control) - bb(E)_n [hat(O)_k]^"exp"_(rho_0, control) )^2
$ <eq:msee>
where the purpose of the sum is to average over a set of $M$ different combinations of observables $hat(O)_k$ as well as initial states $rho_0$.
We use the subscript SE[E] to indicate that the loss is an average squared error of the expectation values of the quantum observable, where the mean here refers to the average over the combinations of initial states and observables.
To approximate the lowest possible value of $cal(L)_"SE[E]"$ given the finite-shot data, we use the decomposition
of the square error @bishopPatternRecognitionMachine2006, applying to an example of the predictive model similar to $caron(f)(control)$ in @eq:predictive-model.
Let us consider the input $X$ as a random variable sampled from a probability distribution $X ~ p_X(x)$, and the output $Y$ as a random variable sampled from a probability distribution $Y ~ p_(Y|X)(y|x)$. The predictor $caron(f)(x) = caron(f)(x; cal(D))$ (not necessary) trained on the fixed size dataset $cal(D) = { (X, Y) }$ and kept fixed should minimize a mean square error given by,
$
  bb(E)_(x, y)[ ( y - caron(f)(x))^2 ] = bb(E)_x [ (caron(f)(x) - bb(E)[y|x] )^2 ] + bb(E)_x [ "Var"(y|x) ]
$ <eq:bias-var-decomp-alt>
The expectations in @eq:bias-var-decomp-alt are with respect to all possible values of input and output. Note that both terms are positive.
@eq:bias-var-decomp-alt is minimized when the predictor is optimal, in other words it predicts the expected value of the output for a given input $caron(f)(x) = bb(E)[y|x]$, reducing the first to zero.
However, the last term, the expectation of conditional variance of the distribution of $Y$ given the input value of $X = x$ over input space $x$, is considered an irreducible error as it is inherited in the observed data $Y$. Thus, the expected conditional variance is the lower bound of the square error, $bb(E)_(x,y)[ ( y - caron(f)(x))^2 ]$, achievable by the optimal predictor.

In our context of @eq:predictive-model,
our model $caron(f)(control)$ should minimize $cal(L)_"SE[E]"$ in @eq:msee where the observed data is the finite-shot expectation value, i.e., $Y = bb(E)_n [hat(O)]^"exp"_(rho_0,control)$.
Therefore, we can apply the decomposition in @eq:bias-var-decomp-alt, replacing $x$ with $control$, $caron(f)(x)$ with $expval(hat(O))^"model"_(rho_0, control)$, taking the average over all possible initial states and observables. We find that the conditional variance in @eq:bias-var-decomp-alt is given by
$
  "Var"(y|x) mapsto "Var"(bb(E)_n [hat(O)]^"exp"_(rho_0,control) | control)
$ <eq:msee-boundby-var>
which, the expectation over the control variable should give us an approximated lower bound for the loss in @eq:msee. The conditional variance $"Var"(y|x)$ could be extracted directly from the experimental data. Given particular control parameters, observables, and initial states, a large number of experiments need to be executed so that one can approximate the exact conditional variance. Then, the sample variance should be averaged over all combinations of initial states and observables. However, this method is costly as it requires additional experiments just to estimate the conditional variance. We instead derive an analytical expression of the conditional variance using the variance of a quantum observable based on Bernoulli distribution.
The variance of an estimation of an observable $hat(O)$ from $n$ measurement shots is
$
  "Var"(bb(E)_n[hat(O)]^"exp"_(rho_0,control) | control) = 1/n (1 - expval(hat(O))_(rho_0,control)^2 )
$ <eq:var-observable>
given the control variables $control$ and the initial state $rho_0$ fixed when evaluating the expectation value.

We then calculate the bound for our single qubit case, where, we consider a set of observables $hat(O) in { hat(X), hat(Y), hat(Z) }$ which fully characterize a quantum state and a set of pure initial states $rho_0 in { ket(+)bra(+), ket(-)bra(-), ket(i)bra(i), ket(-i)bra(-i), ket(0)bra(0), ket(1)bra(1) }$. Then, assuming that the control $control$ only produces unitary operations and all initial states are pure states, we can apply $expval(X)^2_(rho_0, control) + expval(Y)^2_(rho_0, control) + expval(Z)^2_(rho_0, control) = 1$ and obtain the lower bound of an expected loss of @eq:msee as
// #block(label: <eq:var-data>)[
//   #align(
//     bb(E)_(x,y)[cal(L)_"SE[E]"] &>= bb(E)_x[ 1/(18n) sum_(rho_0) sum_k ( 1 - angle(hat(O)_k)_(rho_0, control)^2 ) ],
//     & = bb(E)_x[ 1/(18n) sum_(rho_0) 2 ] = 2/(3n)
//   )
// ]
$
  bb(E)_(x,y)[cal(L)_"SE[E]"] &>= bb(E)_x [ 1/(18n) sum_(rho_0) sum_k ( 1 - expval(hat(O)_k)_(rho_0, control)^2 ) ],
  & = bb(E)_x [ 1/(18n) sum_(rho_0) 2 ] = 2/(3n)
$ <eq:var-data>
We refer to the average of the conditional variances (irreducible errors), which is independent of the choice of control action as _data variance_. We note that the result does not depend on the number of initial states $rho_0$ taken in the averaging process.

We now discuss how to use the data variance in practice.
Since, the expected loss on the LHS of @eq:bias-var-decomp-alt is the expectation with respect to joint distribution $p_(X, Y)(x, y) = p_(Y|X)(y|x) p_X(x)$, we can estimate it by averaging the squared error over samples from the distribution $p_(X, Y)(x, y)$. Since the $x,y$ samples will form a dataset, we denote $bb(E)_({X, Y})[ dot ] -> bb(E)_"dataset" [ dot ]$.
In the subsequent calculations, the samples can be taken from the training or testing dataset.
Then, we can calculate the LHS of @eq:bias-var-decomp-alt by averaging the squared error over the dataset using the predictive model.
Consequently, the optimality of the predictor, i.e., first term on the RHS of @eq:bias-var-decomp-alt, can be calculated from the empirical value of expected loss, i.e. LHS of @eq:bias-var-decomp-alt, subtracted by the data variance.
Thus, the data variance, which is an analytical expression of the irreducible error, would provide additional information, complementing the empirical result in the analysis of the predictive model.
For instance, we could design an optimization stopping criterion in the predictive model construction process based on the empirical optimality of the predictor.

With @eq:var-data, we know that the expected loss should be at most reduced to $2/(3n)$, which also decreases as the number of shots $n$ increases. In practice, we construct a predictive model by minimizing @eq:bias-var-decomp-alt over the training dataset $bb(E)_"train" [cal(L)_"SE[E]"]$. However, care must be taken when selecting a model based on expected training loss. For example, in the case of using a neural network as a part of a predictive model, it may be possible that the model is large enough to overfit the training dataset, but the expected loss approach $2/(3n)$ results in selecting a non-optimal model. Thus, it is considered a good practice to evaluate the expected loss using (unseen) testing dataset $bb(E)_"test" [cal(L)_"SE[E]"]$. Therefore, we expect that selecting a model that predicts the expected loss over the training and testing datasets simultaneously, which also approximately approaches the data variance, is a first guiding step in model selection.


// Bias-Variance Decomposition (BVD) of Mean Squared Error is an excellent analysis that we need @jamesIntroductionStatisticalLearning2023. Suppose that our sample in the dataset ${vx, y }$ from the experiment can be modeled as $y = f(vx) + epsilon.alt(x)$ is a true function $f$ where $epsilon.alt(x) tilde cal(N) (0, sigma(x))$ follow the Gaussian distribution. Note that $epsilon.alt(vx)$ is a function of $vx$. Given pair of data labelled $\{ vx, y \}$, and the model $caron(f)$ used to estimate the true function, the expected MSE loss can be expressed as,
// $
//   "MSE" & = EE_(cal(D)) EE_(epsilon.alt)[y - caron(f)(vx)]^2, \
//         & = EE_(cal(D), epsilon.alt) [y - caron(f)(vx)]^2.
// $
// Now, expand the MSE yield
// $
//   EE_(cal(D), epsilon.alt) [y - caron(f)(vx)]^2 &= EE_(cal(D), epsilon.alt) [(f(vx) - caron(f)(vx) + epsilon.alt(vx))^2] \
//   &= EE_(cal(D), epsilon.alt) [underbrace((f(vx) - caron(f)(vx, cal(D)))^2, "first") + underbrace(2(f(vx) - caron(f)(vx; cal(D)))epsilon.alt(vx), "second") + underbrace(epsilon.alt(vx)^2, "third") ].
// $
// Consider the first term, we follows the typical analysis @gemanNeuralNetworksBias1992,
// $
//   EE_(cal(D), epsilon.alt) [ (f(vx) - caron(f)(vx, cal(D)))^2 ] &= EE_(cal(D)) [ (f(vx) - caron(f)(vx, cal(D)))^2 ] \
//   &= ["Bias"(caron(f)(vx; cal(D)))]^2 + "Var"[caron(f)(vx;cal(D))].
// $
// By noticing that the true model and predictive model does not depends on the $epsilon.alt$, we now have the bias and variance of the predictive model as usual. For the second term, we have
// $
//   EE_(cal(D), epsilon.alt) [2(f(vx) - caron(f)(vx; cal(D)))epsilon.alt(vx)] &= 2 ( EE_(cal(D)) [underbrace((f(vx) - caron(f)(vx) ), "does not depends on " p(epsilon.alt)) underbrace(EE_(epsilon.alt) [epsilon.alt(vx)], 0) ] ) \
//   &= 0.
// $
// Finally, the third term, we have,
// $
//   EE_(cal(D), epsilon.alt) [epsilon.alt(vx)^2] = EE_(cal(D))[ (sigma(vx)^2 ] = EE_(cal(D))[var(epsilon.alt(vx))],
// $
// which is the expected value of the variance of random variable $epsilon.alt(vx)$ over multiple dataset ${cal(D)}$. Note that, in our case, we left the expected value over dataset as it is since the variance of noise depends on the choice of input $vx$.

// The Variance and Bias of the predictive model can be reduced by model selection, and is referred to as *reducible error*. However, the last term is a variance of data, this term cannot be reduced by the model1 selection, as the error is inherit within the data itself. Hence, its name *irreducible error*. In this thesis, we will refer to this quantity as *data variance*. Our main focus is *data variance* (otherwise, we would not give it a special name). As the first two terms depend on the choice of the statistical learning method, further investigate might require extensive numerical studies. On the other hand, data variance, i.e. variance of the expectation value of quantum observable can be analytical analyzed. Furthermore, irreducibility of the data variance imply that, it is a expected minimum MSE of expectation value of observable. Mathematically, we apply decomposition to each term of $#mseeloss$, and group the data variances terms together, resulting in,
// $
//   EE_(cal(D), epsilon.alt) [#mseeloss ] >= EE_(cal(D)) [1 / K sum_(k, rho_0) var (epsilon.alt)_(rho_0, hat(O)_k)].
// $
// In the experiment setting, expectation value of quantum observable $idealexpval$ can not be directly observed. Instead, it was estimated from the binary measurement results. An act of *estimating expectation value of quantum observable using $n$-shots measurement results* is equivalent to sampling from Gaussian distribution $cal(N) ( idealexpval , sigma)$ with mean at the true expectation value $idealexpval$ and variance @kohdaQuantumExpectationvalueEstimation2022,
// $
//   var (hat(O)_(rho_0)) = 1 / n (1 - idealexpval^2) .
// $ <eq:sample-var-expval>
// By sampling from the distribution, this quantity can be estimated by using classical variance. In practice, one needs to perform experiments to estimate $#idealexpval$ for $k$ times, and use $k$ samples of all possible combinations of observables and initial states to calculate sample variance. it is expensive, yet straightforward to estimate data variance.

// However, for single qubit case, we can simplify the expression to obtain an analytical expression. The combinations of initial states $rho_0 in { ketbra(+), ketbra(-), ketbra(i), ketbra(-1), ketbra(0), ketbra(1)}$ and observables $hat(O) in { hat(X), hat(Y), hat(Z)}$ are selected, in total of 18 combinations.

// The data variance of our dataset is an average of the variance of quantum observable of all combinations, that is,
// $
//   1 / 18 sum_(k, rho_0) var (epsilon.alt) = 1 / 18 sum_(k, rho_0) 1 / n (1 - chevron.l hat(O)_k chevron.r_(rho_0)^2) .
// $
// By the normalization condition, given Unitary operator and pure initial state fixed, the Pauli observables are constraint with $chevron.l hat(X) chevron.r_(rho_0)^2 + chevron.l hat(Y) chevron.r_(rho_0)^2 + chevron.l hat(Z) chevron.r_(rho_0)^2 = 1$. Now, we expand the summation of the Pauli observable first,
// $
//   1 / 18 sum_(k, rho_0) var (epsilon.alt) = & 1 / (18 n) sum_(rho_0) [3 -(chevron.l hat(X) chevron.r_(rho_0)^2 + chevron.l hat(Y) chevron.r_(rho_0)^2 + chevron.l hat(Z) chevron.r_(rho_0)^2)] \
//   = & 1 / (18 n) sum_(rho_0)(3 - 1) \
//   = & 2 / (3 n),
// $
// then the simplification can be made by notice match of the normalization condition. We now left with the data variance,
// $
//   EE_(cal(D), epsilon.alt) [#mseeloss] >= EE_(cal(D))[1 / 18 sum_(k, rho_0) var (epsilon.alt)] = 2 / (3 n),
// $ <eq:data-variance>
// that is an lower bound of expected minimum value of MSE of expectation value. The most elegant feature of the bound is that, it is not depends on the control parameter and depends on the number of shot only. *In the case of larger Hilbert space*, the data variance can be derived in the similar manner.

// The results can be incorporate in the model selection, such as model accepting criteria, i.e. if test loss is not close to expected minimum value, we might want to explore a space of more complex statistical learning method to reduce reducible errors. I will refer to the model that achieve test $EE [ #mseeloss ]$ close to theoretical minimum point as Finite-Shot Sub-Optimal (FSSO) model. Moreover, expected minimum value can also be preselected by choosing the value of shots for characterization dataset generation. This is also equivalent to choosing the upper bound of performance in @agf prediction as shown in @eq:agf-upper-bound. However, at the theoretical expected minimum loss the bound becomes,
// $
//   (macron(F)_("model") - macron(F)_("exp"))^2 <= 1 / (2 n) .
// $
// However, it is assuming that the finite-shot expectation value of observable follows the theoretical distribution, i.e. has the mean at true expectation value, and variance following @eq:sample-var-expval.

// In the regard of data-efficiency, we want to construct FSSO model without consume too much sample size. One of the way is to treat sample size as a hyperparameter, and tune it for the minimum number that model can be FSSO model. However, it might imply that we have to have a sufficient large dataset already. Another way is to start model selection with low sample size, then perform new experiment to increase the sample size by small amount until FSSO is successfully constructed. The later approach requires active experiment with target device. Moreover, small sample size might cause model to overfitted, i.e. train loss is not approximately equal to test loss.

==== Performance bound

Let us start with the definition of AGF between an ideal unitary $hat(U)$ parametrized by control variable $#control$ (not shown explicitly) and a map $cal(E)$. Let $rho = |psi chevron.r chevron.l psi|$ be an arbitrary pure state,
an average of state fidelity between $hat(U)^dagger |psi chevron.r chevron.l psi| hat(U)$ and $cal(E) (|psi chevron.r chevron.l psi|)$ integrating over all possible $|psi chevron.r$, which is
$
  macron(F)(cal(E), hat(U)) = integral d psi chevron.l psi hat(U)^dagger cal(E) (|psi chevron.r chevron.l psi|) hat(U) |psi chevron.r
$

Typically, we are interested in the case where the map $cal(E)$ is an experimental realization of $hat(U)$.
For the single qubit case, the system dimension is $d = 2$, and the AGF can be written @nielsenSimpleFormulaAverage2002 with a state decomposition on the Pauli matrices as
$
  macron(F)(cal(E), hat(U)) &= 1/2 + 1/12 sum^3_(j=1) sum^1_(k=0) alpha_(j,k) Tr[ hat(U) P^dagger_j hat(U)^dagger cal(E) (rho_(j,k)) ] #<eq:direct-agf> \
  &= 1/2 + 1/12 sum^3_(j,m=1) sum^1_(k=0) alpha_(j,k) beta_(m,j) Tr [ hat(P)_m cal(E) (rho_(j,k)) ] #<eq:agf-experiment> \
  &= 1/2 + 1/12 sum^3_(j,m=1) sum^1_(k=0) alpha_(j,k) beta_(m,j) expval(hat(O))_(j, k, m) ] #<eq:direct-agf-2>
$
where indices $j, m$ indicate different Pauli observables, i.e., $hat(P)_j, hat(P)_m in {hat(X), hat(Y), hat(Z)}$, and $rho_(j,k) in { ket(+)bra(+), ket(-)bra(-), ket(i)bra(i), ket(-i)bra(-i), ket(0)bra(0), ket(1)bra(1) }$.
In this notation, $j$ indicates the Pauli observable and $k$ indicates whether the state is a +1 or a -1 eigenstate of that observable.
For example, $rho_(2,1) = ket(-i)bra(-i)$.
For any Pauli observable $hat(P)_j$, we have $alpha_(j,0) = 1$ and $alpha_(j,1) = -1$.
From @eq:direct-agf, we substitute $hat(U) P^dagger_j hat(U)^dagger$ with its Pauli basis expansion, resulting in @eq:agf-experiment where $beta_(m, j) = 1/2 tr[ hat(U) hat(P)^dagger_j hat(U)^dagger hat(P)_m ]$. This expansion allows us to estimate the AGF between any ideal (qubit) unitary map and its experimental realization by measuring the expectation values of at most 18 combinations of 6 initial states and 3 Pauli observables.
Notice that $tr[ hat(P)_m cal(E)(rho_(j,k)) ]$ is the exact experimental expectation value, so we denote it as $expval(hat(O))_(j, k, m)$ in @eq:direct-agf-2.

Since the predictive model $caron(f)(control)$ can predict the expectation values of a given control parameter, one can calculate the AGF from the prediction. The quantity of interest is the accuracy of AGF prediction, $macron(F)_"model"$, to the exact value of the experimental realization, $macron(F)_"exact"$. Here, we note that $macron(F)_"exact"$ is not a finite-shot estimation of AGF measurable in experimental realization, but is the infinite-shot (exact) value.
The difference between them is obtained using @eq:agf-experiment,
$
  macron(F)_"model" - macron(F)_"exact" = 1/12 sum^3_(j,m=1) sum^1_(k=0) alpha_(j,k) beta_(m, j) ( expval(hat(O))_(j, k, m)^"model" - expval(hat(O))_(j, k, m)^"exact" )
$ <eq:agf-diff>

We now show that the AGF difference in @eq:agf-diff and the MSE in @eq:msee are related through the Cauchy-Schwarz inequality,
$
  ( sum_i u_i v_i )^2 <= ( sum_i u_i^2 ) ( sum_i v_i^2 )
$ <eq:cauchy-schwarz>

Let us start by defining functions
$
  u_i = expval(hat(O))_i^"model" - expval(hat(O))_i^"exact" equiv expval(hat(O))_(j, k, m)^"model" - expval(hat(O))_(j, k, m)^"exact"
$<eq:ui>

$ v_i & equiv alpha_(j,k) beta_(m, j) $ <eq:vi>

where the indices $j,k,m$ are combined into single index $i$,
which leads to writing @eq:agf-diff as
$
  macron(F)_"model" - macron(F)_"exact" = 1/12 sum_i v_i u_i
$ <eq:lhs-bound>

Note that we use the Cauchy-Schwarz inequality for real numbers, so we have to show that $u_i$ and $v_i$ are real numbers. Since $u_i$ is defined in terms of the expectation value of an observable, it must be a real number. For $v_i$, we already know that $alpha_(j, k) in {-1, 1}$, so we have to show that $beta_(m, j)$ is a real number. We consider $beta_(m,j)^*$ as follows,
// #align(
//   $ (1/2 tr[ hat(U) hat(P)^dagger_j hat(U)^dagger hat(P)_m ])^* &= 1/2 tr[ (hat(U) hat(P)^dagger_j hat(U)^dagger hat(P)_m)^dagger ], $ ,
//   $ &= 1/2 tr[ hat(U) hat(P)^dagger_j hat(U)^dagger hat(P)_m ], $ <eq:beta-3>,
//   $ beta_(m,j)^* &= beta_(m,j) $ <eq:beta-4>
// )

$
  (1/2 tr[ hat(U) hat(P)^dagger_j hat(U)^dagger hat(P)_m ])^* &= 1/2 tr[ (hat(U) hat(P)^dagger_j hat(U)^dagger hat(P)_m)^dagger ], #<eq:beta-3> \
  &= 1/2 tr[ hat(U) hat(P)^dagger_j hat(U)^dagger hat(P)_m ] \
  beta_(m,j)^* &= beta_(m,j) #<eq:beta-4>
$

where, we use the cyclic property of trace, and use the fact that $hat(P)_m = hat(P)_m^dagger$ and $hat(P)_j = hat(P)_j^dagger$ in @eq:beta-3. From @eq:beta-4, we show that $beta_(m, j)$ must be a real number. Therefore, $v_i$ must be a real number.
Now, we will express @eq:msee in terms of $u_i$. We denote the experimental data $bb(E)_n [hat(O)_k]^"exp"_(rho_0, control)$ with $bb(E)_i^"exp"$ for notation brevity,
$
  cal(L)_"SE[E]" = 1/18 sum_i ( expval(hat(O))_i^"model" - bb(E)_i^"exp" )^2
$ <eq:msee-loss-i>

We then express @eq:msee-loss-i in terms of $u_i$ as follows,
$
  cal(L)_"SE[E]" &= 1/18 sum_i ( expval(hat(O))_i^"model" - expval(hat(O))_i^"exact" + expval(hat(O))_i^"exact" - bb(E)_i^"exp" )^2 #<eq:se-expand-1> \
  &= 1/18 sum_i (u_i - epsilon_i)^2 #<eq:se-expand-2> \
  &= 1/18 sum_i u_i^2 - 1/9 sum_i u_i epsilon_i + 1/18 sum_i epsilon_i^2 #<eq:se-expand-3>
$ <eq:sum-ui>

In @eq:se-expand-1, we inserted zero in the forms of the exact experimental expectation value $expval(hat(O))_i^"exact"$. Then, in @eq:se-expand-2, we defined $epsilon_i = bb(E)_i^"exp" - expval(hat(O))_i^"exact"$ and the remaining terms are $u_i$ as we defined in @eq:ui. Lastly, we expanded the expression in @eq:se-expand-3. We then, rearrange for $sum_i u_i^2$ as follows,
$
  sum_i u_i^2 = 18 cal(L)_"SE[E]" + sum_i (2 u_i epsilon_i - epsilon_i^2)
$

For the summation $sum_i v_i^2$, we can look at the properties of $alpha_(j,k)$ and $beta_(m, j)$. Since we know that $alpha_(j, k)$ has two possible values, $alpha_(j,0) = 1$ and $alpha_(j,1) = -1$, and that $beta_(m, j)$ does not depend on the index $k$, we can write
$
  sum_i v_i^2 = sum_(j, k, m) (alpha_(j,k) beta_(m, j))^2 = 2 sum_(j, m) (beta_(m, j))^2
$

This quantity can be simplified even further. Note that $beta_(m, j) = 1/2 tr[ hat(U) hat(P)^dagger_j hat(U)^dagger hat(P)_m ]$, where $hat(P)_j$ and $hat(P)_m$ are Pauli matrices.
We express $hat(P)_j^dagger = hat(P)_j = rho_(j,+) - rho_(j, -)$ in terms of its $+1$ and $-1$ eigenvectors.
When the initial states $rho_+$ and $rho_-$ evolve under the same unitary, their expectation values when measuring the same Pauli observable will have the same magnitude but opposite sign, $tr[ hat(U) rho_(j, +) hat(U)^dagger hat(P)_m ] = - tr[ hat(U) rho_(j, -) hat(U)^dagger hat(P)_m ]$
We can therefore write
$
  beta_(m, j) = tr[ hat(U) rho_(j, +) hat(U)^dagger hat(P)_m ] = expval(hat(P)_m)_(rho_(j, +))
$ <eq:bmj-1>

Note that the quantity we want is the summation $sum_(j, m) (beta_(m, j))^2$, which, given @eq:bmj-1, we see that the summation with the index $m$ leads to the exact form of purity of a state $hat(U) rho_(j, +) hat(U)^dagger$.
Given that we know $rho_(j, +)$ is a (pure) eigenstate, we are sure that the purity is unity. That is, the summation of index $m$ will be $1$. Since index $j$ is the sum over initial states, this means that the summation with the index $j$ gives a multiplicative factor of $3$, leading to
$
  sum_i v_i^2 = 2 sum_(j, m) (beta_(m, j))^2 = 2 sum_(j, m) expval(hat(P)_m)_(rho_(j, +))^2 = 6
$ <eq:sum-vi>

Substituting the results of $u_i$ and $v_i$ summation into @eq:cauchy-schwarz, we arrive at the following inequality,
$
  (macron(F)_"model" - macron(F)_"exact")^2 <= 6/144 (18 cal(L)_"SE[E]" + sum_i (2 u_i epsilon_i - epsilon_i^2))
$ <eq:performance-bound>

To remove the random variable $epsilon_i$, we compute the expected value of the inequality. Notice that, $u_i$ and $v_i$ do not depend on $epsilon$, so $macron(F)_"model" - macron(F)_"exact"$ is constant with respect to $bb(E)_(y|x)$. We use the fact that $bb(E)_epsilon [epsilon_i] = 0$, and notice that $bb(E)_(y|x)[epsilon_i^2] = "Var"(bb(E)_n [hat(O)]^"exp"_(rho_0,control) | control)$ is the conditional variance with the result from @eq:var-data to obtain the following bound,
$
  (macron(F)_"model" - macron(F)_"exact")^2 <= 3/4 (bb(E)_(y|x)[cal(L)_"SE[E]"] - (2)/(3n))
$ <eq:before>

Furthermore, we can connect @eq:bias-var-decomp-alt to the absolute difference between the AGF predicted by model and exact experimental value by considering an expected value of inequality in @eq:before with respect to $x$ and using the Jensen inequality for a random variable $abs(macron(F)_"model" - macron(F)_"exact")$ with respect to $x$. We then have
$
  bb(E)_x [ abs(macron(F)_"model" - macron(F)_"exact") ] <= sqrt(3/4 (bb(E)_(x, y)[cal(L)_"SE[E]"] - 2/(3n)))
$ <eq:agf-bound-alt>
where the inequality becomes the bound of the expected difference of AGF over different inputs in terms of the expected loss.

With decomposition of $cal(L)_"SE[E]"$, the upper bound of the AGF difference above depends only on the $bb(E)_x [ (caron(f)(x) - bb(E)[y|x] )^2 ]$ since the dependence on $n$ is cancelled out. Note that we can estimate $bb(E)_x [ (caron(f)(x) - bb(E)[y|x] )^2 ]$ using the testing dataset discussed in previous section.
This result suggests that a high-precision quantum gate can be reliably produced by using a predictive model trained on a few-shot dataset.
In the best case scenario, with optimal predictive model $bb(E)_(x,y)[cal(L)_"SE[E]"] = 2/(3n)$, the upper bound on the RHS becomes zero, so the model can predict the exact experimental AGF.
We would like to highlight that the LHS of @eq:agf-bound-alt is the difference between the model prediction and the exact experimental value, and is not the difference between the model prediction and the finite-shot experimental value.
These analytically derived bounds in @eq:var-data, @eq:before, and @eq:agf-bound-alt confirm the observed behaviour in @youssryMultiaxisControlQubit2023.

Given the expected #mseeloss, we visualize the scaling of the performance bound in @fig:bound_mse_plot. We can choose to stop the model training afford when the criteria is reach. 

#figure(
  image("../figures/fig_bound_mse_plot.svg"),
  caption: [
    Plot of the performance bound in terms of optimality. The x-coordinate of annotate tuples is the expected #mseeloss, while the y-coordinate is the corresponding performance bound.
  ],
) <fig:bound_mse_plot>


== The Way of Probabilistic Machine Learning <sec:pml-way>

In this section, we will explore the way of probabilistic machine learning. The most appeal feature is that, it provides a natural to quantify uncertainty of prediction. With uncertainty quantification associate to the prediction, i.e. predicting distribution of the output instead of point estimation, a derived quantity would also be a distribution, i.e. has the associate uncertainty. Moreover, uncertainty quantification can also be used to make an informed decision. That is, we will now know how certain the model is of its prediction. Using statistical learning method, the model cannot say "I'm not sure", or "I don't know", but with probabilistic learning method, the model can. This is possible as probabilistic learning method constructs a predictive model using Bayesian Inference. Thus, instead of training to find model parameter, it will perform #quote[*inference*] for model parameter instead. In the higher level, probabilistic learning method is to find a probabilistic model that produce a distribution as close as a distribution that generate observed data. The ability say how certain it is to its prediction, allows us to perform the experiment where we are uncertain the most.

We will take the next step by define a predictive model in the probabilistic way, and point out the differences from the statistical predictive model. Then, we will review how does model construction in PML work. As an example, we discuss how we transform statistic Graybox model to the probabilistic Graybox model using Bayesian Neural Network (BNN). Finally, we will discuss what we can learn from using PML to characterize quantum device.

=== Predictive model, but probabilistic

Since, the raw measurement result of quantum device is stochastic by nature. Statistical predictive model aims to predict expectation value of observable, given some initial state and control parameters. The expectation value is deterministic in ideal case, i.e. without noise and infinite number of measurement result. However, with finite-shots constraint in the realistic setting, the expectation value is stochastic, and can be view as a random variable. Consequently, it become a limiting factor of the expected minimum MSE loss that the statistical model can achieve. On the other hand, probabilistic predictive model, or simply probabilistic model predicts distribution of the output data instead. Thus, it is naturally fit well with the nature of the experiment data.

To model the probabilistic model is to model the physical process that generate our observed data. This requires prior knowledge (domain specific knowledge), so that we can model our transformation as close as possible to the actual process. For instance, part of the process might be a probabilistic event that depends on other probabilistic events. Meaning that, we have to assign a particular type of distribution to the random variable, and set priors on the distribution parameters. For example, if we are trying to model a weather in Thailand during the day, we may model the observed temperature as a Gaussian distribution based on our experience, and we might want to set the initial parameter (prior) at $30$ Celsius instead of $5$ Celsius.

I will first start the modeling with the output of the observed data first as it is what we are fairly confidence in our prior on measurement process. There are two possible ways to model the output of the probabilistic model for quantum device characterization. (1) Predicting expectation value of observable by model it as a Gaussian distribution

$
  y tilde cal(N)(idealexpval, sqrt(var(hat(O))_n))
$ <eq:normal-based-obs>
where $n$ is the number of shots used to estimate the expectation value. (2) Predicting the raw measurement result as a Bernoulli Distribution,
$
  y tilde "Bern"((1 + idealexpval) / 2)
$ <eq:bern-based-obs>
The former is a direct analogy to the statistical version. While, the latter is one step closer to the actual physical process. Note that the expectation value as a prior does not restrict a posterior to be the ideal value, but it served as our belief of a whereabout of the true value, in case of the presence of the noise.

=== How to infer your #strike[dragon] PM <sec:how-to-pml>

We now discuss about the conventions using in the PML paradigm. The output label is typically referred to observable or *observed data*. The trainable parameter (model parameter) that was trained for in the SML paradigm, is referred to as *latent variable*.

Predictive models defined in @eq:normal-based-obs and @eq:bern-based-obs require different observed data. I will refer to them as Gaussian-based model and Bernoulli-based model respectively. Gaussian-based model required observed data to be the finite-shots expectation value of observable. While, Bernoulli-based model required observed data to be the binary measurement result from device. Both of observed data types can be converted into each other. In fact, finite-shots expectation value is a compact representation of binary results. Since we do not need an order of binary data, bit $b$, that used to estimate finite shots expectation value, we can recover the number of each bit $n_b \in \{ 0, 1 \}$ by multiplying number of shots $n$ to the expectation value $estexpval$ and using the constraint $n = n_0 + n_1$. Formally consider
$
  estexpval = 1 / n (n_0 + n_1),
$
substituting $n_1 = n - n_0$, we get
$
  n_0 = n / 2 (estexpval + 1).
$
For the ease of the calculation, we can store $n_0$ and $n_1$ instead of $estexpval$.


Regardless of the model choice, we will now discuss how can we using Bayesian Inference to infer for the latent variables and make a prediction following the review in @arbelPrimerBayesianNeural2023 @p.murphyProbabilisticMachineLearning2023. Suppose that our dataset is generated from a likelihood $p(cal(D), cal(w))$ where $cal(w)$ are some parameters. From Bayes' rule, the *posterior distribution* is
$
  p(cal(w), cal(D)) = p(cal(w))p(cal(D)|cal(w)) / p(cal(D)),
$

where $p(cal(w))$ is a prior distribution of $cal(w)$ and $cal(D)$ called _evidence_ or _marginal likelihood_. During the predicting phase of a new observation $bold(y^*)$ given an input $bold(x^*)$ , the (posterior) predictive model is then
$
  p(bold(y^*)| bold(x)^*, cal(D)) = integral p(bold(y^*)| bold(x^*), cal(w))p(cal(w)|cal(D)) d cal(w).
$ <eq:posterior-predictive-model>
In words, we start by inferring for the posterior distribution of model parameters first, then we use it to make the *posterior predictive distribution*, i.e. producing the distribution of the prediction.

The problem is now lie on the calculation of the posterior. In PML literatures, the evidence term is the term that is mostly impossible to obtain a closed-form in general, i.e. *intractable*. There are several methods that are developed to approximate the posterior. In this thesis, we consider a *Variational Inference (VI)* method for its computational performance.

In VI, we introduce a tractable variational posterior distribution $q_(theta)(cal(w))$ to approximate the true posterior. That is, we have to perform optimization for variational parameters $theta$ to find $p(cal(w| cal(D)) approx q_theta(cal(w))$. Objectively, we have to minimize KL divergence between $p(cal(w)|cal(D))$ and $q(cal(w| theta)$ which is
$
  "KL"(q_theta|| p(cal(w)|cal(D))) = integral q_theta (cal(w)) log q_theta (cal(w)) / p(cal(w)|cal(D))d cal(w),
$
or in the tractable form,
$
  "KL"(q_theta|| p(cal(w)|cal(D))) = - underbrace(integral q_theta (cal(w)) log p(cal(w))p(cal(D)|cal(w)) / q_theta(cal(w)) d cal(w), "ELBO") + log p(cal(D)).
$
Since the log of evidence does not depends on variational parameters, maximizing the Evidence Lower Bound (ELBO) is equivalent to minimizing KL divergence. Therefore, in practice, we optimize for $theta$ by minimizing the negative ELBO,
$
  "ELBO"(theta) = - "KL"(q_theta||p) + log p(cal(D)).
$
In practice, the it is harder to perform inference with Gaussian-based model. Because, the expectation value of observable is constrained within the range of $[-1, 1]$. At the edges, i.e. $idealexpval in { -1, 1 }$, the variances become $0$. Whereas our current implementation using `numpyro` cannot handle it properly.

=== Probabilistic Graybox model with Bayesian Neural Network

Transition of statistical version of Graybox model in @sec:graybox-characterization is straightforward to implement. The stochastic element within the Graybox model is the Blackbox part where we can view the model parameters as random variables. Otherwise, Whitebox is a deterministic transformation. Thus, we need to implement a probabilistic Blackbox model (PBM) that predict a distribution instead of point prediction.

Depends on the statistical learning method of choice, the PBM implementation may has to drastically change. However, in the case of Deep Neural Network as statistic Blackbox model (SBM), we can directly convert it to @bnn. The key difference of @bnn to statistical NN is that, the trainable parameters is a distribution instead of point value. We have a freedom to choose the type of distributions (can be mixed), which can directly reflect our prior knowledge of the parameters. Normal distributions with zero mean and variance of unity are generic choice $cal(N)(0, II)$. BNN perform prediction by sampling from the posterior distribution of model parameters, then the output of the model is calculated from the parameter samples by usual matrix multiplication. The ensemble of output from multiple sampling is then the distribution of model output.

Promoting @dnn to @bnn seems like an upgrade in a first grace, however, in pratice, @bnn might not necessary always perform better than @dnn. From @jospinHandsOnBayesianNeural2022, they states that @bnn can learn small sample size without overfit. If true, it would be highly benefical to the quantum device characterization, since performing experiment to collect data points is expensive.

#task[
  #todocontinue[
    Insert section 2B from paper 2 when the paper is published?.
  ]
]



== Routes to Data-Efficient <sec:route-to-data-efficient>

#tldr[
 The data-efficient approach generally involves quantifying the current (prior) knowledge of the system, estimating some utility value, such as expected information gain, based on the prior knowledge, and collecting data based on the estimated value. The steps are repeatedly performed until some conditions are met.
]

From the discussion in @sec:data-efficient-notion, there are a number of ways to reduce the consumption of data in the model characterization (not necessarily specific to a quantum system). Here, we are going to discuss them in the context of quantum device characterization and a possible uses of the @eq:var-data to reduce the data consumption.

=== One-step Optimal

As discussed in @sec:data-efficient-notion, we can calculate @eig using estimators proposed in @fosterVariationalBayesianOptimal2019 and simply select the control parameters that maximize the @eig as the next design for the experiment. This procedure is referred to as one-step optimal in the information-theoretic perspective.

In @fosterVariationalBayesianOptimal2019, they demonstrated the example use case of @eig with the memory capacity experiment. The concept is, however, general. In our context, we can also apply the use of @eig as well. Thus, to estimate the @eig, we need a probabilistic model of the quantum system. Fortunately, we have constructed and discussed such a probabilistic model in @sec:pml-way. There are wide classes of probabilistic models that can model the quantum system. For instance, from the parametrized Hamiltonian, we can identify some or all of the system parameters (qubit frequency, detuning, etc.) as random variables. We then apply the @boed to identify those system parameters. Alternatively, we can also consider the probabilistic Graybox model where we promote the @dnn to @bnn. The model parameters of the @bnn, which are the weights of the model, are the random variables. So, we apply the @boed to identify those weights. Either way, we expect that with the @boed approach, we can identify the target parameters with fewer experiments than the traditional approaches.

=== Sequential-BOED

We have discussed from the results presented in previous works that one-step @boed does not necessarily yield the highest total @eig in the sequential experiments setting. Successive works proposed the sequential-BOED approaches, which aim to maximize total @eig rather than a single-step @eig. However, intensive precomputation is required.

To handle the computationally intensive requirement of the schemes developed for sequential-BOED @fosterDeepAdaptiveDesign2021 @ivanovaImplicitDeepAdaptive2021, we can strategize the selection of experimental designs based on our priors on the problem. For demonstration purposes, we will consider one of the possible strategies in @sec:boed-experiment, namely, a *subspace strategy* that divides the control parameters space into multiple subspaces, then selects the designs that maximize the @eig within the subspace. The strategy is motivated by the fact that we want to perform inference for @bnn (the Blackbox part of the Graybox model). Hence, we want to retain some balance (to not fully bias the selection to @eig) in our dataset.

It is also possible to use `inspeqtor` to perform sequential-BOED as proposed in @fosterDeepAdaptiveDesign2021 @ivanovaImplicitDeepAdaptive2021, however, we do not explore them numerically in this thesis since they require too much computational resources.
// #margin-note()[The prototype code take too much resource to run]

=== Data-variance approach <sec:data-variance-approach>

From @eq:var-data, we can estimate the optimality of the predictive model. In the case of a statistical predictive model, we can use its prediction directly. While in the case of a probabilistic model, we have to use the mean of the prediction as a prediction representation. Either way, we can perform experiments to construct the dataset in mini-batches. For each iteration, we use all of the dataset collected so far to characterize the model. Then, with the testing dataset, we can estimate the optimality of the model, in which we can stop the data collection when some thresholds are reached.

Alternatively, from @eq:var-data, we can estimate the quantity by sampling the control parameters $x_1$ first, then performing experiments multiple times with the control parameters to collect the sample $y$ from the conditional distribution $p(y|x_1)$. We then repeated with $x_2, ..., x_m$ to approximate the joint $p(x, y)$. With this method, we form the specialized dataset such that we can also benchmark the model performance using probability distribution measures, such as @jsd, in addition to the optimality of the model. In the @sec:boed-experiment, we will discuss the use of the data-variance approach in the device characterization experiment.

=== Bayesian Optimization approach

While a Bayesian optimization is initially designed for a Gaussian process, it might be possible to adapt the algorithm and mathematical model to a probabilistic model, such as the Graybox model. For example, an expected improvement, one of the acquisition functions, needs the mean and variance of the observations, and the observed best observation to calculate the metric @kamperisAcquisitionFunctionsBayesian2021. So we can use the probabilistic Graybox as a surrogate model. Then the function to be optimized is a function of the distance between distributions. This approach requires a specialized dataset to implement. 

// #let caution-rect = rect.with(inset: 1em, radius: 0.5em)
// #inline-note(rect: caution-rect, fill: orange.lighten(80%))[
//   Write about reviews in this paper @rainforthModernBayesianExperimental2024. Paritcularly about (1) BED, then (2) BAD, then talk about the need for two-steps optimization, which possible to overcome with result in @fosterUnifiedStochasticGradient2020, and then (3) iDAD for "policy-based" approach @ivanovaImplicitDeepAdaptive2021.
//   @sarraDeepBayesianExperimental2023 use the result of @fosterUnifiedStochasticGradient2020. @fidererNeuralNetworkHeuristicsAdaptive2021 use "policy-based" appraoch but not as introduced in @ivanovaImplicitDeepAdaptive2021.

//   Discuss these work @fidererNeuralNetworkHeuristicsAdaptive2021 @staceOptimizedBayesianSystem2024a @sarraDeepBayesianExperimental2023 for the related works.
// ]


