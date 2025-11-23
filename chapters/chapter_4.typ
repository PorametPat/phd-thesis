#import "@preview/drafting:0.2.2": inline-note, margin-note
#import "@preview/callisto:0.2.4"
#import "../utils.typ": (
  class, control, expval, func, meth, modu, pkg, const, spec, style, tldr, tldr2, todocontinue, todoneedwrite,
)
#import "@preview/gentle-clues:1.2.0": *
#import "@preview/fletcher:0.5.8": diagram, edge, node
#import "@preview/chronos:0.2.1"
// #import "@preview/zebraw:0.5.4": *

// #show: zebraw.with(
//   background-color: rgb("#1d2433"),
//   lang-color: rgb("#1d2433")
// )

// #codly(fill:rgb("#1d2433"),zebra-fill: none, stroke: 0pt + none)


// #show raw.where(block: false): it => box(
//   it,
//   fill: rgb("#d7d7d7"),
//   outset: 0.2em,
//   radius: 0.25em,
// )




= `inspeqtor` Design and Implementation <sec:inspeqtor>

In this section, the implementation details of the `inspeqtor` package will be discussed. The following is the design decision that drives the development of `inspeqtor`.
+ Functional style programming: The function should be pure, i.e., it should have no side effects (it does not mutate non-local variables), and the same input always produces the same output. This is to avoid unintended actions triggered by the class's method since the input might be hidden away when calling the method.
+ Platform-Agnostic: We aims to support a varities implementation of quantum device. This goal also motivate the choice of functional programming style which off-load the users the responsibility to master and understand classes and how to inherit them properly to just define functions that compatible and plug use them directly without worry.
+ Utility-based: We aim to provide utility functions for developers to use with their favorite packages.
+ First class support for the Predictive model: With automatic differentiation numerical framework, we can calculate the gradient of function without having to use specialized formulation. Thus, we design our framework revolve around the predictive model as defined in @eq:predictive-model.


`inspeqtor` aim to be a library providing an opinionated way to store characterization dataset, utility functions to characterize and benchmark the predictive model.

Quantum device maintainance routine is differ depending the harware specifications, for instance, the physical realization of the qubit. Thus, there are multiple possible characterization and calibration pipelines. To design a library that provide a resuable code for maintainance engineer, we consider the abstraction of the pipelines for the flexibility in practical adaptation. Let us consider the sequence diagrams for the quantum device characterization and control calibration.

Assume that, we would like to use open-loop approach to produce quantum gates. One of the simple maintainance pipeline is the sequence of characterization, calibration, and opeartion phases. We illustrate the pipeline using sequence diagram presented in @fig:pipeline-open-loop.
The choice of using open-loop optimization required the characterization to construct the model of the device.
For example, in the form of device Hamiltonian, then the model is used to calibrate for the control that maximize some objective function @khanejaOptimalControlCoupled2005 @machnesTunableFlexibleEfficient2018.
Or in the form of (may be differentiable) predictive model, which we can employ gradient-based optimizer to find the optimal control @youssryCharacterizationControlOpen2020 which is the main method we use in this thesis. Finally, the optimzied gates are deployed to the device and used in operation. Depends on the physical system, engineer has to routinely benchmark the performance of the quantum gate since the drift in system parameters might cause the operation fidelity to drop. If the qualites are drop to some threshold, characterization and calibration processes are start again.

#figure(
  chronos.diagram({
    import chronos: *

    // Define participants
    _par("User", display-name: "User")
    _par("Model", display-name: "Model")
    _par("Device", display-name: "Quantum Device")

    // Characterization phase
    _sep("Characterization")
    _loop("Characterization Loop", {
      _seq("User", "Device", comment: "Perform experiments")
      _seq("Device", "User", comment: "Data", dashed: true)
      _seq("User", "Model", comment: "Characterization")
      _alt("Selection strategy", {
        _seq("Model", "User", comment: "Select new experiments", dashed: true)
      })
    })

    // Calibration phase
    _sep("Calibration")
    _loop("Optimization", {
      _seq("User", "Model", comment: [Find the control that \ maximizes fidelity])
      _seq("Model", "User", comment: "Return optimized control", dashed: true)
    })
    _seq("User", "Device", comment: "Deploy calibrated control")

    // Operational phase
    _sep("Operational")
    _loop("Operating and monitoring", {
      _seq("User", "Device", comment: "Use device")
      _seq("User", "Device", comment: "Check the quality")
    })
  }),
  caption: [
    Quantum device maintainance pipline with characterization, calibration, and opertational sequence using open-loop optmization approach.
  ],
) <fig:pipeline-open-loop>

Moreover, during the characterization, data gathering experiments may perform in batches or in feedback loop. As discussed in @sec:route-to-data-efficient, it is possible to select experimental designs to perform rather than randomly selecting for a better data utilization. Note that, we do not assume a choice of predictive model.

Alternatively, device calibration using closed-loop approach does not required extensive device characterization.
The priors of the device that necessary for closed-loop approach are gather and prepared then the optimizer is deploy close to the device as illustrate in @fig:pipeline-closed-loop.

#figure(
  chronos.diagram({
    import chronos: *
    // Define participants
    _par("User", display-name: "User")
    _par("Optimizer", display-name: "Optimizer")
    _par("Device", display-name: "Quantum Device")

    // Calibration phase
    _sep("Calibration")
    _loop("Optimization", {
      _seq("Optimizer", "Device", comment: [Suggest new parameters])
      _seq("Device", "Optimizer", comment: [Data])
    })
    _seq("Optimizer", "User", comment: "Return optimized control")
    _seq("User", "Device", comment: "Deploy calibrated control")
    // Operational phase
    _sep("Operational")
    _loop("Operating and monitoring", {
      _seq("User", "Device", comment: "Use device")
      _seq("User", "Device", comment: "Check the quality")
    })
  }),
  caption: [
    Quantum device maintainance pipline with characterization, calibration, and opertational sequence using closed-loop optmization approach
  ],
)<fig:pipeline-closed-loop>


With the guidelines from the abstract pipeline, we can now discuss the implementation of the `inspeqtor`. First, we discuss the choice of the numerical backend `jax` that we used and then discuss the essential modules then finish with other modules brief disucssion. Note that, we did not implement blackbox optimizer. Because, the advantange of using `jax` is the benefit of automatic-differentiation which allow for gradient-based optimization. However, blackbox optimization algorithm does not required gradient of the objective function. Thus, user can utilize other libraries for the blackbox optimizers.

== Structure of #pkg[`inspeqtor`]

By design, #pkg[`inspeqtor`] library is a collection of modules that works together via data structure representing physical entities. To maintain a platform-agnostic design, we model the physical entities following the mathematical representation rather than object with attributes and methods. Thus, in the following sections, we will sometime discuss the design of the function by introducing the function signature base on its mathmematical represenation. While, its actual implementation may includes additional arguments which can be static. For instance, let us consider a time-dependent Hamiltonian of a system controllable by a vector of $n$ control parameter $control$ denoted $H(control, t)$. In our framework, we represent $H(control, t)$ with function $cal(H)$ with the following signature,
$
  cal(H) ( control: bb(R)^n times t: bb(R) ) -> "Hermitian"(bb(N), bb(C)).
$
The function return a Hermitian matrix with appropiate dimension. In the implementaiton, user may construct the function $cal(H)$ using #modu(`control`).
The Hamiltonian function $cal(H)$ is then used to produce a solver function $cal(S)$ that solve the system differential equation given an initial state $rho_0 in "Hermitian"(bb(N), bb(C))$
with the following signature,
$
  cal(S) ( control: bb(R)^n times rho_0: "Hermitian"(bb(N) times bb(C)) times cal(H) times arrow(t): RR^(|t|) ) -> rho_f: "Hermitian"(bb(N), bb(C)).
$
The function $cal(S)$ is given in #modu(`physics`). Then, user can use functions in #modu(`utils`) to compose a function that return a finite-shot expectation value $EE[ hat(O) ]$ with the following signature,
$
  EE (control: bb(R)^n) -> EE[ hat(O) ]: RR.
$

We organize #pkg[`inspeqtor`] into multiple modules that provide functionalities for composing a pipeline for characterizing and calibrating the quantum device. To main it intuitive for user, each module aims to provide functions related to its name. The  platform-agnostic functions are exported under the `module` namespace, while the predefined functions for platform specific or common use case are export under the `module.library` namespace. In the case of #modu[`models`], we define extra namespaces to organize the functions under their related `submodule`. Below, we visualize the tree structure of #pkg[`inspeqtor`].

#import "@preview/treet:1.0.0": *


#[
  #set par(leading: 0.45em, justify: true, first-line-indent: (
    amount: 0.0em,
  ))
  #tree-list[
    - inspeqtor/
      - control
        - library.py
      - data
        - library.py
      - models
        - library.py
          - nnx.py
          - linen.py
        - adapter.py
        - shared.py
        - probabilistic.py
      - physics
        - library.py
      - optimize.py
      - utils.py
      - boed.py
  ]
]




// #chronos.diagram({
//   import chronos: *

//   // Define participants
//   _par("User", display-name: "User")
//   _par("Strategy", display-name: "Strategy")
//   _par("Model", display-name: "Model")
//   _par("Device", display-name: "Device")

//   _seq("User", "Strategy", comment: "Prepare experiment")

//   _loop("Characterization Loop", {
//     _seq("Strategy", "Strategy", comment: [Select next \ experiment parameters])

//     _seq("Strategy", "User", comment: "Recommend experiment", dashed: true)

//     _seq("User", "Device", comment: "Perform experiment", color: purple)
//     _seq("Device", "User", comment: "Measurement data", dashed: true, color: purple)

//     _seq("User", "Model", comment: [Update/Characterize Model])

//     _seq("Model", "Strategy", comment: [Provide Posterior Model \ (if adaptive)], dashed: true, color: blue)

//     _seq("Strategy", "Strategy", comment: "Check termination", color: red)
//   })

//   _seq("Strategy", "User", comment: "Return Final Characterized Model", dashed: true)
// })





== Why `jax`

#tldr[
  We chose `jax` for the:
  + Automatic differentiation framework that hard to make a mistake.
  + Reproducibility by providing `key` for each stochastic function calling instead of global seed.
  + Performance with Just-In-Time compilation.
]

`jax` is a library for numerical computation similar to `numpy` but also provides automatic differentiation and Just-In-Time (JIT) compilation for high performance machine learning research @JAXComposableTransformations2018. The numerical framework provided by `jax` allows the community to build a machine learning library around `jax` easily without having to use specialized `array` and Pseudo Random Number Generator (PRNG) instances. `jax` makes the use of a functional programming style easy and natural, because `jax` handles automatic differentiation and PRNG explicitly. We will discuss them in the following.

In research, reproducibility is a must. We used `jax` mainly for its PRNG. Conventionally, a `python` user who uses NumPy would be familiar with setting the seed globally for reproducibility. In modern `numpy`, a pseudo-random number `Generator` instance must be created and passed around in the code where a random number is needed. Now, to perform machine learning, we have to choose a suitable library; the primary choices are `pytorch` or `tensorflow` for their rich and mature community. They have their internal handling of PRNG. Consequently, ensuring the reproducibility becomes complex.

To generate a random number using `jax`, we must provide a `key`, which is how `jax` ensures reproducibility. The `key` can be created with the following code,


```python
key = jax.random.key(0)
```

and the random number can be sampled by,

```python
sample = jax.random.uniform(key, shape=(10, ))
```

If the same key is used for the same function, the sampled random number will be the same. We can also derive a new `key` based on the old key by

```python
key, subkey = jax.random.split(key)
```

In fact, the preferred way to use PRNG capability of `jax` is to split the `key` first and use the `subkey` to generate a random number. For example, consider the following code snippet for recommended usage of PRNG:

```python
# Initialize master key
key = jax.random.key(0)
# Split the key into subkey
key, subkey = jax.random.split(key)
# Use the subkey to sample the random number
sample = jax.random.uniform(subkey, shape=(10, ))
```

This design decision allows users to define a stochastic function as a pure function.

Another alluring feature of `jax` is its automatic differentiation API. `jax.grad` and `jax.value_and_grad` are functions that can be used to evaluate the gradient of a function. They have the following call signature,

```python
f = lambda x: x**2
f_prime = jax.grad(f)
grad_value = f_prime(jnp.array([0.]))
# Or with the value
value_and_grad = jax.value_and_grad(f)(jnp.array([0.]))
```

Note that these functions return a function. In our opinion, this design of the function-based API of JAX is opposed to the method-based approach of PyTorch and the need for a context manager in TensorFlow, which makes it easier for users to make mistakes. Here is how to calculate the gradient using TensorFlow, Pytorch, and JAX.

#callisto.render(nb: json("../code/review_0001_why_jax.ipynb"))

Furthermore, `jax` provides automatic vectorization, allowing the user to define a function that only needs to calculate one batch. Then, the user can use `jax.vmap` to perform batch calculation. For example, the following snippet is possible,

```python
def matrix_multiplication(x, y):
	# Assume that `x` and `y` is n by n matrix
	assert len(x.shape) == 2 and y.shape = x.shape
	return x @ y

batch_matrix_multiplication = jax.vmap(matrix_multiplication)

x_batch, y_batch = jnp.ones((10, 2, 2)), jnp.ones((10, 2, 2))
z_batch = batch_matrix_multiplication(x_batch, y_batch)
```

This design allows users to easily create functions without worrying about batch dimension.

The benefits of the functional style of `jax` will become apparent as we continue our discussion on the API design of `inspeqtor`.


== Control

#tldr[
  The assumption of the control module is that, the control is a function of at least time, other parameters are refered to as control parameters. Moreover, to support the different arbitrary shape of control composing together, the user can define their atomic #class[`Control`] classes, and compose them together via a single #class[`ControlSequence`] object.
]

#spec[
  + Represent the control function of the quantum device.
  + Be able to convert the vector of the control parameter to the vector control simulable by #modu[`physics`]'s solver.
  + User can sample the control parameter from the control instance.
  + Provide a conversion between structured data type and array type for readable serialization and ready to used by machine learning framework.
]

The control of the quantum device depends on the system's physical realization. To model the possibility of control, we implement a base class, `BaseControl`, intended for developer to inherit from it with their implementation. Then, the defined control, denoted #class[`Control`], can be used via a unified interface, implemented via #class[`ControlSequence`]. Within the sequence, the developer can mix different types of #class[`Control`].

As an illustrative example, in the superconducting qubit platform, a classical control field has a finite resolution of $Delta t$ , we consider a Gaussian envelope control with a max amplitude of $A_m$, drive strength of $Omega$, a standard deviation of $sigma = sqrt(2 pi) / (A_m dot.op 2 pi dot.op Omega dot.op "dt")$, total area of $theta$ and the amplitude of $A = theta / (2 pi dot.op Omega dot.op "dt")$. The envelope is defined as
$
  h(theta, t) = A / (sqrt(2 pi) sigma) exp(-((t - T \/2)^2) / (2 sigma^2)) .
$ <eq:gaussian-envelope>
In this example, the only control parameters is $theta$. Control parameters are structured with `ParametersDictType`, simply a Python dictionary with a key as the control parameter's name corresponding to the parameter's value, which is a value with a type of `float` or `jnp.ndarray` of shape `shape = ()`. To inherit the `BaseControl`, one has to implement the `get_bounds` and `get_envelope` methods. The `get_bounds` method is a function with no input arguments that return a tuple of lower (of `ParametersDictType` type) and upper bound ( of `ParametersDictType` type) of the control parameters. This is the bound that will be call by a unified `sample_params` function to sample from uniform distribution defined with control bound and return sample of control parameter with the same structure as the lower and upper bound. Then the `get_envelope` function receives the control sample and return the envelope function which is a function of time. The mathematical specification of the `get_envelope` function is
$
  "get_envelope" (control: RR^n) -> (t: RR -> h(control, t): CC)
$

From the Gaussian envelop example, a control class may defined as follow,

// Get more functions preconfigured for this notebook
#let (display, result, source, output, outputs, Cell, cell) = callisto.config(
  nb: json("../code/chapter-4/note_0001.ipynb"),
)


#source("gaussian_control")

The #class[`ControlSequence`] can be instantiated as,
#figure(
  caption: [
    The syntax of how to define #class[`ControlSequence`] with #class[`GaussianPulse`] envelope.
  ],
  // kind: "code",
  // supplement: "Listing"
)[
  #source("ctrl_seq")
] <code:gaussian-seqence>

Note that the control sequence receives a dict of control classes. Internally, the total envelope is the sum of each control class, and the bound is a dict of the bounds of each control class. Now, the sample of the control sequence would look like this,

#Cell("single_sample")

We can visualize the envelope of the control sequence as follows,

#Cell("envelope")

Furthermore, we can also perform a batch sampling using `jax.vmap` as follows,

#Cell("batch_sample")

Then, the sample can be use for experiment, and model selection. The data structure of control parameters as a dict of dict is suitable for storing on disk and for readability. We will discuss how to store the control pararmeters later in @sec:data.
However, this structure might not be optimal for predictive model training. Thus, we also provide generic convenient functions #func[`ravel_unravel_fn`], which returns `ravel_fn` and `unravel_fn` to transform the `dict[ParametersDictType]` to `jnp.ndarray` and vice versa. Let us see the following example,

#Cell("ravel_unravel_fn")

We decide to implement control module revolving the class object because we have to standardize a way to structure the control parameters and consequently, how to properly save and load them on disk for later utilization. It is also serve as a way to perform automatic validation for the user at the instantiated time, which intented to help user detect potential bugs before proceed further. Below is the code snippet for save and load the control to and from file system.

#Cell("save_load_ctrl")

== Data <sec:data>
#tldr[
  This module provide inference to in-memory and file strorage system for the characterization data. We provide a way for user to enforce data schema to be consistent using type hinting, and store data enough for predictive model construction.
]

In `inspeqtor`, we prefer to keep the data structure flat and cross-platform. Thus, we avoid using `pickle` to store the data directly on the local machine and use JSON instead. The next design choice is to keep the data structure flat, i.e., avoid nested structure. So that when we read the data from disk using Python, we only have to deal with relatively shallow nested dicts. Moreover, to prevent inconsistent key naming, we prefer using `dataclass` or `namedtuple` when possible. We are also using Python's type hint feature. They allow developers to enjoy the benefit of auto-completion (#link("https://code.visualstudio.com/docs/editing/intellisense#_intellisense-features")[IntelliSense]), resulting in a better developer experience. In this section, we will discuss the details of the data entities implemented in Inspeqtor. First, we list the specifications for the #modu[`data`].

#spec[
  + Store data in a cross-platform and human-readable format
  + Be the interface between in-memory and file system
  + Enforce schema on the data and perform validation
  + Storing experimental data in three parts:
    + Experimental Configutation: the experimental settings such as the device and date perform the experiment
    + Parameters table: Table that each row is the control parameters $control$
    + Observed data table: The observations associated to the control paramters.
]


Sometimes, we have to store the specifications of a quantum device, especially when a predictive model using a closed-form Hamiltonian is needed. The fundamental unit of data of the quantum device is the qubit. Thus, we provide a `dataclass` for storing information about a qubit, which is `QubitInformation`. The instance can be initialized using the following code snippet:

#Cell("qubit_info")

Note that the package's initial design is based on the use case of a superconducting device. However, the data class can be easily generalized. Auto-completion makes it easy to access the properties of the qubit. Furthermore, the class also implements methods to convert from and to `dict` object `qubit_info.to_dict()`.

=== Experiment Config
Initial design of the #pkg[`inspeqtor`] revolve around Graybox characterization method. Consequently, we designed the data schema to support storing data for arbitrary control sequence #class[`ControlSequence`] as columns of control parameters and its corresponding expectation values. Turn out that it is a good design choice. Since characterization often involve sending some specialized control sequence and store the observations for post-processing. Especially, the characterization for control calibration, that a only few qubits system data have to be stored.

#Cell("config")

=== Experiment Data

The central data entity is #class[`ExperimentalData`]. This is an object that responsible for organizing the experimental data. It provides the save and load functionality. It stores experimental data as `polars.DataFrame` objects and separately stores `ExperimentConfig` to keep track of how the data is produced. It saves the data in a `csv` format to the disk and load them back to the ready to use format.

It requires `parameter_dataframe`, `observed_dataframe` and, `ExperimentConfig` for the instantiation. The `parameter_dataframe`, `observed_dataframe` are a `polars.DataFrame` objects where each row corresponded to a `parameter_id`. The following code snippet demonstrate the intented usage,

#Cell("exp_data")

Note that user has a freedom to include additional columns to the object.

Note that characterization that we are interested is only for small system. Not the randomized benchmarking kind. The initial developement of the package revolve around Graybox characterization method, but the concept can be generalized. For the characterization experiment of specific model parameter such as $T_1$ and $T_2$ time, experimental results can also be format and save using #class[`ExperimentalData`] object.

User can save and load the #class[`ExperimnetData`] from and to folder directly using #meth[`save_to_folder`] and #meth[`from_folder`] functions.

=== Operator, State, and Expectation Value
Throughout the characterization and calibration process, the user will often has to repeatedly access to observable $hat(O)$ and initial state $rho_0$ both in terms of its string and matrix representation. `ExpectationValue` is an object that standardize way to interact with the mathematical definition of expectation value of quantum observable $expval(hat(O))_rho_0$ in `inspeqtor`. User can access string representation via `initial_state` and `observable`.

#Cell("expval")

To get the matrix representation of the initial state and observable, we use the functional approach as follows:

#Cell("initial_state")
#Cell("observable")

Each token in the string of the initial state and observable represents a common eigenvector of Pauli operator and Pauli operator.

We expect the user to interact with the expectatation values via a predefined list of `ExpectationValue`, defined in `sq.utils.default_expectation_values_order`. The list is a default order of the expectation values that used throughout `inspeqtor`, since we always have to loop over the combinations of the expectation values. For a system of `n` qubits, we can generate the list of order using #func[`sq.data.get_complete_expectation_values(n)`].

=== Data flow

With the #modu[`control`] and #modu[`data`] ready, let us illustrate the flow of data using `inspeqtor` in more details in a simple case. Consider the characterization pipline where we perform the characterization experiment using random strategy, i.e., there is no exprimental design component in the process. In @fig:data-module, we illustrate how can user use functions and data entities provided by `inspeqtor` in the experiments from the start until the end of the process where data is store in the storage.


#figure(
  chronos.diagram({
    import chronos: *

    _par("User", display-name: "User")
    _par("Control", display-name: [#modu[`control`]], color: white)
    _par("Device", display-name: "Device")
    _par("Data", display-name: [#modu[`data`] ], color: white)
    _par("Storage", display-name: "Storage")

    // Setup phase
    _sep("Experiment Setup")
    _seq("User", "Control", comment: "Define atomic control action", color: blue)
    _seq("User", "Control", comment: "Create ControlSequence", color: blue)
    _seq("Control", "User", comment: "Validate & return sequence", dashed: true, color: blue)

    // Device configuration
    _sep("Device Configuration")
    _alt(
      "Real Hardware",
      {
        _seq("User", "Device", comment: "Setup the device", color: red)
      },
      "Simulation",
      {
        _seq("User", "Device", comment: "Setup Hamiltonian & Solver", color: green)
      },
    )

    // Experiment execution
    _sep("Experiment Execution")
    _loop("For each sample (e.g., 100x)", {
      _seq("User", "Control", comment: "Sample parameters")
      _seq("Control", "User", comment: "Return control params", dashed: true)
      _seq("User", "Device", comment: "Execute with params", color: orange)
      _seq("Device", "User", comment: "Return expectation values", dashed: true, color: orange)
      _seq("User", "Data", comment: [Store data to #class[`ExperimentalData`]])
    })

    // Data management
    _sep("Data Management")
    _seq(
      "User",
      "Data",
      comment: [
        Create #class[ExperimentalData] instance
      ],
      color: purple,
    )
    _seq("Data", "Storage", comment: "Save to disk", color: purple)
    _seq("Storage", "User", comment: [ Load #class[ExperimentalData] back], dashed: true, color: purple)
  }),
  caption: [This sequence diagram shows the flow of interaction between the user, `inspeqtor`, quantum device and storage.],
) <fig:data-module>


== Physics
#tldr[
  `physics` module is a collection of intense physics-related functions. This module include Unitary solvers based on ODE and Trotterization method. It has utility functions such as automatic rotating transformation function given Hamiltonian and frame, and functions to calculate fidelity metrics.
]

The main usages of `physics` module can be divivded into two proposes. (1) The first usage is to calculate the unitary propagator given the Hamiltonian and control. The overview flow of `physics` module is illustrate in @fig:physics-module. The second usage is to calculate a physical quantities regulary used in the quantum inforamtion processing such as the fidelity and the state tomography.

#spec[
  + Solve Schrödinger and Lindblad master equations given System Hamiltonian and collapse operator.
  + Support physical relate functions e.g. @agf, process fidelity, state fidelity and, tomography.
]

#let classy(body) = block(
  inset: 8pt,
  stroke: 0.5pt,
  radius: 3pt,
  body,
)
#let func_box(body) = rect(
  inset: 8pt,
  stroke: 0.5pt,
  radius: 0pt,
  body,
)

#figure(
  diagram(
    spacing: (4em, 2em),
    node-stroke: 1pt,
    edge-stroke: 1pt,

    // First group: envelope, qubit information, and transformation
    node((0, 0), [Qubit Information]),
    edge((0, 0), (0, 1), "->"),

    node((0, 1), [envelope: $bb(R)^n -> (bb(R) -> bb(C))$]),
    edge((0, 1), (0, 2), "->"),


    node((0, 2), align(left)[Transformation such as\ signal function]),

    // Arrow from first group to second group
    edge((0, 2), (0, 3.5), "->", []),

    // Second group: Hamiltonian and Solver
    node((0, 3.5), [$cal(H): bb(R)^n times bb(R) -> "Hermitian"(bb(N), bb(C))$]),
    edge((0, 3.5), (0, 4.5), "->"),

    node((0, 4.5), [Solver]),
  ),
  caption: [Overview of usage flow of `physics` module for unitary calculation.],
  gap: 2em,
) <fig:physics-module>




`physics` module provides Unitary solver functions based on ODE solver and trotterization method. The solver function $tilde(U)(bold(Theta); T)$ aims to solves for unitary $hat(U)(bold(Theta), T)$ given the control parameters $bold(Theta)$ at the final time $T$ as,
$
  tilde(U)(bold(Theta); T) approx hat(U)(bold(Theta), T) = cal(T)_+ exp { -i integral_(0)^(T) hat(H)_("total")(bold(Theta) ,s) d s }.
$
Usually, the final time $T$ is fixed, so unitary solver is used as a function of control parameters only. Most of the utility functions that provide ready to use solver is also return a solver as a function of control parameters only. However, as we adopt functional programming paradigm, solver is not necessary only a function of control parameters. Moreover, we also provide a `lindblad_solver` which solves the Lindblad master equation as given in @eq:master-equation.

For the ODE solver, we use ODE solver from `diffrax`, which is a general purpose package for differential equation solver. This package is one of the package in the `jax` ecosystem. The accuracy of the solution can be set by using `rtol` and `atol` arguments. For the trotterization based solver, we only use the first order approximation, where the step size can be set to adjust the approximation quality. The solvers accept Hamiltonian function with the same arguments signature. That is Hamiltonian function is a function of control parameters and time,
$
  cal(H) ( control: bb(R)^n times t: bb(R) ) -> "Hermitian"(bb(N), bb(C)).
$
The Lindblad master equation solver also accept the list of collapse operators ${C_i}_(i)$ which have the same dimension as the Hamiltonian $cal(H)$.

We provide common Hamiltonian of single qubit such as transmon qubit model. We also provide a rotating transmon qubit model which rotate the transmon qubit with the predefined frame analytically. However, in the case that user need to transform the Hamiltonian with another frame, we also provide an automatic transformation function, `auto_rotating_frame_hamiltonian`, that will apply a unitary transformation on the given Hamiltonian with the given frame. The expected usage of the function is as follows,
#source("hamiltonian")
The function returns a new function that the output of the original function will be transform with the given frame and returns it as a result.

In the experimental realization, the envelope function returns from #class[`ControlSequence`] may not be a function that is directly used in the Hamiltonian, but it might need to be transform to the form that match the experimental instrument. For example, in the superconducting qubit platform, the envelope function of the microwave pulse has to be transformed into signal function and then physically send it to the qubit. `inspeqtor` provides signal functions that accept different structure of control parameters. We implement a simple signal function. Below, we demonstrate how to use it using the gaussian envelope defined previously.

#Cell("signal")

The code snippet demonstrate the expected setup of the Hamiltonian is presented below.

#Cell("solver")

Various functions are available in the inspeqtor for the quality assessing related function in quantum experiments. Currently, we support the calculation of single-qubit case @javadi-abhariQuantumComputingQiskit2024. Here are the list of the functions and their description:
- `gate_fidelity`: Calculate the fidelity between two unitary operators defined as,
$
  F_("gate")(hat(U), hat(V)) = (tr[hat(U)^dagger hat(V)])/(sqrt(tr[hat(U)^dagger hat(U)] tr[hat(V)^dagger hat(V)]))
$
- `process_fidelity`: Calculate the fidelity of target superoperator using @eq:process-fidelity.
- `avg_gate_fidelity_from_superop`: Calculate the average gate fidelity from given superoperator as follows,
$
  macron(F)(hat(U), cal(E)) = (2 F_"process" (hat(U), cal(E)) + 1) / (3),
$
where $F_"process" (hat(U), cal(E))$ is the process fidelity defined in @eq:process-fidelity
- `to_superop`: Convert a given unitary operator into its superoperator representation using
$
  S_hat(U) = hat(U)^(*) times.o hat(U),
$
which is a tensor product of the conjugate of itself and itself.
- `state_tomography` Calculate the process tomography using a simple decomposition of a quantum state of $n$-qubit system in terms of the expectation value of observable as
$ rho = 1/(2^n) sum_i expval(hat(O)_i) hat(O)_i, $ where the index $i$ iterate over the tensor product of Pauli Matrix span the system of qubit.
- For the direct average gate fidelity esitmation, we design the atomic functions to be composed and a function that can be used repeatedly in the control calibration routine as follows:
  - `direct_AFG_estimation`: This function calculate the @agf as defined in @eq:agf. Its first argument is the result returns from `direct_AFG_estimation_coefficients` which only depends on the references unitary operator. While the second argument is the array of expectation values of observable in the same order of `default_expectation_values_order`.
  - `direct_AGF_estimation_fn`: For a convenient usage, user can also use this function to generate a single argument function (array of expectation values) that will calculate the @agf against the given reference unitary operator. Following is the intended usage
  ```python
  sx_agf_fn = sq.physics.direct_AGF_estimation_fn(sq.constant.SX)
  agf = sx_agf_fn(expvals) # expvals.shape == (18,)
  # agf.shape == (1,)
  agfs = jax.vmap(sx_agf_fn)(batched_expvals) # expvals.shape == (100, 18,)
  # agfs.shape == (100,)
  ```
  where user can use `jax.vmap` with`sx_agf_fn` in a cleaner way.

== Optimize
#tldr[
  This module implements optimization functions. The #func[`minimize`] that return parameters that minimize a loss function with gradient-based optimizer with optional upper and lower bound. The #func[`stochastic_minimize`] that return parameters that minimize a stochastic loss function with gradient-based optimizer with optional upper and lower bound.
]

We separate the opitmization baked in the `inspeqtor` into two types depending on their main proposes. (1) The `minimize` function is a gradient-based optimization function with optional parameters bound. This function is mainly used for control calibration. (2) The `train_model` function is a single line function for model training with dataset splitting into training, validating and testing dataset. The specification of #modu[`optimize`] is as follows.

#spec[
  + Support gradient-based optimization algorithm.
  + Support minimization of stochastic function.
  + Support bounded and unbounded optimization.
  + Support Blackbox optimization.
]


Let us show the `minimize` function first. We design `minimize` to be a simple function with minimal requirements to be called. Our objective is to find parameters $arrow(x)$ that minimize some scalar-output objective function $f(arrow(x))$, mathematically express as,
$
  arrow(x)^(*) = arg min f(arrow(x)).
$
User can define their own `loss_fn` (loss function $f(arrow(x))$) which accept only single argument of the parameter of type `pytree`. We also did not assume an inner implementation of the `loss_fn` aside from using `jax` for numerical calculation. For example, consider a simple function
$f(x) = (x - 4)^2$, which we wish to find $x$ that minimize $f(x)$. Below is a code snippet to achieve the propose.

#callisto.render(nb: json("../code/review_0003_minimize.ipynb"))

With this pattern, we can compose the `loss_fn` using predictive model which produces expectation values and `sq.physics.direct_AGF_estimation_fn` to calculate the scalar-output function of @agf for minimizer to find the control parameters that minimize the function. For example, the code snippet below show how can we leverage the utility functions provided by `inspeqtor` to quickly compose program to calibrate quantum control.
```python

sx_agf = sq.physics.direct_AGF_estimation_fn(sq.constant.SX)
ctrl_seq = ...
predictive_fn = ... # the predictive model.

def loss_fn(params: jnp.ndarray) -> tuple[jnp.ndarray, typing.Any]:
    AGF = sx_agf(predictive_fn(params))
    # Compute the cost.
    return (1 - AGF) ** 2, {"AGF": AGF}

init_params = ctrl_seq.sample_params(jax.random.key(0))
lower, upper = ctrl_seq.get_bounds()
ravel_fn , _ = (
    sq.control.ravel_unravel_fn(ctrl_seq)
)

res = sq.optimize.minimize(
    ravel_fn(init_params),
    jax.jit(loss_fn),
    sq.optimize.get_default_optimizer(1000),
    ravel_fn(lower),
    ravel_fn(upper),
)
```

For the #func[`stochastic_minimize`], the loss function receives two arguments. The first argument is the function parameters. The second parameter is a random key.

=== Bayesian Oprimization.

We also support an optimization of Blackbox function, i.e. non-gradient optimization via Bayesian Opitmization. By default, the flow will try to maximize the function. We use #pkg[gpjax] @pinderGPJaxGaussianProcess2022 for Gaussian Process computation, and we built on the algorithms in #pkg[bayex] @alonsoAlonfntBayex2025.


=== Control calibration process

Let us consider a simple open-loop control calibration pipline using `inspeqtor`. Assume for a momment that we can contruct a predictive model using #modu[`data`] and #modu[`models`]. In @fig:optimize-module, we illustrate the sequence diagram of how can user use #modu[`optimize`] in the open-loop control calibration.

#figure(
  chronos.diagram({
    import chronos: *

    _par("User", display-name: "User")
    _par("CostFunction", display-name: [Cost Function])
    _par("Optimizer", display-name: [#modu[`optimize`]], color: white)
    _par("Model", display-name: "Model")
    _par("Device", display-name: [Device])

    _sep("Starts with Trained Predictive Model")
    _seq("User", "CostFunction", comment: [Define target gate
      - #modu[`physics`]
      - Predictive model
    ])
    _seq("User", "Optimizer", comment: "Start Optimization")

    _loop("Optimization Steps", {
      _seq("Optimizer", "CostFunction", comment: "Evaluate(current_params)")
      _seq("CostFunction", "Model", comment: "Predict(current_params)")
      _seq("Model", "CostFunction", comment: "Return Expectation Values", dashed: true)
      _seq("CostFunction", "Optimizer", comment: "Return Loss", dashed: true)
      _seq("Optimizer", "Optimizer", comment: "Update Parameters")
    })

    _seq("Optimizer", "User", comment: "Return Optimized Control Parameters", dashed: true)

    _sep("Benchmarking")
    _seq("User", "Device", comment: "Execute(Optimized Params)")
    _seq("Device", "User", comment: "Return Measured Fidelity", dashed: true)
    _seq("User", "Model", comment: "Predict(Optimized Params)")
    _seq("Model", "User", comment: "Return Predicted Fidelity", dashed: true)
  }),
  caption: [
    A simple open-loop approach control calibration sequence diagram using #modu[`optimize`].
  ],
)<fig:optimize-module>


== Models
#tldr[
  This module contain the implementation of Blackbox model along with the elementary functions to create custom model in both statistical and probabilistic paradigm. Currently we support Blackbox model implement using `linen` and `nnx` of `flax` out of the box. With our unified abstractions, we implement `make_loss_fn` and `create_step` which make up for common training loop with customizability. The interface `*_predictive_fn` for @sgm model allows for utility function such as `make_predictive_resampling_model` which is a SGM-based model-agnostic function. We also provide a common `ModelData` dataclass for saving and loading model parameters.

  Our probabilistic sub-module provide a compatability layer to convert statistical model to the probabilistic model as well as components to create a custom neural network. However, we will discuss it in a @sec:probabilistic-module as it deserves a self-contain presentation.
]

The `models` module provide component functions to implement Blackbox part of the Graybox characterization method and beyond. The `library` sub-module serves as an example of how to implement a custom model that compatible with the rest of the `inspeqtor` package. We implement models in the module using #pkg[`flax`], a neural netowrk package in the `jax` ecosystem @jonathanheekFlaxNeuralNetwork2024.
We support both `linen` and `nnx` API of `flax` in the sub-module with the same name. In this section, we will discuss the design decision such as the interface of the functions and models instead of focusing on the actual code implementation, since it will likely change in thte future. Below is the specifications of the module which we aims to follows.

#spec[
  + Be able to construct a Predictive model in the form of @eq:predictive-model.
  + Equipped with Learning algorithm and logging ability.
  + Be able to Save and load predictive model to and from file system in a human-readable format without having to perform learning from scratch.
]

Currently, we provide support to different machine learning API via submodule. However, the name of the models and functions are essentially the same. However, the usage details may differ but similar.

=== Predictive model interface

From @eq:predictive-model, mathematically, the predicitve model is a function of control parameters that return expectation values. For each sub-module, we implement #func[`make_predictive_fn`] to make a predictive function signature mirror the @eq:predictive-model, but with the additional of unitary operators corresponding to the control parameters as a argument. Because, user can pre-calculate the unitary operators and supply it manually. Moreover, user can create the predictive function with only control parameters as a argument by define a new function combinding our predictive function and `solver` from elsewhere. The key argument required for #func[`make_predictive_fn`] is `adapter_fn` which decouple the specific implementation of machine leanring model from the interface of the predictive function.

The adapter function responsibles for transforming the output of the model to the expectation values. The adapter function is design to be machine learning implementation agnostic, i.e., it is the same regardless of the machine learning library used. In @tab:predictive-fn, we listed a pair of adapter function and the model.

#figure(
  table(
    columns: (auto, auto),
    inset: 10pt,
    align: left,
    table.header([Adapter function], [Model]),
    [#func[`observable_to_expvals`]
      - Parametrized operator in @eq:wo-operator.
    ],
    [#class[`WoModel`]],

    [#func[`unitary_to_expvals`]
      - Parametrized operator in @eq:noisy-unitary.
    ],
    [#class[`UnitaryModel`]],

    [#func[`toggling_unitary_to_expvals`]
      - Parametrized operator in @eq:toggling-unitary.
      - SPAM can be handle with `ignore_spam` argument.
    ],
    [#class[`UnitaryModel`]],

    [#func[`toggling_unitary_with_spam_to_expvals`]
      - Parametrized operator in @eq:toggling-unitary.
    ],
    [#class[`UnitarySPAMModel`]],
  ),
  caption: [
    List of the `adapter_fn` currently supported in `inspeqtor` and the corresponding name of the model.
  ],
) <tab:predictive-fn>

With the adapter function, we can construct the predictive function as required in specification. In the case that user needed to use the statistical model as a stochastic model, we provide a transform function `sq.models.probabilistic.make_predictive_resampling_model` that will use the expectation values predicted from the statistical model as a probability to sample from to produce a finite-shot expectation value. This is useful when user want to create a predictive model that intimidate the behavior of real device, or quantify the prediction uncertainty via a Monte-Carlo method.

=== Loss function interface

With the selected #func[`adapter_fn`], user can prepare a predefined loss function using #func[`make_loss_fn`]. The returned loss function is compatible with #func[`create_step`]. We predefine the metric function such as `sq.models.mse` that implements the Mean Squared Error loss function. The metric function can then pass to `evaluate_fn` argument of the #func[`make_loss_fn`]. User can define a custom metric function with the following signature,
$
  "metric" (expval(hat(O))^"model":  RR^(n times m) times EE[hat(O)]: RR^(n times m) times U: "Unitary"(N, CC)^n) -> cal(L): RR.
$
The first and second arguments are the predicted and experimental expectation values of $m$ combinations of size $n$ samples. The thrid argument is the ideal unitary operators corresponding to each sample.

=== Train model

Next, we discuss the `train_model` function. On the high-level, this function will train the predictive model with the following steps. (1) It first initialize `flax` model and `optax` optimizer. (2) Create the `train_step` and `eval_step` functions which responsible for updating model parameters based on the gradient of the loss function and evaluate the loss function without updating the model parameters using #func[`create_step`]. (3) It will loop over the training dataset with batch size equal to the size of validating dataset and log the metrics at each step. (4) At the end of the epoch (i.e. the last batch of the training dataset), it will evaluate validating and testing dataset at once and log the metrics for both of the dataset, and also call each user supplemented callback functions. (6) The function then return the model parameters, the state of an optimizer, and the log as a list of instances of `HistoryEntryV3`. The first and second returned values depends on the submodule implementation. In general, the `train_model` has the following call signature.

```python
def train_model(
    # Random key
    key: jnp.ndarray,
    # Data
    train_data: DataBundled,
    val_data: DataBundled,
    test_data: DataBundled,
    # Model to be used for training
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    # Loss function to be used
    loss_fn: typing.Callable,
    # Callbacks to be used
    callbacks: list[typing.Callable] = [],
    # Number of epochs
    NUM_EPOCH: int = 1_000,
    # Optional state
    **kwargs,
) -> [Unknown, Unknown, list[HistoryEntryV3]]:
```

The first argument is a random key from `jax`, which is useful when reproducibility is required. The same key would yield the same optimized result. The 2nd to 4th arguments are the training, validating and testing dataset respectively. We use the instance of `DataBundled` for the dataset to provide a typesafe interface for user to handle dataset required for model training using this function. The `optimizer` argument is the `optax` optimizer. The `**kwargs` input arguments depends on the submodule implementation. We provide a callback pattern for user to define a custom behavior for data logging. Each callback in the list will be called after the evalution of validating and testing dataset. We expect the callback to accept the model paramters, optimization state, and history entries. User can perform a custom logic based on the current state of the model in addition evaluating everything in the `loss_fn`. One of the usage is to save the intermediate state of the model training. For example, with `linen` module, user can then later resume the training on the latest checkpoint instead of starting from scratch by providing `model_params` and `opt_state` arguments and adjust the `NUM_EPOCH` argument accordingly.

In the case that user need to define a custom training logic, we also expose the utility functions that we use to compose `train_model` which are `sq.model.create_step` and `sq.utils.dataloader`. The `create_step` create training (update optimization state) and evaluating (just evaluate loss function) step. The `dataloader` is a python generator that will randomly shuffle the supply dataset and also yield only a batch for each iteraction. Let us take a look at the simple code snippet of how to use `dataloader`.
```python
for (step, batch_idx, is_last_batch, epoch_idx), (
        batch_p,
        batch_u,
        batch_ex,
    ) in dataloader(
        (train_p, train_u, train_ex),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCH,
        key=loader_key,
    ):
    pass
```
We assume that the first argument is a tuple of arrays. The length of the tuple can be arbitrary, which the second tuple yield from the generator will be a tuple of the same length as the first argument but each of the array will be a mini batch instead of full length array. For the first tuple yielded from the generator, we provide (1) a current opitmization step, the current batch index, the boolean indicate whether the current step is the last batch in the epoch or not, and the epoch index. Furthermore, user can specify the batch size, the number of epoch and the random key.


=== Serialization and Deserialization

We provide a simple dataclass #class[`sq.models.ModelData`] to save and load a model parameters from disk to memory. User can store a nested dict of strings as keys and values of generic Python types (e.g., `dict`, `list`, `float`, `int`, and `str`) and also `jnp.ndarray`. #class[`ModelData`] is compatible with a probabilistic predictive model, which will be discussed later in @sec:probabilistic-module. Typically, we expect users to store the information necessary to re-instantiate the model without model re-training. Currently, only model configuration and model parameters are two distinctive attributes that the #class[`ModelData`] stores. User can also check #class[`ModelData`] equality by a simple equality check statement `model_1 == model_2`, which will check both configuration and model parameters.

=== Characterization process

With the #modu[models] ready, we illustrate how can user utilize our library in the predictive model construction. Consider a simple model training loop without a the hyperparameter tuning, we show a sequence diagram for the pipeline in @fig:models-module. We note that the pipeline can be use in the hyperparameter tuning in a straightforward manner. It is up to user to design the library for the hyperparameter tuning task, for example, #pkg[`ray-tune`] @raytune, and #pkg[`optuna`] @optuna_2019.

#figure(
  chronos.diagram({
    import chronos: _alt, _loop, _note, _par, _sep, _seq

    _par("User", display-name: "User")
    _par("Data", display-name: [#modu[`data`]], color: white)
    _par("Optimizer", display-name: [#modu[`optimize`]], color: white)
    _par("Models", display-name: [#modu[`models`]], color: white)

    _sep("Data Preparation")
    _seq("User", "Data", comment: [Prepare & Split Data \ #func[`sq.utils.random_split`]])
    _seq("Data", "User", comment: "Return Training & Testing Data", dashed: true)

    _sep("Model Initialization")

    _seq("User", "Models", comment: [Define Model & Loss Function
      - #modu[`models`]
      - #func[`make_loss_fn`]
    ])
    // _seq("Optimizer", "Models", comment: "Initialize Parameters")

    _sep("Model Training")
    _seq("User", "Optimizer", comment: [Start Training
      - One liner using #func[`train_model`]
      - Custom loop using #func[`create_step`]
    ])

    _loop([For each epoch], {
      _seq("Optimizer", "Models", comment: [
        - Update parameters
        - Validates against \ testing data
      ])
    })

    _seq("Optimizer", "User", comment: "Return Trained Model", dashed: true)
  }),
  caption: [
    Sequence diagram for simple model training pipeline using #modu[`models`].
  ],
)<fig:models-module>


== Probabilistic <sec:probabilistic-module>

#tldr[
  Probabilistic module is a collection of probabilistic model implementations relying on probabilistic programming language `numpyro`.
]

// #task[
//   + We implement bayesian neural network model mirroring the blackbox models implemented in #modu[`models`].
//     + We support converting `linen` models to probabilistic through `numpyro.contrib.module`.
//     + We also implement our simple version of @bnn library which can be use to implement MLP-based model. This serves as a comfirmation of the underlying mechanism of @bnn from other sources. Furthermore, we relies on the custom matrix multiplication for #modu[`boed`].
//   + We also implement a function to create a guide for the probabilistic model similar to `autoguide`. Our custom guide is designed by keeping flexibility in mind. We support functional based variational parameters initialization. The guide return the samples from the variational distribution which is particular useful when contructing a posterior prediction model for control calibration using SVI method.
//   + We implement serveral utility functions for:
//     + Contructing a posterior sample which can be used as a prior for a probabilistic model initialization.
//     + Making a predictive model with the same signature as @sgm version.
//     + Functions to perform SVI.
// ]

In this section, we are going to introduce a concept of probabilistic programming language (PPL). Currently, we are relying on #pkg[`numpyro`] for performing probabilistic programming. However, the concept introduce here should be independent of the choice of PPL. We fill focus on how to use PPL in a quantum device characterization. In particular, we will dicuss the implementation of the probabilistic predictive model which involve implementing @bnn. Let us consider the rough specifications of the #modu[`probabilistic`] as follows.

#spec[
  + Be able to construct a Predictive model in the form of @eq:posterior-predictive-model.
  + Equipped with Inference algorithm and logging ability.
  + Be able to Save and load predictive model to and from file system in a human-readable format without having to perform inference from scratch.
  + Be able to setup the model with priors distribution.
]

The specification above is similar to the specification of #modu[`models`] with addition of be able to setup model with prior distribution and furthermore, we require a probabilistic predictive model instead of point predictive model in @eq:predictive-model.

=== Predictive model

// Models discussion
The posterior predictive distribution defined in @eq:posterior-predictive-model can be viewed as a probabilistic predictive model or simply predictive model in a probability machine learning paradigm. To achieve the probabilistic predictive model in our case is to replace part of the predictive model with probability distributions. For example, consider the predictive model constructed purely from parametrized model of the system model. We can assume a distribution for one or more system parameters. Then we can perform bayesian inference to infer for the system parameters. Alternatively, we can consider a system modelled by neural network such as was done in Graybox characterization method. We then promote a neural network model to the @bnn.

There are multiple options to transform DNN to @bnn, e.g. using functions in `numypro.contrib.module` In the `inspeqtor`, we provide multiple methods to define a probabilistic predictive model based on Graybox characterization method. First, user can convert `flax` models using `make_flax_probabilistic_graybox_model` to transform compatible `flax` models to the probabilistic Graybox model. Consider the code snippet belows.

#figure(
  [
    ```python
    from inspeqtor.models.probabilistic import make_flax_probabilistic_graybox_model

    base_model = ... # Flax model
    adapter_fn = ... # Transformation function
    flax_module = ... # Function to promote DNN to BNN

    graybox_model = make_flax_probabilistic_graybox_model(
        name="graybox",
        base_model=base_model,
        adapter_fn=adapter_fn,
        prior=sq.probabilistic.dist.Normal(0, 1),
        flax_module=flax_module,
    )
    ```
  ],
  caption: [
    Code snippet for transforming compatible `flax` moodel to @bnn.
  ],
) <code:flax-to-bnn>


From the @code:flax-to-bnn, there three variables that user has to appropiately specify which are `base_model` `adapter_fn`, and `flax_module`. The argument `flax_module` is to let user choose to transform DNN to @dnn or just to simply make the statistical module inferrable with SVI method. This is useful in the case that user want to train the statistical model by minimizing negative ELBO instead of minimizing MSE loss. User has to choose the appropiate `flax_module` depends on which sub-module chosen, i.e., `linen` or `nnx`.

Alternatively, user can define a custom @bnn from scratch by using our neural network library implementaion. Our implementation of `dense_layer` which is the building block of @mlp model. The layer uses a specialized matrix multiplication algorithm which handle operations for vectorization of the probabilistic model using `numpyro`. The ability to support vectorization by `numpyro` is necessary for @boed, e.g. algorithm proposed in  @fosterDeepAdaptiveDesign2021.

To take an advantage of using probabilistic machine learning, the predictive function in #modu[`probabilistic`] predicts a binary measurement value by default. We separate the layer of graybox prediction and the compelete predictive function because user do not has to worry about the correct usage of observable in `numpyro`, making debugging easier. Because we will automatically use the expectation values predicted by the user-supplied predictive model as the probability to sample from the Bernoulli distributions as defined in @eq:bern-based-obs.
```python
model = sq.models.probabilistic.make_probabilistic_model(
    predictive_model=graybox_model,
    shots: int = 1,
    block_graybox: bool = False,
    separate_observables: bool = False,
    log_expectation_values: bool = False,
)
```
User can customize the behavior of the probabilistic model. For example, user can specify the number of predicition shot of binary measurement result similar to what happen in the real device. This is useful when the model will be used in other probabilistic model. Furthermore, we might want to use the model within other model without training it, we can set `block_graybox` to `True` to hide the model from optimizer.

// Moreover, user can choose between the observation distributions defined in @eq:normal-based-obs and @eq:bern-based-obs with this abstraction.
// ```python
// predictive_fn = sq.probabilistic.make_predictive_fn(
//     posterior_model, sq.probabilistic.LearningModel.BernoulliProbs
// )
// ```

=== Model Infernece
// SVI discussion
As disucssed in @sec:how-to-pml, we consider the variational inference as a main method to perform inference. The requirements for SVI are (1) the probabilistic moedl which we are discussed earlier, and (2) the variational distribution which we will discuss in the following.

// Guide (variational distribution)
The variational distribution is referred to as a guide in `numpyro`. We also adopt the term for the consistency. On the high-level, guide is a function that parametrized the distributions found in the probabilistic model. These parameters are called variational parameters. The variational distributions in the guide does not necessary have to be the same as the disributions in the model. For instance, they can be more simpler so that it is easier to optimize. Next, the we sample from the variational distributions and produce the observations (samples from the target distribution). SVI is basically optimize for a variational parameters that produce the observations that minimize the negative ELBO.

We can automatically generate the guide corresponding to the probabilistic model using `autoguide` module provided by `numpyro`. Alternatively, for the customizability specifically to the @bnn inference, we also implement our version of guide automatic generation functions which accept advance variational parameters initialization strategy. We also implement a good default initialization strategy which avoid numerical unstability in optimization for the @bnn. Moreover, our guide provide the option to "block the sample site" which is a design to allow the guide to be used with its model in other probabilistic model.

For the inferencing loop, we did not implement a single-line call function to perform the SVI, since `numpyro` provide the built-in method for this already. Instead, we implement components for inferencing loop building which extended from `numpyro` to perform both update and non update call separately. This is similar to the implementation for #modu[`models`].

// To be write for the last specification.
For both ways of defining predictive model (from @dnn and from scratch), we support setting the prior distributions. To set the prior distribution with posterior distribution is similar to transfer learning that the parts of weights of statistical model is initialized by the weights of pre-trained model. However, with prior setting, instead of point weighted transferred, the dsitribution of the pre-trained weights are transferred.

=== Characterization process

Similarly to the sequence diagram presented in #modu[models], we illustrate how can user utilize our #modu[`probabilistic`] in the posterior predictive model construction. Consider a simple model training loop without a the hyperparameter tuning, we show a sequence diagram for the pipeline in @fig:probabilistic-module.

#figure(
  chronos.diagram({
    import chronos: _alt, _loop, _note, _par, _sep, _seq

    _par("User", display-name: "User")
    _par("Data", display-name: [#modu[`data`]], color: white)
    _par("Prob", display-name: [#modu[`probabilistics`]], color: white)
    // _par("Models", display-name: [#modu[`models`]], color: white)

    _sep("Data Preparation")
    _seq("User", "Data", comment: [Prepare & Split Data \ #func[`sq.utils.random_split`]])
    _seq("Data", "User", comment: "Return Training & Testing Data", dashed: true)

    _sep("Model Initialization")

    _seq("User", "Prob", comment: [Define Model & Prior, Guide,
      - #modu[`models`] using \ #func[`make_flax_probabilistic_graybox_model`]
      - From scratch using #modu[`probabilistics`]
    ])

    _sep("Model Inference")
    _seq("User", "Prob", comment: [Start Inference
      - One liner using `SVI.run`
      - Custom loop using #func[`create_step`]
    ])

    _seq("Prob", "Prob", comment: [
      - Update parameters
      - Validates against \ testing data
    ])

    _seq("Prob", "User", comment: "Return Trained Model", dashed: true)
  }),
  caption: [
    Sequence diagram for simple model training pipeline using #modu[`probabilstic`].
  ],
)<fig:probabilistic-module>

=== Distance measures of probability distribution

The module also provide utility functions to calculate the distance measure of the probability distribution. The KL-divergence between two probability density functions $p$ and $q$ we use in calculate is given by
$
  D_"KL" (p || q) = sum p log(p / q)
$
Currently, we support the calculation of KL divergence from the probability mass function (PMF) #func[`kl_divergence`]. Alternatively, since the $D_"KL"$ can be numerical unstable, we implement #func[`safe_kl_divergence`] which safely convert infinity to zero. We support the calculation from PMF since the data to be calculated are samples. Thus the binning strategy is needed to approximate the PMF.

KL divergence is asymmetric in its argument, i.e., $D_"KL" (p || q) != D_"KL" (q || p)$. Alternative choice of measure of distance of the probability distribution is @jsd. The @jsd between two probability density functions $p$ and $q$ is given by
$
  D_"JSD" (p || q) = 1 / 2 (D_"KL" (p || m) + D_"KL" (q || m)); m = (p + q) / 2.
$

For the sake of convenient, we implement a utility function to calculate the @jsd from samples of the two distributions #func[`jensenshannon_divergence_from_sample`].

== Bayesian Optimal Experimental Design

#tldr[
  This module provides capability to calculate @eig using method in @fosterVariationalBayesianOptimal2019.
]

From the mathematical point of point we discussed in @sec:route-to-data-efficient, the concept of @boed revolve around calculation of @eig and making a decision using @eig. We implement the variational marginal estimator of @eig proposed in @fosterVariationalBayesianOptimal2019.

#spec[
  1. Given experimental designs, probabilistic model, be able to estiamte @eig
]

We implement a single function call #func[`estimate_eig`] to estimate the @eig for the given experimental designs. The function signature is presented in the code snippet below.
```python
def estimate_eig(
    key: jnp.ndarray,
    model: typing.Callable,
    marginal_guide: typing.Callable,
    design: jnp.ndarray,
    *args,
    optimizer: optax.GradientTransformation,
    num_optimization_steps: int,
    observation_labels: list[str],
    target_labels: list[str],
    num_particles: tuple[int, int] | int,
    final_num_particles: tuple[int, int] | int | None = None,
    loss_fn: typing.Callable = marginal_loss,
    callbacks: list[] = [],
) -> tuple[jnp.ndarray, dict[str, typing.Any]]:
```
User can supply the estimator in the form of a loss function `loss_fn`passed as an argument of #func[`estimate_eig`]. Although, we only support variational marginal estimator currently, we expect the function to be able to used with other estimators. The difference of our implementation and the `pyro.contrib.oed` is that we also support supplement of the additional positional arguments `*args` to the probabilistic model `model`. In our case, we required this functionality for the unitary operators corresponding to the control parameters (`design`). User needs to specify the observed values by using the label `observation_labels` for the estimator to calculate the @eig. The `target_labels` refer to the target parameters that we wish to find the estimate the @eig given the design. However, in the case of the variational marginal estimator, they did not use the target labels in @eig calculation. For the logging purpose, we use callback patterns instead of directly return the list of metrics for each step. Because, the array can be large and consume a lot of computational memory. User can selectively log the metrics as appropiate.

== Utils

#tldr[
  The utils module contains unitility functions and classes useful for data transformations, visualization, common patterns and etc.
]

`inspeqtor` aims to be the opinionated characterization and calibration framework. We realize that different harewares have different requirements. Instead of dictate every aspect of device characterization and calibration within one pipeline, we defined abstraction of the pipeline and implement useful functions intended for user to compose them freely. While, the abstractions and pipelines we defined serve as checkpoints in which the code implementations are reusable, reproducible and consistent.

In the #modu[`utils`], we define serveral functions and classes for data transformation, visualization, and shortcut to the common usecases. In @tab:utils, we list some of the examples of the functionality expose via #modu[`utils`].

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    align: (left, left),
    stroke: 0.4pt,
    table.header([*Goodies*], [*Description*]),
    [#const[`default_expectation` \ `_values_order`]],
    [Provides a list of `ExpectationValue` for a single qubit system. The order of the list is what #pkg[`inspeqtor`] usually uses in iteration.],

    [#func[`finite_shot_quantum_device`]],
    [A function that return finite-shot expectation value given number of shots.],

    [#func[`get_measurement_probability`]],
    [Calculate the probability of measuring each projector of tensor product of Pauli operators],

    [#func[`count_bits`]], [Count the integer representation of bits and return as a dictionary.],

    [#func[`prepare_data`]],
    [Prepare data from the given #func[`ExperimentalData`], #func[`ControlSeqeunce`], and #func[`whitebox`], and return the instance of #class[`LoadedData`]],

    [#func[`plot_control_envelope`]], [Plot the complex waveform on the given Matplotlib's axes],

    [#func[`plot_expectation_values`]],
    [Plot the trajectories of the given unitaries operators in terms of the expectation values, currently support only a single qubit unitary operators.],
  ),
  caption: [
    Descriptions of the auxillary modules in `inspeqtor`. Note that, the details may change in the future. We encourage those who are interested in using the package to check out the document.
  ],
) <tab:utils>

