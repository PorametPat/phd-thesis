#import "@preview/drafting:0.2.2": inline-note
#let caution-rect = rect.with(inset: 1em, radius: 0.5em)
#import "../utils.typ": implement-note, todocontinue, todoneedcite, todoneedwrite
#import "chapter_3.typ": mseeloss
#import "@preview/gentle-clues:1.2.0": *
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/chronos:0.2.1"
#import "@preview/callisto:0.2.4"

= Data efficient experiments
// #inline-note(rect: caution-rect, fill: orange.lighten(80%))[
//   Backup plan for thesis
//   - Use the data variance to incrementally increase the number of the dataset
//   - Try to design an example control that we know some paramter will give more EIG than other.
//   - The control that move state to the axis will maximize information gain in other axes?
// ]

This chapter will demonstrate how Inspeqtor can help researchers characterize and calibrate a quantum device. First, we will introduce the noisy quantum device models that we will use in this chapter. Second, we will characterize the quantum device with the statistical Graybox model and show how we can naively quantify prediction uncertainty. Then, we will use the predictive models to calibrate for a target quantum gate. Next, we will use the Probabilistic Graybox model to characterize the quantum device and calibrate the quantum gate. We will go further by using @boed to characterize the quantum device in a data-efficient manner. We will compare and analyze the data usage of @boed -based and random strategies.

== Noise model

Quantum device is expected to be a large system of qubits. Complete characterization of the system in a classical representation is inefficient, if not impossible. For a large-scale characterization, only some system properties are of interest, and the information will be used to improve the system's performance. In the context of the characterization of the device into the predictive model, we only need to consider a subsystem of the device, i.e., only up to a two-qubit system. In this chapter, we will consider a single-qubit quantum device for demonstration purposes.

Consider a system of a qubit with frequency $omega_q$ with a time-dependent control signal as a function of control $s(bold(Theta),t)$ and drive strength $Omega$. An ideal Hamiltonian of superconducting qubits in the rotating frame with respect to the qubit frequency is defined as
$
  H_("rot") = 2 pi Omega s(bold(Theta), t) (cos(2 pi omega_q t) sigma_x - sin(2 pi omega_q t) sigma_y)
$ <eq:hamiltonian-rot>
where $sigma_i in {X, Y, Z}$ is a Pauli matrix. The control signal is a function of envelope function $h(bold(Theta), t)$ with driving frequency $omega_d$ typically set equal to the qubit frequency and phase $phi$ defined as follows,
$
  s(bold(Theta), t) = Re { h(bold(Theta), t) exp(i (2 pi omega_d t + phi)) }
$
In a realistic setting, multiple sources of noise influence the dynamics of the system. However, for the simplicity and interpretability of our numerical study, we consider the case where the ideal Hamiltonian is perturbed by $hat(H)_("noise") = Delta sigma_x$. Thus, the total Hamiltonian is,
$
  H_("total") = H_("rot") + H_("noise")
$
This particular choice of noise model will only affect the ideal evolution in the X-axis at the level of the Hamiltonian.

== Level of Characterization

We now demonstrate the uses of `inspeqtor` to characterize the quantum device. Here, we are going to use (1) @sgm, (2) @mcdgm, and (3) @pgm to characterize the same dataset from the same noisy device. The steps are listed as follows,
+ Define the control
+ Perform expreiments and save the data to disk.
+ Load and prepare the dataset into training and testing dataset.
+ Model training/inference
+ Benchmark model with @jsd and $cal(L)_("MSE[E]")$

We now explain in details of each step. The first four steps are the same for each predictive model.

*The control:* Motivated by the noise model, we consider a single-parameter control in the Pauli X direction. The control envelope is a Gaussian shape with an area under the curve as a control parameter. The control is a rotation along the X-axis of the Bloch sphere, where the area is the rotation angle. The control is the same as defined in @eq:gaussian-envelope. Thus, we define the `ControlSequence` for inspeqtor using the code snippet defined in @code:gaussian-seqence.

*Perform experiments and save to disk:* For the synthesis dataset, the inspector provides a convenient predefined function `sq.predefined.generate_experimental_data` to generate the noisy dataset given the system Hamiltonian, the control, and the other necessary information. We use the following code snippet to define the Hamiltonian of the noisy device.

```python
def get_data_model(
    detune: float, trotterization: bool = False, trotter_steps: int = 1000
) -> sq.utils.SyntheticDataModel:
    qubit_info = sq.predefined.get_mock_qubit_information()
    control_sequence = sq.predefined.get_gaussian_control_sequence(
        qubit_info=qubit_info
    )
    dt = 2 / 9

    ideal_hamiltonian = partial(
        sq.predefined.rotating_transmon_hamiltonian,
        qubit_info=qubit_info,
        signal=sq.physics.signal_func_v5(
            get_envelope=sq.predefined.get_envelope_transformer(
                control_sequence=control_sequence
            ),
            drive_frequency=qubit_info.frequency,
            dt=dt,
        ),
    )

    total_hamiltonian = sq.predefined.detune_x_hamiltonian(
        ideal_hamiltonian, detune * qubit_info.frequency
    )

    ode_solver = sq.predefined.get_single_qubit_whitebox(
        hamiltonian=total_hamiltonian,
        control_sequence=control_sequence,
        qubit_info=qubit_info,
        dt=dt,
    )

    trotter_solver = sq.physics.make_trotterization_whitebox(
        hamiltonian=total_hamiltonian,
        control_sequence=control_sequence,
        trotter_steps=trotter_steps,
        dt=dt,
    )

    solver = ode_solver if not trotterization else trotter_solver

    return sq.utils.SyntheticDataModel(
        control_sequence=control_sequence,
        qubit_information=qubit_info,
        dt=dt,
        ideal_hamiltonian=ideal_hamiltonian,
        total_hamiltonian=total_hamiltonian,
        solver=solver,
        quantum_device=None,
        whitebox=None,
    )

```


We define the ideal Hamiltonian first, and will reuse it for the construction of the Whitebox. We then use the ideal Hamiltonian to define the total Hamiltonian. Now, we generete the dataset using the following code snippet.
#figure(
  caption: [],
  // kind: "code",
  // supplement: "Listing"
)[
  ```python

  detune = 0.001
  trotterization = True
  trotter_steps = 10_000
  sample_size = 1000
  shots = 1000

  data_model = get_data_model(
    detune=detune,
    trotterization=trotterization,
    trotter_steps=trotter_steps,
  )

  data_key = jax.random.key(0)

  exp_data, control_sequence, unitaries, noisy_simulator = (
    sq.predefined.generate_experimental_data(
        key=data_key,
        hamiltonian=data_model.total_hamiltonian,
        sample_size=sample_size,
        shots=shots,
        strategy=sq.predefined.SimulationStrategy.SHOT,
        get_qubit_information_fn=lambda: data_model.qubit_information,
        get_control_sequence_fn=lambda: data_model.control_sequence,
        method=sq.predefined.WhiteboxStrategy.TROTTER,
    )
  )
  ```
] <code:generate-dataset>
We can pass `total_hamiltonian` to the function as simple as show in @code:generate-dataset:8. The `strategy` argument in @code:generate-dataset:11 is the argument to specify how to calculate expectation value. In this case, we use finite-shot estimate strategy. We also specify the simultor to use Trotterization method to calculate the evolution of the device in @code:generate-dataset:14. The function returns a simple `NamedTuple` containing the necessary data and functions for model training. In this study case, we want to share the synthetic data across characterization appraoch, so we save the data using `sq.predefined.save_data_to_path`. The `path` is the name of the folder for storing the data, if the folder does not exist, the function will automatically create it along with the parent folders if necessary.

*Load and prepare dataset:*
Next step is to read the dataset from local storage. We  use `sq.predefined.load_data_from_path` for this task as follows.
```python
from inspeqtor.experimental.predefined import HamiltonianEnum

loaded_data = sq.predefined.load_data_from_path(
    data_path,
    hamiltonian_spec=sq.predefined.HamiltonianSpec(
        method=sq.predefined.WhiteboxStrategy.TROTTER
        trotter_steps=TROTTER_STEPS,
        hamiltonian_enum=HamiltonianEnum.rotating_transmon_hamiltonian
    ),
)
```
The predefined function allows us to specify the Whitebox specifications such as solver method and the Hamiltonian. The `HamiltonianSpec` is a predefined dataclass to specifiy the ideal Hamiltonian and solver method to use. In this case, the Hamiltonian of Whitebox is the same as the ideal Hamiltonian defined in @eq:hamiltonian-rot. We can now solve the Schrödinger equation for an ideal unitary operator to be combined with results from Blackbox. To solve for a unitary operator from the Hamiltonian, we use the Trotterization method offered by inspeqtor. The Whitebox function is a function of the control parameter. It will be used to precompute the unitary operators for the training and testing datasets and can also be used to compute the unitary operators on demand.

After we load the dataset, we split them into training and testing dataset. We can access the control parameters, and their unitary and expectation values from `loaded_data` which is a simple data holder. `inspeqtor` also provide a simple function to randomly split the dataset into two sets using `sq.utils.random_split`.
```python
key = jax.random.key(0)
key, random_split_key, train_key, prediction_key = jax.random.split(key, 4)
(
    train_control_parameters,
    train_unitaries,
    train_expectation_values,
    test_control_parameters,
    test_unitaries,
    test_expectation_values,
) = sq.utils.random_split(
    random_split_key,
    int(loaded_data.control_parameters.shape[0] * 0.1),  # Test size
    loaded_data.control_parameters,
    loaded_data.unitaries,
    loaded_data.expectation_values,
)
```

To test our model performance, we generate a specialized test samples apart from testing dataset. Specifically, we want to test if our model can predict distribution with the same statistics as produced by device. We now define a helper function to sample one control parameters and use the device solver to repeatedly generate expectation values for multiple times to form a distribution.
```python
def make_test_sample_fn(
    data_model: sq.utils.SyntheticDataModel,
    shots: int,
):
    _, to_array_fn = sq.control.get_param_array_converter(data_model.control_sequence)

    def generate_test_distribution(key: jnp.ndarray, sample_size: int = 100):
        control_key, solver_key = jax.random.split(key)

        control_param = to_array_fn(
            data_model.control_sequence.sample_params(control_key)
        )

        expvals = jax.vmap(sq.utils.shot_quantum_device, in_axes=(None, 0, None, None))(
            control_param.reshape(1, -1),
            jax.random.split(solver_key, sample_size),
            data_model.solver,
            shots,
        )

        return control_param, expvals.squeeze(axis=1)

    return generate_test_distribution
```
We can use the function with the following code snippet.
```python
generate_test_distribution = make_test_sample_fn(
  data_model,
  shots=shots
)
test_control_param, test_samples = generate_test_distribution(jax.random.key(0))
```
With `test_samples`, we can test our model performance by predicting a distribution of `test_control_param` and measure the @jsd of the distribution of the expectaction values and calculate $cal(L)_("MSE[E]")$. Although @jsd is a direct measure of how close the distributions are to each other, we use $cal(L)_("MSE[E]")$ as an indicator of how much the mean of predicted distributions are close to the ideal values.

We previously discussed the mathematical construction of the Graybox model in @sec:graybox-characterization. Next, we will disucss the implementation of Graybox model in the code, specifically for the study in this example. We already pre-computed the Unitary operators associate to the control parameters using pre-defined Whitebox. So, we now turn our attention to the Blackbox part. The architecture of Blackbox is present in @fig:blackbox-arch. On the high-level, the Blackbox model performs the following operations.
+ We scale the control parameters with factor of $2pi$ so that the maxmimum value of each feature is $1$ and then transform it with 4th-order polynomial feature transformation. We use the following code snippet.
  ```python
  def custom_feature_map(x: jnp.ndarray) -> jnp.ndarray:
      return sq.predefined.polynomial_feature_map(
        x / (2 * jnp.pi),
        degree=4
      )
  ```
  The `sq.predefined.polynomial_feature_map` is a function that maps $x -> [x, x^2, ..., x^n]$ to the $n$th order.
+ The Blackbox take in augmented features to the  Shared layers. The Shared layers is a @mlp neural network with `relu` activation function.
+ The output is then sent to three Pauli layers and being processed independently. Again, Pauli layers is a @mlp Neural Network
+ Finally the Hermailtian layers will return parameters that parametrized the Hermitian matrix in @eq:wo (See `sq.model.Wo_2_level_v3` for more details.)
With the Hermitian Matrices, we combine the Unitary operator from Whitebox to produce the expectation values using @eq:exp-wo. The technical differences of each model will be discuss along with its training/inference and results disucssion.

#figure(image("../figures/fig_model_arch.svg"), caption: [
  The architecture of Blackbox model of Graybox model.
]) <fig:blackbox-arch>

=== Statisitical approach
With preparation in-place, we now begin to characterize the quantum device with @sgm. From @fig:blackbox-arch, the model parameters (weights and biases) are point values. We cosntruct the Blackbox with the predefined model provided by `inspeqtor` as follows.

// Get more functions preconfigured for this notebook
#let (display, result, source, output, outputs, Cell, cell) = callisto.config(
  nb: json("../code/chapter-5-random/note_0001_SGM.ipynb"),
)

#source(
  "model",
)

Here, the `sq.model.make_basic_blackbox_model` is the function that generate a Blackbox implemented in `flax`, where user can supplement custom activation functions to the final layers of the Hermitian layers. In above example, we simply use the default activation functions. The following call with `hidden_sizes_1` and `hidden_sizes_2` are the Initialization of `flax` model which are specifications of Shared and Pauli Layers. Both arguments accept a list of integer, where each integer is the width of Dense layer in @mlp. In this case, Both Shared and Pauli layers will consist of single Desne layer with width of 10.

We can now use a predefined function `train_model` to train the @sgm. In practice, we might want to monitor the performance of the model as we train the model. One of the way is to take advanage of the `callbacks` argument of the `train_model` function which will call each callback in the list once at the end of the epoch. In this case, we use progress bar from `rich` to track the progress of model training.

#source(
  "training",
)

We save the model using the following code snippet.

#source("save_model")

And we can load the model back using the following code snippet.

#source("load_model")

We plot the #mseeloss of training and testing dataset in @fig:sgm-loss.

#figure(
  result("plot_loss"),
  caption: [
    The training and testing loss of SGM.
  ],
) <fig:sgm-loss>

To properly benchmark the model performance, we test it on the test control parameters. The pre-generated `test_samples` has a shape of `(100, 18)` which is 100 samples for each 18 combinations of expectation value distribution. We then use the @sgm to predict the distribution given `test_control_param` using the following code snippet.

#source("predict_test_controls")

Now we can calculate @jsd between `test_prediction` and `test_samples` by using the following code snippet.

#source("calculate_jsd")

We also plot the difference of mean and variance of both test and prediction with @jsd in @fig:sgm-stats.
#figure(
  result("plot_metrics"),
  caption: [
    @jsd, mean and varaince prediction of @sgm to the test control point.
  ],
) <fig:sgm-stats>

We can see that @sgm is capable of predicting near the empirical result. With accurate mean value, it should be natural that the variance is accurate in the case that the noise does not change the statistics of the expectation value. The disadvantage of using @sgm is that we cannot target the model parameters for data utilization propose using @boed.

=== Monte-Carlo Dropout Approach

We can initialized @mcdgm with `sq.model.make_dropout_blackbox_model` where the arguments are the activation functions of Pauli layers, same as in the case of @sgm.

#let (display, result, source, output, outputs, Cell, cell) = callisto.config(
  nb: json("../code/chapter-5-random/note_0002_MCDGM.ipynb"),
)

#source("model")

However, `inspeqtor` did not provide a utility function to train @mcdgm out-of-the-box as in the case of @sgm. Instead, we can compose our training loop using low level utility functions as follows.

#source("training")

The code snippet above is roughly similar to the algorithm used in `sq.optimize.train_model`. The main difference is that we have to pass the random key when performing forward pass to the model. The training result can be save using `ModelData` as well.

#source("save_model")

Again, we can load the model using the following code snippet in the same manner as was done in the case of @sgm.

#source("load_model")

We use the model to test against `test_samples` and plot the results in @fig:mcdgm-stat. Compare to the statistics of @sgm in @fig:sgm-stats, we can see that the predictions made by @mcdgm are not good. The calculated @jsd are close to the upper limit value at $ln(2)$. So the predicted distribution cannot be used. This is to be expected, since @mcdgm is not design to capture the distrubtion of the observed data.

#figure(
  // image("../figures/fig_MCDGM_stats.svg"),
  result("plot_metrics"),
  caption: [
    @jsd, mean and varaince prediction of @mcdgm to the test control point.
  ],
) <fig:mcdgm-stat>

=== Probabilistic approach <sec:pgm-approach>

For the probabilistic model inference, we need additional step to transform observable to raw binary measurement results.

#let (display, result, source, output, outputs, Cell, cell) = callisto.config(
  nb: json("../code/chapter-5-random/note_0003_PGM.ipynb"),
)

#source("transform_data")

We select `WoModel` as a based model, then convert it to @bnn using the following code snippet.

#source("model")

We then perform stochastic vairational inference using functions exported by `inspeqtor` which integrated with `numpyro`.

#source("training")

Similarly, we can save, load, and prepare model for prediction using following code.

#source("save_model")
#source("load_model")

We plot the results in @fig:pgm-stat. The prediction performance of @pgm is similar to the @sgm. This is not surprising since from @fig:sgm-loss, we can see that the model is trained to the optimal point. Consequently, the model predict the exact expectation values with high accuracy, and since the distribution of the finite-shot expectation value depends only on the expectation value, @jsd of the @sgm should close to zero already.

#figure(
  result("plot_metrics"),
  caption: [
    @jsd, mean and varaince prediction of @pgm to the test control point.
  ],
) <fig:pgm-stat>

=== BOED approach <sec:boed-experiment>

Now, we are going to demonstrate the advantage of using probabilistic model which is the ability to use it for calculation of @eig and use the information in device characterization. We are going to show how to use the model in sequential experiments and compare its performance to the approach without @eig.

The study setting is the same as appraoches discussed previously with an modification to the data acquisition step. Previously, we uniformly sample the control parameters to construct the dataset and perform the experiment on the device in a single batch fashion. However, in this study, we are going to sequentially perform experiment and acquire data in multiple batches. For each addition batch, we then perform characterization and benckamrk the model. Next, we perform the next experiments for new batch of data based on the strategy (policy). The steps are repeatedly performed until some conditions are met. The strategies we are interested are the following.
1. *Random strategy*: This is a strategy that does not utilize @eig for experimetnal design at all and simply chose the experimental design randomly. This strategy is equivalent to the approach discussed in @sec:pgm-approach. We illustrate the flow of the strategy in @fig:random-flow.

#figure(
  chronos.diagram({
    import chronos: *

    // Define participants
    _par("User", display-name: "User")
    _par("Strategy", display-name: "Random\nStrategy")
    _par("Model", display-name: "Model")
    _par("Device", display-name: "Device")

    _seq("User", "Strategy", comment: "Prepare experiment")

    _loop("Random Loop", {
      _seq("Strategy", "Strategy", comment: "Random new experiments")
      _seq("Strategy", "User", comment: "Select experiments", dashed: true)

      _seq("User", "Device", comment: "Perform experiment", color: purple)
      _seq("Device", "User", comment: "Measurement data", dashed: true, color: purple)

      _seq("User", "Model", comment: [Characterize @pgm])
      _seq("Model", "Device", comment: [Benchmark @pgm])
      _seq("Device", "User", comment: "Performance metrics", dashed: true)
      _seq("Strategy", "Strategy", comment: "Check termination", color: red)
    })
  }),
  caption: [
    The sequence diagram of the random strategy.
  ],
)  <fig:random-flow>
2. *Subspace strategy*: This strategy divide the control parameters space into multiple subspaces and then select the designs that maximize @eig within the subpsace. Note that this stategy is the representitive of @boed appraoch.Since we are performing inference for regression (@bnn) predictive model, so we want our model to be able to generalized to unseen data without overconfident. Thus, instead of chosing designs that maximize @eig globally which will bias our dataset heavily, we chose the designs that maximize @eig in their own subspace.  We illustrate the flow of the strategy in @fig:subspace-flow.

#figure(
  chronos.diagram({
    import chronos: *

    // Define participants
    _par("User", display-name: "User")
    _par("Strategy", display-name: "Subspace\nStrategy")
    _par("Model", display-name: "Model")
    _par("Device", display-name: "Device")

    _seq("User", "Strategy", comment: "Prepare experiment")
    _seq("Model", "Strategy", comment: [Prior model])

    _loop("Adaptive Loop", {
      _seq("Strategy", "Strategy", comment: [Estimate @eig])
      _seq("Strategy", "User", comment: "Select next experiments", dashed: true)

      _seq("User", "Device", comment: "Perform experiment", color: purple)
      _seq("Device", "User", comment: "Measurement data", dashed: true, color: purple)

      _seq("User", "Model", comment: [Characterize @pgm])
      _seq("Model", "Device", comment: "Benchmark process")
      _seq("Device", "User", comment: "Performance metrics", dashed: true)
      _seq("Model", "Strategy", comment: "Posterior model", color: blue)
      _seq("Strategy", "Strategy", comment: "Check termination", color: red)
    })
  }),
  caption: [The sequence diagram of the subspace strategy. ],
) <fig:subspace-flow>

3. *Alternative strategy*: Using an insight from subspace and random strategy, we also consider the alternative strategy which first select the control using subspace strategy, then use random for the next round. This strategy is motivated by the observation that subspace strategy can select a good first batches of the control to boost the performance of the model ahead of random strategy. However, after the long sequence, random sequence is able to catch up. Thus, it is interesting to consider the best of both worlds.

Since @bnn is capable of learning from small dataset, we focus on a few sample dataset compare to the `sample_size = 1000` used in the previous approaches. Here, we set the maximum sample size to `50` samples. With 10 sequential experiments, we collect 5 samples for each consecutive experiment.

For the subspace strategy, we illustrate how do we select the next experimental designs based on the @eig in @fig:eig-example. The plot is one of realization of the sequential experiments. We plot the histogram of selected control parameters across the strategies in @fig:control-histogram.

// #let (display, result, source, output, outputs, Cell, cell) = callisto.config(
//   nb: json("../code/chapter-5-strategies/note_0010_visualization_v2.ipynb"),
// )

#let (display, result, source, output, outputs, Cell, cell) = callisto.config(
  nb: json("../code/chapter-5-strategies-v2/note_0004_visualization_polars.ipynb"),
)

#figure(
    // image("placeholder.png"),
  output("subspace_selection"),
  caption: [
    An example of @eig for each control parameter throughout the sequential experiments and characterization. The red points are the selected control parameters at the current step. The gray points are the selected control parameters in the previous step.
  ],
) <fig:eig-example>

#figure(
  output("control_histogram"),
  // image("placeholder.png"),
  caption: [
    The histogram of the control parameters selected by the subspace strategy across 10 trajectories.
  ],
) <fig:control-histogram>

We repeatedly perform sequential experiments for 10 times to compare the performance statistics. We test on multiple control parameters which is basically the specialized testing dataset dicussed in @sec:data-variance-approach. The figure summarize the benchmark results comparing @jsd of the model train using dataset obtained from random and subspace strategy is presented in @fig:compare-strategies. 
We chose test control parameters $theta = {0.0^degree, 35.7^degree, 71.7^degree, 107.7^degree, 143.8^degree, 179.8^degree, 215.9^degree, 251.9^degree, 287.9^degree, 324.0^degree, 360.0^degree}$. 

#figure(
  output("compare_strategies"),
  // image("placeholder.png"),
  caption: [
    The @jsd and at difference sample size. We perform 10 realizations of random, subspace, and alternative strageties and plot the mean as a solid line. Each realization use different random seed.
  ],
) <fig:compare-strategies>


We can see from the result in @fig:compare-strategies that subspace strategy can consistancy choose control parameters (i.e. experimental design) that allows model to have performance better than a random strategy on the first few batches. From @fig:eig-example and @fig:control-histogram, we can observe that subspace strategy actively select control parameters near $theta = { 0^degree, 90^degree, 180^degree, 270^degree, 360^degree }$. These subspace dataset is a good starter dataset which allow model to start at a good initial model parameters. However, since the landscape of the @eig does not change significantly, the choice of experiment designs are concentrate at the same points. Consequently, there is no significant performance improvement with additional samples. On the other hand, the random strategy catch up and perform better after more data are collected. However, this strategies does not perfom well at the $theta = 0^degree$ compare to subspace which deliberately choose $theta = 0^degree$. So we consider the alternative strategy with subspace as a first in the sequence. We observe the improvement of the model performance conpare to the pure subspace strategy. The alternative strategy manage to perform well at $theta = 0^degree$ similar to subspace strategy, while we can improve the performance of the model with more samples. Although the improvement is not significant and the random strategy can produce the model that could catch up after more samples, we can see that subspace strategy allow for a better start than random strategy. Furthermore, with alternative strategy, we can see that there are a lot of improvement oppurtinities for more advance strategies. 

Apart from directly address the possibilities above, there are multiple oppurtinities of improvement. (1) Currently, for each characterization step with addition dataset of subspace model, we did not use posterior distribution from the previous step as a prior but start with the default distribution. So, it is also interesting to explore the use of posterior distribution in the previous step as a prior distribution of model parameters of the next step. (2) It is also interesting open question whether, the bias of the dataset due to @eig, does indeed allow model to learn faster with fewer data points or the model performance is better because there are less things in the dataset (non-uniformness of input feature) for model to learn. 



