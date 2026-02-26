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

A quantum device is expected to be a large system of qubits. Complete characterization of the system in a classical representation is inefficient, if not impossible. For a large-scale characterization, only some system properties are of interest, and the information will be used to improve the system's performance. In the context of the characterization of the device into the predictive model, we only need to consider a subsystem of the device, i.e., only up to a two-qubit system. In this chapter, we will consider a single-qubit quantum device for demonstration purposes.

Consider a system of a qubit with frequency $omega_q$ with a time-dependent control signal as a function of control $s(bold(Theta),t)$ and drive strength $Omega$. An ideal Hamiltonian of superconducting qubits is defined as
$
  hat(H) = (2 pi omega_q) / 2 sigma_z + 2 pi Omega s(bold(Theta), t) sigma_x
$ <eq:transmon>
where $sigma_i in {X, Y, Z}$ is a Pauli matrix. The control signal is a function of the envelope function $h(bold(Theta), t)$ with driving frequency $omega_d$ typically set equal to the qubit frequency and phase $phi$ defined as follows,
$
 s(bold(Theta), t) = Re { h(bold(Theta), t) exp(i (2 pi omega_d t + phi)) }
$
In our numerical simulation, we test the Graybox performance by considering a scenario with a drift (or mismatch) in the qubit frequency.
The real value of the qubit frequency denoted by $omega'_q = 5.0005$ GHz is shifted from the value assumed by an experimenter, $omega_q = 5.0$ GHz, by $0.5$ MHz. This way, to simulate the real qubit's dynamics, we use @eq:transmon with $omega'_q$ for the qubit frequency, but $omega_d = 5.0$ GHz for the microwave drive, reflecting the experimenter's wrong assumption about the resonance condition. For the drive strength, we choose $Omega_d = 0.1 "GHz"$.

== Level of Characterization

We now demonstrate the uses of `inspeqtor` to characterize the quantum device. Here, we are going to use (1) @sgm, (2) @mcdgm, and (3) @pgm to characterize the same dataset from the same noisy device. The steps are listed as follows,
+ Define the control
+ Perform experiments and save the data to disk.
+ Load and split the dataset into training and test sets.
+ Model training/inference
+ Benchmark model with @jsd and $cal(L)_("MSE[E]")$

We now explain each step in detail. The first four steps are the same for each predictive model.

*The control:* Motivated by the noise model, we consider a single-parameter control in the Pauli X direction. The control envelope is a Gaussian shape with an area under the curve as a control parameter. The control is a rotation around the X-axis of the Bloch sphere, with the area of the Bloch sphere equal to the rotation angle. The control is the same as defined in @eq:gaussian-envelope. Thus, we use the `ControlSequence` for the experiment, as defined in the code snippet @code:gaussian-sequence.

*Experiments and data storing:* For the synthesis dataset, the inspector provides a convenient predefined function `sq.data.library.get_predefined_data_model_m1` to generate the noisy dataset given the system Hamiltonian, the control, and the other necessary information. We use the following code snippet to define the Hamiltonian of the noisy device.

We define the ideal Hamiltonian first, and will reuse it for the construction of the Whitebox. We then use the ideal Hamiltonian to define the total Hamiltonian. Now, we generate the dataset using the following code snippet.
#figure(
  caption: [],
  // kind: "code",
  // supplement: "Listing"
)[
  ```python

 data_model = sq.data.library.get_predefined_data_model_m1(
 detune, trotterization=trotterization, trotter_steps=trotter_steps
    )

 sample_size = 1000
 shots = 1000

 data_key = jax.random.key(0)

 exp_data, control_sequence, unitaries, noisy_simulator = (
 sq.data.library.generate_single_qubit_experimental_data(
          key=data_key,
          hamiltonian=data_model.total_hamiltonian,
          sample_size=sample_size,
          shots=shots,
          strategy=sq.physics.library.SimulationStrategy.SHOT,
          qubit_inforamtion=data_model.qubit_information,
          control_sequence=data_model.control_sequence,
          method=sq.physics.library.WhiteboxStrategy.TROTTER
          if trotterization
          else sq.physics.library.WhiteboxStrategy.ODE,
          trotter_steps=trotter_steps,
      )
  )

 sq.data.save_data_to_path(
      path=path, experiment_data=exp_data, control_sequence=control_sequence
  )
  ```
] <code:generate-dataset>
We can pass `total_hamiltonian` to the function as simply as shown in @code:generate-dataset:8. The `strategy` argument in @code:generate-dataset:11 is the argument to specify how to calculate expectation value. In this case, we use a finite-shot estimate strategy. We also specify the simulator to use the Trotterization method to calculate the evolution of the device in @code:generate-dataset:14. The function returns a simple `NamedTuple` containing the necessary data and functions for model training. In this case study, we want to share the synthetic data across characterization approaches, so we save the data using `sq. predefined.save_data_to_path`. The `path` is the name of the folder where the data will be stored; if the folder does not exist, the function will automatically create it, along with any necessary parent folders.

*Load and prepare dataset:*
The next step is to read the dataset from local storage. We use `sq. predefined.load_data_from_path` for this task as follows.
```python
loaded_data = sq.data.load_data_from_path(
 data_path,
    hamiltonian_spec=sq.physics.library.HamiltonianSpec(
        method=sq.physics.library.WhiteboxStrategy.TROTTER
        if TROTTERIZATION
        else sq.physics.library.WhiteboxStrategy.ODE,
        trotter_steps=TROTTER_STEPS,
    ),
)
```
The predefined function allows us to specify the Whitebox specifications, such as the solver method and the Hamiltonian. The `HamiltonianSpec` is a predefined data class that specifies the ideal Hamiltonian and the solver method to use. In this case, the Hamiltonian of Whitebox is the Hamiltonian defined in @eq:transmon in the rotating frame with respect to $omega_q$. We can now solve the Schrödinger equation for an ideal unitary operator and combine the results with those from Blackbox. To obtain a unitary operator from the Hamiltonian, we use the Trotterization method provided by inspeqtor. The Whitebox function is a function of the control parameter. It will be used to precompute unitary operators for the training and testing datasets and can also compute them on demand.

After loading the dataset, we split it into training and test sets. We can access the control parameters and their unitary and expectation values from `loaded_data`, which is a simple data holder. `inspeqtor` also provides a simple function, ' sq.utils. random_split', to randomly split the dataset into two sets.
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

To test our model's performance, we generate specialized test samples in addition to the test dataset. Specifically, we want to test whether our model can predict the distribution with the same statistics as those produced by the device. We now define a helper function to sample a single control parameter and use the device solver to repeatedly generate expectation values to form a distribution.
```python
def make_test_sample_fn(
    data_model: sq.utils.SyntheticDataModel,
    shots: int,
):
 ravel_fn, _ = sq.control.ravel_unravel_fn(data_model.control_sequence)

    def generate_test_distribution(key: jnp.ndarray, sample_size: int = 100):
 control_key, solver_key = jax.random.split(key)

 control_param = ravel_fn(data_model.control_sequence.sample_params(control_key))

 expvals = jax.vmap(sq.utils.shot_quantum_device, in_axes=(0, None, None, None))(
 jax.random.split(solver_key, sample_size),
 control_param.reshape(1, -1),
 data_model.solver,
 shots,
        )

        return control_param, expvals.squeeze(axis=1)

    return generate_test_distribution
```
We can use the function with the following code snippet.
```python
data_model = sq.data.library.get_predefined_data_model_m1(
    detune=DETUNE,
    trotterization=TROTTERIZATION,
    trotter_steps=TROTTER_STEPS,
)

generate_test_distribution = ml.make_test_sample_fn(data_model, shots=shots)
test_control_param, test_samples = generate_test_distribution(jax.random.key(0))
```
With `test_samples`, we can test our model performance by predicting a distribution of `test_control_param` and measure the @jsd of the distribution of the expectation values and calculate $cal(L)_("MSE[E]")$. Although @jsd is a direct measure of how close the distributions are to each other, we use $cal(L)_("MSE[E]")$ as an indicator of how close the means of predicted distributions are to the ideal values.

We previously discussed the mathematical construction of the Graybox model in @sec:graybox-characterization. Next, we will discuss the implementation of the Graybox model in the code, specifically for the study in this example. We already precomputed the Unitary operators associated with the control parameters using a predefined Whitebox. So, we now turn our attention to the Blackbox part. The architecture of Blackbox is present in @fig:blackbox-arch. At the high level, the Blackbox model performs the following operations.
+ We scale the control parameters by a factor of $2pi$ so that each feature has a maximum value of $1$, and then transform them using a 4th-order polynomial. We use the following code snippet.
  ```python
  def custom_feature_map(x: jnp.ndarray) -> jnp.ndarray:
      return sq.predefined.polynomial_feature_map(
 x / (2 * jnp.pi),
        degree=4
      )
  ```
 The `sq.predefined.polynomial_feature_map` is a function that maps $x -> [x, x^2, ..., x^n]$ to the $n$th order.
+ The Blackbox takes in augmented features to the  Shared layers. The Shared layers are a @mlp neural network with `relu` activation function.
+ The output is then sent to three Pauli layers, which are processed independently. Again, Pauli layers are an @mlp Neural Network
+ Finally, the Hermitian layers will return parameters that parametrized the Hermitian matrix in @eq:wo (See `sq.model.Wo_2_level_v3` for more details.)
With the Hermitian Matrices, we combine the Unitary operator from Whitebox to produce the expectation values using @eq:exp-wo. The technical differences among the models will be discussed, along with their training/inference and results.

#figure(image("../figures/fig_model_arch.svg"), caption: [
 The architecture of the Blackbox model and the Graybox model.
]) <fig:blackbox-arch>

=== Statistical approach
With the preparation in place, we now begin characterizing the quantum device with @sgm. From @fig:blackbox-arch, the model parameters (weights and biases) are point values. We construct the Blackbox with the predefined model provided by `inspeqtor` as follows.

// Get more functions preconfigured for this notebook
#let (display, result, source, output, outputs, Cell, cell) = callisto.config(
  nb: json("../code/chapter-5-random/note_0001_SGM.ipynb"),
)

#source(
  "model",
)

Here, the `sq.model.make_basic_blackbox_model` function generates a Blackbox implemented in `flax`, where the user can supplement custom activation functions to the final layers of the Hermitian layers. In the above example, we simply use the default activation functions. The following call with `hidden_sizes_1` and `hidden_sizes_2` initializes the `flax` model, which specifies the Shared and Pauli Layers. Both arguments accept a list of integers, where each integer is the width of a Dense layer in @mlp. In this case, both Shared and Pauli layers will consist of a single Desne layer with a width of 10.

We can now use a predefined function `train_model` to train the @sgm. In practice, we should monitor the model's performance as we train it. One way is to use the `callbacks` argument of the `train_model` function, which will call each callback in the list at the end of each epoch. In this case, we use a progress bar from `rich` to track model training progress.

#source(
  "training",
)

We save the model using the following code snippet.

#source("save_model")

We can load the model back using the following code snippet.

#source("load_model")

We plot the #mseeloss for the training and test datasets in @fig:sgm-loss.

#figure(
  result("plot_loss"),
  caption: [
 The training and testing loss of SGM.
  ],
) <fig:sgm-loss>

To properly benchmark the model performance, we test it on the test control parameters. The pre-generated `test_samples` has a shape of `(100, 18)`, i.e., 100 samples for each of 18 combinations of expectation value distributions. We then use the @sgm to predict the distribution given `test_control_param` using the following code snippet.

#source("predict_test_controls")

Now we can calculate @jsd between `test_prediction` and `test_samples` by using the following code snippet.

#source("calculate_jsd")

We also plot the differences in mean and variance between the test and prediction with @jsd in @fig:sgm-stats.
#figure(
  result("plot_metrics"),
  caption: [
    @jsd, mean and variance prediction of @sgm to the test control point.
  ],
) <fig:sgm-stats>

We can see that @sgm can predict results that are close to the empirical results. With an accurate mean value, it should be natural that the variance is accurate when the noise does not change the expectation value. The disadvantage of using @sgm is that we cannot target the model's parameters for BOED.

=== Monte-Carlo Dropout Approach

We can initialize @mcdgm using `sq.model.make_dropout_blackbox_model`, with the same arguments as for @sgm.

#let (display, result, source, output, outputs, Cell, cell) = callisto.config(
  nb: json("../code/chapter-5-random/note_0002_MCDGM.ipynb"),
)

#source("model")

However, `inspeqtor` did not provide a utility function for training @mcdgm out of the box, unlike @sgm. Instead, we can compose our training loop using low-level utility functions as follows.

#source("training")

The code snippet above is roughly similar to the algorithm used in `sq.optimize.train_model`. The main difference is that we have to pass the random key to the model during the forward pass. The training result can be saved using `ModelData` as well.

#source("save_model")

Again, we can load the model using the following code snippet, as in @sgm's case.

#source("load_model")

We use the model to test against `test_samples` and plot the results in @fig:mcdgm-stat. Compared with the statistics in @fig:sgm-stats, it appears that @mcdgm's predictions are not accurate. The calculated @jsd are close to the upper limit value at $ln(2)$. So the predicted distribution cannot be used. This is to be expected, since @mcdgm is not designed to capture the distribution of the observed data.

#figure(
  // image("../figures/fig_MCDGM_stats.svg"),
  result("plot_metrics"),
  caption: [
    @jsd, mean and variance prediction of @mcdgm to the test control point.
  ],
) <fig:mcdgm-stat>

=== Probabilistic approach <sec:pgm-approach>

For the probabilistic model inference, we need an additional step to transform the observable into raw binary measurement results.

#let (display, result, source, output, outputs, Cell, cell) = callisto.config(
  nb: json("../code/chapter-5-random/note_0003_PGM.ipynb"),
)

#source("transform_data")

We select `WoModel` as a base model, then convert it to @bnn using the following code snippet.

#source("model")

We then perform stochastic variational inference using functions exported by `inspeqtor`, which is integrated with `numpyro`.

#source("training")

Similarly, we can save, load, and prepare a model for prediction using the following code.

#source("save_model")
#source("load_model")

We plot the results in @fig:pgm-stat. The prediction performance of @pgm is similar to that of @sgm. This is not surprising since from @fig:sgm-loss, we can see that the model is trained to the optimal point. Consequently, the model accurately predicts the exact expectation values. Since the distribution of the finite-shot expectation value depends only on the expectation value, @jsd of the @sgm should be close to zero already.

#figure(
  result("plot_metrics"),
  caption: [
    @jsd, mean and variance prediction of @pgm to the test control point.
  ],
) <fig:pgm-stat>

== BOED approach <sec:boed-experiment>

Now, we are going to demonstrate the advantage of using a probabilistic model, which is the ability to use it for the calculation of @eig and use the information in device characterization. We are going to show how to use the model in sequential experiments and compare its performance to the approach without @eig.

The study setting is the same as the approaches discussed previously, with a modification to the data acquisition step. Previously, we uniformly sampled the control parameters to construct the dataset and experimented on the device in a single batch. However, in this study, we will perform experiments and acquire data in multiple batches sequentially. For each additional batch, we then perform characterization and benchmark the model. Next, we run the subsequent experiments on a new batch of data using the strategy (policy). The steps are repeatedly performed until some conditions are met. The strategies we are interested in are the following.
1. *Random strategy*: This is a strategy that does not utilize @eig for experimental design at all and chooses the experimental design randomly. This strategy is equivalent to the approach discussed in @sec:pgm-approach. We illustrate the strategy's flow in @fig:random-flow.

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
2. *Subspace strategy*: This strategy divides the control parameters space into multiple subspaces and then selects the designs that maximize @eig within the subspace. Note that this strategy represents one of the @boed's approaches. Since we are performing inference for a regression predictive model (@bnn), we want our model to generalize to unseen data without being overconfident. Thus, instead of choosing designs that maximize @eig globally, which would bias our dataset heavily, we chose designs that maximize @eig within their own subspace. We illustrate the strategy's flow in @fig:subspace-flow.

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

3. *Alternative strategy*: Using an insight from subspace and random strategy, we also consider the alternative strategy, which first selects the control using the subspace strategy, then uses random for the next round. This strategy is motivated by the observation that a subspace strategy can select a good initial batch of controls, thereby boosting the model's performance relative to a random strategy. However, after the long sequence, the random sequence can catch up. Thus, it is worth considering the best of both worlds.

Since @bnn can learn from a small dataset, we focus on a few sample datasets rather than the `sample_size = 1000` used in the previous approaches. Here, we set the maximum sample size to `50` samples. In 10 sequential experiments, we collect five samples each time.

For the subspace strategy, we illustrate how do we select the next experimental designs based on the @eig in @fig:eig-example. The plot is one of the realizations of the sequential experiments. We plot the histogram of selected control parameters across the strategies in @fig:control-histogram.

// #let (display, result, source, output, outputs, Cell, cell) = callisto.config(
//   nb: json("../code/chapter-5-strategies/note_0010_visualization_v2.ipynb"),
// )

#let (display, result, source, output, outputs, Cell, cell) = callisto.config(
  nb: json("../code/chapter-5-strategies-v3/note_0004_visualization_polars.ipynb"),
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

We repeatedly run 10 sequential experiments to compare performance statistics. We test across multiple control parameters, as discussed in @sec:data-variance-approach. The figure summarizes the benchmark results comparing the @jsd of the model trained on datasets generated by the random and subspace strategies, as presented in @fig:compare-strategies.
We chose test control parameters $theta = {0.0^degree, 35.7^degree, 71.7^degree, 107.7^degree, 143.8^degree, 179.8^degree, 215.9^degree, 251.9^degree, 287.9^degree, 324.0^degree, 360.0^degree}$.

#figure(
  output("compare_strategies"),
  // image("placeholder.png"),
  caption: [
 The @jsd and at the difference sample size. We perform 10 realizations of random, subspace, and alternative strategies and plot the mean as a solid line. Each realization uses a different random seed.
  ],
) <fig:compare-strategies>


We can see from the results in @fig:compare-strategies that the subspace strategy can consistently choose control parameters (i.e., experimental design) that allow the model to perform better than a random strategy on the first few batches. From @fig:eig-example and @fig:control-histogram, we can observe that the subspace strategy actively selects control parameters near $theta = { 0^degree, 90^degree, 180^degree, 270^degree, 360^degree }$. This subspace dataset is a good starter dataset that allows the model to start with good initial model parameters. However, since the landscape of the @eig does not change significantly, the choice of experimental designs is concentrated at the same points. Consequently, there is no significant improvement in performance with additional samples.
On the other hand, the random strategy catches up and performs better after more data are collected. However, this strategy does not perform well at the $theta = 0^degree$ compared to the subspace, which deliberately chooses $theta = 0^degree$. So we consider the subspace alternative as the first in the sequence. We observe improved model performance compared to the pure subspace strategy. The alternative strategy performs well at $theta = 0^degree$, similar to the subspace strategy, and we can improve the model's performance with more samples. Although the improvement is not significant, and the random strategy can produce a model that catches up with more samples, the subspace strategy offers a better start than the random strategy. Furthermore, with an alternative strategy, there are many opportunities for improvement.

Note that using an @eig -informed dataset may yield a highly biased model that does not generalize well to unseen data. Furthermore, we have observed a dataset that causes a model to perform worse (i.e., predict a very high @jsd). Please see the source code for the outlier. In particular, the following command produces an outliner,
```python
# experiment.py
trajectory(
    ["subspace", "random"],
    exp_id="exp_alter/0005",
    ikey=5, # The key that causes outliner
)
```
We want to note that if the @eig -informed strategy degrades model performance, we can abort the operation early. One possible preventive measure is to mix @eig informed designs with random designs. 

In addition to directly addressing the possibilities above, there are several opportunities for improvement. (1) Currently, for each characterization step with an additional dataset of the subspace model, we do not use the posterior distribution from the previous step as a prior but start with the default distribution. It is also interesting to explore using the posterior distribution from the previous step as a prior for the model parameters in the next step. (2) It is also an interesting open question whether the bias of the dataset due to @eig does indeed allow the model to learn faster with fewer data points, or the model performance is better because there are fewer things in the dataset (non-uniformness of input feature) for the model to learn.



