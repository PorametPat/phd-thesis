#import "../utils.typ": style, tldr, tldr2, todoneedwrite, task, style

= Prelude

== The Challenges

Quantum devices with all-time peak performance are extremely important for achieving computational power beyond what other devices can provide. To gain this power, there are two main problems that we have to solve first. The first is to achieve a peak performance quantum device. The second is to maintain the peak performance. The first problem may be solved by using quantum optimal control to optimize for the control that achieves the peak performance. However, due to a temporal drifting of the noise, we have to perform a routine calibration, which solves the second problem. Yet, this protocol may induce another problem. Since calibration requires downtime, the device's availability for actual computation is consequently reduced. Thus, it is also extremely important to reduce the downtime for device calibration as much as possible. 

*What if we use error correction? You might ask...* Similar to the well-established computer system, a quantum computer will eventually use fault-tolerance quantum error correction (FTQEC) protocol for its logical operations instead of the raw physical operations. Nevertheless, FTQEC relies on an assumption that the underlying physical operations operate below a noise threshold. Thus, calibration of physical-level operations is a requirement even in the era of FTQEC. Moreover, since the physical operation quality may degrade over time due to the temporal noise, routine calibration is unavoidable. The need for routine calibration of physical-level operations is not limited to devices such as quantum computers but also applies to small quantum devices like quantum repeaters, which are essential for long-range communication using quantum protocols in the early generations.
// Cite quantum repeater architecture, distributed quantum computer, FTQEC protocol. 

Reducing maintenance downtime improves device availability. However, it's also important to minimize the amount of quantum data required for calibration, as high operational costs could be financially devastating for quantum devices. Thus, efficiently calibrating operations at the physical level solves the problem at its core.

*The next question is how to calibrate and characterize a quantum device efficiently?...* In fact, achieving this leads to several follow-up questions and challenges to be addressed. 

*Challenge number one*: There are multiple possible physical realizations (platform) of the quantum device. As of the time this thesis is written, the superconducting platform is prominent due to our ability to engineer the behavior of these artificial qubits to some extent. 
// Can cite IBM Q and google
Other choices include a trapped-ion platform, which offers a long lifetime, and a photonic platform, which is suitable for long-distance communication. Each platform has its own physical model and requires different infrastructure to operate and, consequently, maintain. #underline(stroke: 1.5pt + rgb(style.red))[Thus, calibration of each platform may leads to the needs of specialized software and hardware implementation]. Fortunately, it is possible to apply the theory of quantum optimal control to each platform, since they are quantum systems nonetheless. Regardless of the platform, we can use an appropriate abstraction level to calibrate for optimal control. 

*Challenge number two*: The abstraction requires a mathematical model of the device. Consequently, we have to estimate accurate model parameters from the system. The process of constructing a model from a device is called "characterization". Device characterization typically involves experiments on the actual hardware, from which we can extract model parameters. Once the model is constructed, we can perform the optimal control using the model in parallel since it is a local replica of the quantum device. Nonetheless, depending on the model of choice, #underline(stroke: 1.5pt + rgb(style.red))[the characterization process may also turn out to be another costly operation on the quantum device]. 
// Closed-loop approrach, but needs sequential experiments. 
Alternatively, we can directly calibrate a quantum device using a closed-loop approach, skipping characterization and allowing the optimizer to interact with the device rather than the replica model. Consequently, we must sequentially calibrate the device for each quantum operation required for the universal gateset. 

As a challenger, we aim to attack these challenges by proposing #underline(stroke: 1.5pt + rgb(style.blue))[a framework for quantum device characterization and calibration that focuses on utilizing experimental data efficiently]. Our framework uses abstractions of the quantum system that are compatible with Bayesian Optimal Experimental Design (BOED), a general framework @rainforthModernBayesianExperimental2024 @fosterVariationalBayesianOptimal2019 that allows us to optimize resource usage in device characterization. The device maintainer can start by using a predefined, platform-agnostic model (Graybox characterization method @youssryCharacterizationControlOpen2020 @youssryExperimentalGrayboxQuantum2024) to quickly characterize a subsystem of the device. Using the replica model, the maintainer can then calibrate for the numerical optimal control directly (gradient-based method also works!) or use the model as a testbed for a local experimental development. If the needs of platform-specific arise, the maintainer can implement the model that is compatible with our model abstraction to utilize the power of BOED. In this thesis, we present the framework and, together with it, create `inspeqtor`, a Python library that implements it. 

== How to read this thesis
#tldr[
 At the beginning of some sections, we use the #text(rgb("#A6E22E"))[*TL;DR*] blog to summarize the section. The aim is to convey the messages of the section to the reader as fast as possible for the impatient reader.
]

This thesis aims to explain and demonstrate the data-efficient characterization of quantum device framework packaging into `inspeqtor`. The design decisions and examples are discussed throughout the thesis.
Thus, the actual `python` code snippet is presented instead of pseudo-code.

This thesis is structured as follows,
+ *Chapter 2: Before the Journey Begins:* In this chapter, we review quantum mechanics, traditional quantum device characterization and calibration, and the previous works on efficiently performing characterization and calibration on quantum devices. 
+ *Chapter 3: The ways:* In preparation for efficiently characterizing the quantum device, we analyze the statistical properties of the (statistical) predictive model (a.k.a., replica model) to identify that it is possible to characterize an optimal model with realistic measurable quantities from a real device. We then construct a probabilistic predictive model from the statistical version, which is compatible with the BOED framework. 
+ *Chapter 4: `inspeqtor` Design and Implementation:* `inspeqtor` is both the name of the framework and the library implementing the framework. In this chapter, we discuss the core idea, design, and how we implement it as a Python library.
+ *Chapter 5: Data-efficient experiment:* Built on top of the solid foundation, we demonstrate the characterization experiments using `inspeqtor`. Using the qubit characterization task, we first apply the basic statistical Graybox method, then the probabilistic enhancement. We then demonstrate the framework's potential by performing data-efficient characterization.
+ *Chapter 6: Conclusion:* We finally conclude and discuss our results thus far, and outline possible directions in this chapter. 

// #place(
//   dy: 10pt,
//   box(fill: rgb(style.black), inset: 1em, radius: 0.5em)[
//   #text(fill: rgb(style.green))[*TL;DR*] \
//   #text(fill: rgb(style.white))[#lorem(20)]
// ])
