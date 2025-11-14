#import "../utils.typ": expval

= Alternative derivation
== MSE
From Bishop's Equation (1.87), the model prediction of input $x$ is denoted as $y(x)$, and the observation from experiment is a random variable $t$. The expectation of $(y(x) - t)^2$ with respect to the all possible input and output is
$
  bb(E)[L] = integral integral (y(x) - t)^2 p(x,t) d x d t.
$
From Bishop's Equation (1.89), the expected (true) value of the observation $t$ given $x$ is,
$
  h(x) = bb(E)[t|x] = integral t p(t|x) d t.
$ <eq:def-true>
Now, consider the expansion of square error,
$
  (y(x) - t)^2 & = ( y(x) - bb(E)[t|x] + bb(E)[t|x] - t )^2 \
               & = (y(x) - bb(E)[t|x])^2 + 2(y(x) - bb(E)[t|x])(bb(E)[t|x] - t) + (bb(E)[t|x] - t)^2,
$ <eq:exp-1>
then compute the expectation with respect to $t$ conditioned on a single input $x$,
$
  bb(E)_(t)[(y(x) - t)^2] = & integral (y(x) - t)^2 p(t|x) d t, \
                          = & (y(x) - bb(E)[t|x])^2 \
                            & + 2(y(x) - bb(E)[t|x])(
                                underbrace(integral (bb(E)[t|x] - t)p(t|x) d t, "A")
                              ) \
                            & + integral (bb(E)[t|x] - t)^2 p(t|x) d t.
$ <eq:exp-noise>
Note that $y(x)$ and $bb(E)[t|x]$ are independent to random variable $t$ by definition. From @eq:exp-1, consider the term A,
$
  integral (bb(E)[t|x] - t)p(t|x) d t = & bb(E)[t|x] - integral t p(t|x) d t = 0,
$
where we use the definition of expected value of $t$ in @eq:def-true. Therefore, the @eq:exp-noise is reduced to
$
  bb(E)_(t)[(y(x) - t)^2] = (y(x) - bb(E)[t|x])^2 + underbrace(integral (bb(E)[t|x] - t)^2 p(t|x) d t, "Conditional Variance").
$
Note that the last term is conditional variance of $t$ given value of $x$ by definition. So, we denote the conditional variance as
$
  "Var"[t|x] = integral (bb(E)[t|x] - t)^2 p(t|x) d t.
$
We then take the expectation over random variable $x$, and define $h(x) = bb(E)[t|x]$ as was done by Bishop,
$
  bb(E)_(x)[ bb(E)_(t)[L] ]= bb(E)_(x,t)[L] = & integral bb(E)_(t)[(y(x) - t)^2] p(x) d x \
                                            = & integral (y(x) - h(x))^2 p(x) d x + integral "Var"[t|x] p(x) d x,
$ <eq:337>
which resulting in the equation (3.37) in Bishop. Next, we know that the model is trained using a particular choice of dataset $cal(D)$, so in practice, the model depends in the dataset as $y(x; cal(D))$. Let's consider the following,
$
  (y(x; cal(D)) - h(x))^2 = & (y(x; cal(D)) - bb(E)_(cal(D))[y(x; cal(D))] + bb(E)_(cal(D))[y(x; cal(D))] - h(x) )^2 \
                          = & (y(x; cal(D)) - bb(E)_(cal(D))[y(x; cal(D))])^2 \
                            & + (bb(E)_(cal(D))[y(x; cal(D))] - h(x))^2 \
                            & + 2(y(x; cal(D)) - bb(E)_(cal(D))[y(x; cal(D))])(bb(E)_(cal(D))[y(x; cal(D))] - h(x)),
$
and the last term will be zero when we take the expectation over dataset $cal(D)$, we get,
$
  bb(E)_(cal(D))[ (y(x; cal(D)) - h(x))^2 ] &= underbrace((y(x; cal(D)) - bb(E)_(cal(D))[y(x; cal(D))])^2, "Bias"^2) \ &+ underbrace(bb(E)_(cal(D))[ (bb(E)_(cal(D))[y(x; cal(D))] - h(x))^2 ], "Variance").
$ <eq:bias-variance>
We now take the expectation over dataset $cal(D)$ on @eq:337,
$
  bb(E)_(cal(D)) [bb(E)_(x, t)[L]] =& integral bb(E)_(cal(D))[ (y(x) - h(x))^2] p(x) d x + bb(E)_(cal(D))[integral "Var"[t|x] p(x) d x].
$ <eq:94>
We then substitute the result from @eq:bias-variance and note that integral of conditional varaince over $x$ does not depend on the choice of dataset $cal(D)$,
$
  bb(E)_(cal(D), x, t|x)[L] = & integral "Bias"^2 p(x) d x \
                              & + integral "Variance" p(x) d x \
                              & + integral "Var"[t|x] p(x) d x,
$
where the expectations on the LHS are over the possible choice of dataset $cal(D)$, test input $x$, and the observation $t$ given the test input $x$. We can consider only particular value of input $x$ by taking derivative of $x$,
$
  bb(E)_(cal(D), t|x)[L] = & "Bias"^2 + "Variance" + "Var"[t|x], \
                       0 = & bb(E)_(cal(D), t|x)[L] - "Bias"^2 - "Variance" - "Var"[t|x],
$
which recover the result that agree with other references, except that the variance of the data is has now become the conditional variance instead.

Consider our context, the minimum value of expected $cal(L)_("SE[E]")$ is,
$
  bb(E)_(cal(D), t|x)[cal(L)_("SE[E]")] & >= 1/(18) sum_(rho_0, k) "Var"[ bb(E)[hat(O)_k]_(rho_0) | bold(Theta)] \
                                        & >= 2/(3n),
$
which is the usual result we expected.
== The performance inequality
From the Cauchy-Schwarz inequality,
$
  (sum_i v_i u_i)^2 <= (sum_i u_i^2)(sum_i v_i^2),
$
we define the following,
$
  u_i & = expval(hat(O))^("model")_(j, k, m) - expval(hat(O))^("exp")_(j, k, m), \
  v_i & = alpha_(j,k) beta_(m,j),
$
to get
$ sum_i v_i^2 = 6, $ and
$
  sum_i v_i u_i = 12 (macron(F)_("model") - macron(F)_("exp")).
$
Note that $expval(hat(O))^("exp")_(j, k, m)$ is the true (infinite-shot) expectation value of observable, where the finite-shot expectation value is denoted as $bb(E)^("exp")_(j, k, m)$ which is a random variable conditioned on input $bold(Theta)$.
From the $cal(L)_("SE[E]")$ definition, we reindex $j, k, m$ to $i$ and rewrite it in terms of $u_i$ as follows,
$
  cal(L)_("SE[E]") &= 1/18 sum_(j,k,m) (expval(hat(O))^("model")_(j, k, m) - bb(E)^("exp")_(j, k, m))^2 \
  &= 1/18 sum_(i) (expval(hat(O))^("model")_(i) - expval(hat(O))^("exp")_(i) + expval(hat(O))^("exp")_(i) - bb(E)^("exp")_(i))^2 \
  &= 1/18 sum_(i) (u_i + underbrace(expval(hat(O))^("exp")_(i) - bb(E)^("exp")_(i), epsilon_i))^2 \
  &= 1/18 sum_(i) ( u_i^2 - 2u_i epsilon_i + epsilon_i^2 ).
$
Rearranging above in the more useful form, we get,
$
  sum_i u_i^2 = 18 cal(L)_("SE[E]") + sum_i (2u_i epsilon_i - epsilon_i^2 ).
$
Then, we substitute the results back to CS inequality
$
  (macron(F)_("model") - macron(F)_("exp"))^2 <= 6/(12^2) (
    18 cal(L)_("SE[E]") + sum_i (2u_i epsilon_i - epsilon_i^2 )
  ).
$
Now we take the expectation over the distribtution of the observed (finite-shot) expectation value $y$ given input $x$, we simplify the cross term first as follows,
$
  bb(E)_(y|x)[sum_i 2u_i epsilon_i ] & = sum_i 2u_i bb(E)_(y|x)[ epsilon_i] \
                                     & = sum_i 2u_i bb(E)_(y|x)[ expval(hat(O))^("exp")_(i) - bb(E)^("exp")_(i) ] \
                                     & = sum_i 2u_i (expval(hat(O))^("exp")_(i) - bb(E)_(y|x)[bb(E)^("exp")_(i)] ) \
                                     & = 0,
$
where note that $u_i$ is constant with respect to the experimental noise. So the inequality becomes,
$
  (macron(F)_("model") - macron(F)_("exp"))^2 &<= 6/(12^2) (
    18 bb(E)_(y|x)[cal(L)_("SE[E]")] - sum_i underbrace(bb(E)_(y|x)[ epsilon_i^2], "conditional variance") ), \
  &<= 6/(12^2) (
    18 bb(E)_(y|x)[cal(L)_("SE[E]")] - sum_i "Var"[bb(E)^("exp")_(i) | bold(Theta)] ), \
  &<= 3/4 (bb(E)_(y|x)[cal(L)_("SE[E]")] - 2/(3n)).
$
we now compute the expectation over the choice of dataset $cal(D)$ to recover the definition of expected loss as follows,
$
  bb(E)_(cal(D))[(macron(F)_("model") - macron(F)_("exp"))^2] & <= 3/4 (bb(E)_(cal(D), y|x)[cal(L)_("SE[E]")] - 2/(3n)).
$
From the Jensen inequality, consider $|macron(F)_("model") - macron(F)_("exp")|$ as a random variable, we get,
$
  EE_(cal(D))[ abs(macron(F)_("model") - macron(F)_("exp")) ] <=& sqrt(3 / 4 (EE_(cal(D), y|x)[cal(L)_("SE[E]")] - 2/(3n)))
$

#pagebreak()

=== What we can learn from PML

Let consider the simple case of simulation where there is a detuning in X-axis and the control with gaussian envelope. As mentioned, due to the stochastic noise (e.g., colored noise), we expected that the distribution of the finite-shot expectation values may shift. Indeed, if we increase the strength of the noise, the mean of the distribution is shifted.
But at a glace, seems like the variance is approximately the same.

#grid(
  columns: (1fr, 1fr),
  inset: 10pt,
  figure(
    image("image.png"),
    caption: [The example of control signal. Blue represent the noiseless signal while the red represent the signal with noise.],
  ),
  figure(
    image("image-1.png"),
    caption: [The distribution of finite-shot estimate expectation value for both stochastic and deterministic cases.],
  ),
)

#let rvexpval = $expval(hat(O))_(rho_0)$

*So, let us analyze the problem by math, yet again.* Before we begin, let us setup the stage by the sequence diagram of how the data was generated. Given a control parameter, initial state $rho_0$ and observable $hat(O)$, we solve the Schrödinger equation for unitary operator $hat(U)_i$ with noise sampled from some source (e.g., colored noise). From the unitary operator, initial state, and the observable, we calculate the *intermediate* expectation value $rvexpval$. Note that $rvexpval$ is a derived random variable, we only assume that the expected value and variance of intermediate expectation value is given by
$ bb(E)[rvexpval] = mu_0, "Var"(rvexpval) = sigma_0^2 $
The value of $rvexpval$ is available to us since we do the simulation but in experiment, it cannot be directly observe. To mimic the experimental setting, we use $rvexpval$ to calculate the probability of observing eigenvalue $e_i = +1$ of observalbe $hat(O)$ and sample from it. Mathematically, eigenvalue $e_i$ is a random variable,
$
  e_i ~ "Bern"((1+rvexpval)/2 ),
$
depending on the value of $rvexpval$. We repeatedly sample $e_i$ for $n$ number of shots. Finally, with the ensemble of eigenvalues, we estimate the desired expectation value using the estimator defined as
$
  tilde(expval(hat(O)))_(rho_0) = 1/n sum_i^n e_i.
$

#import "@preview/chronos:0.2.1"

#chronos.diagram({
  import chronos: *

  // Define the participants in the sequence
  _par("Control", display-name: "Control")
  _par("Solver", display-name: "Solver")
  _par("Sampler", display-name: "Sampler")
  _par("Estimator", display-name: "Estimator")

  // Initial message from Control to Solver
  _sep("Initialization")
  _seq(
    "Control",
    "Solver",
    comment: [
      + control
      + initial state
      + observable
    ],
  )

  // Loop for n shots
  _sep("Simulation Loop")
  _loop("For a number of n shots", {
    _seq("Solver", "Sampler", comment: "sampling noise")
    _seq("Sampler", "Solver", comment: "add noise to signal", dashed: true)
    _seq("Solver", "Solver", comment: [solve Schrodinger equation
    ])
    _seq("Solver", "Sampler", comment: [calculate the intermediate \ expectation value $rvexpval$])
    _seq("Sampler", "Sampler", comment: [sample eigenvalue \ $e_i ~ "Bern"( (1+rvexpval)/2 )$])
    _note("over", [$rvexpval$ is a derived random variable \ from noise realization ], pos: "Solver")
  })


  // Final estimation step
  _sep("Estimation")
  _seq("Sampler", "Estimator", comment: [ensemble of $e_i$])
  _seq(
    "Estimator",
    "Estimator",
    comment: [
      estimate finite-shot \
      expectation value $tilde(expval(hat(O)))_(rho_0)$
    ],
  )
})

#pagebreak()

#import "@preview/gentle-clues:1.2.0": *

#goal[
  The questions are what are the (1) mean and (2) variance of the estimator $tilde(expval(hat(O)))_(rho_0)$.
]

The expected value of the estimator is 
$
  bb(E) [tilde(expval(hat(O)))_(rho_0)] = 1/n sum_i^n bb(E)[e_i].
$
Since $e_i$ is a random variable that depends on another random variable, we have to use the law of the total expectation as follows,
$
  bb(E)[e_i] = bb(E) [ bb(E) [ e_i | rvexpval ] ].
$
The expected value of eigenvalue $e_i$ given that the intermediate expectation value is fixed to $rvexpval$ is,
$
  bb(E) [e_i | rvexpval] =& (+1) ( (1+ rvexpval)/2 ) + (-1) ( (1- rvexpval)/2 ) \
  =& rvexpval.
$ <eq:cond-expected-eigen>
Thus, substituting back to the total expectation yield,
$
  bb(E)[e_i] = bb(E)[rvexpval] = mu_0,
$
by assumption of the intermediate expectation value. Therefore, the expected value of the estimator is 
$
  bb(E) [tilde(expval(hat(O)))_(rho_0)] = 1/n n mu_0 = mu_0.
$

Now, let consider the variance of the estimator which is
$
  "Var"(tilde(expval(hat(O)))_(rho_0)) &= "Var"( 1/n sum_i^n e_i ) \
  &= 1/n sum_i^n "Var"(e_i).
$
So, we have to calculate the variance of eigenvalue. Again, we have to calculate the variance using the law of total variance given by,
$
  "Var"(e_i) = bb(E) ["Var"(e_i|rvexpval)] + "Var"( bb(E)[e_i| rvexpval] ).
$
Consider the first term, the conditional variance is
$
  "Var"(e_i|rvexpval) &= bb(E) [ (e_i - bb(E)[e_i | rvexpval] )^2 | rvexpval ] \
  &= bb(E) [ (e_i - rvexpval)^2 | rvexpval ] \
  &= bb(E) [ e_i^2 - 2e_i rvexpval + rvexpval^2 | rvexpval] \
  &= bb(E)[e_i^2 | rvexpval] - rvexpval^2.
$
In the last equation, the conditional expected value of random variable given fixed value of itself should be itself. Now consider the first term of the last equation,
$
   bb(E)[e_i^2 | rvexpval] &= (+1)^2 ( (1+ rvexpval)/2 ) + (-1)^2 ( (1- rvexpval)/2 ) \
   &= 1.
$
So
$
  "Var"(e_i|rvexpval) = 1 - rvexpval^2.
$
The above equation is not surprising, since it is a variance of quantum observable given that the observable is Hermitian. Back to the first term of total variance, we have
$
  bb(E) ["Var"(e_i|rvexpval)] = 1 - bb(E) [rvexpval^2].
$
For the second term, we can use the result from @eq:cond-expected-eigen. Combining the first and second terms together, the total variance is
$
  "Var"(e_i) = 1 - rvexpval^2 + "Var"(rvexpval).
$
From the definition of variance of a random variable $X$, we have
$
  "Var"(X) = bb(E)[X^2] - (bb(E)[X])^2.
$
Thus, the total variance can be simplified as,
$
  "Var"(e_i) = 1 + (- bb(E)[rvexpval]^2) = 1 - mu_0^2
$
Finally, the variance of estimator is
$
  "Var"(tilde(expval(hat(O)))_(rho_0)) &= 1/n^2 n (1- mu_0^2) = 1/n (1-mu_0^2).
$
This is similar to the variance of estimator in the deterministic case where it is depends only the mean of the intermediate expectation value and independent of variance! 

= Rotating Frame

The unitary transformation,
$
  H -> U H U^(dagger) + i dot(U) U^(dagger)
$
Let us define $a_0 = 2 pi omega_q$ and $a_1 = 2 pi Omega$
$
  H(t) = a_0 / 2 hat(sigma)_(z) + a_(1) s(t) hat(sigma)_x
$

With the frame 
$
  U_(0) = exp( i a_(0)/2  hat(sigma)_z t )
$
We have
$
  dot(U)_(0) = diff / (diff t) U_(0) = i (a_(0))/2 hat(sigma)_z U_(0),
$
and 
$
  i dot(U)_(0) U_(0)^(dagger) = - (a_(0))/2 hat(sigma)_z.
$
And also
$
  U_(0) H U_(0)^(dagger) &= U_(0) (a_0 / 2 hat(sigma)_(z) + a_(1) s(t) hat(sigma)_x) U_(0)^(dagger) \
  &= a_(0)/2 sigma_z + a_1 s(t) U_(0) hat(sigma)_x U_(0)^(dagger)
$
Consider the term $U_(0) hat(sigma)_x U_(0)^(dagger)$ and define $a' = a_0/2 t$, we have,
$
  U_(0) hat(sigma)_x U_(0)^(dagger) =& (cos(a') I + i sin(a') hat(sigma)_z ) hat(sigma)_x (cos(a') I - i sin(a') hat(sigma)_z ) \
  =& cos^2(a') hat(sigma)_x - i cos(a') sin(a') hat(sigma)_x  hat(sigma)_z \
  &+ i sin(a') cos(a') hat(sigma)_z hat(sigma)_x + sin^2(a') hat(sigma)_z hat(sigma)_x  hat(sigma)_z
$
We use the fact that $hat(sigma)_y = i hat(sigma)_x hat(sigma)_z$ and $hat(sigma)_x hat(sigma)_z = - hat(sigma)_z hat(sigma)_x$, to simplify above to
$
  U_(0) hat(sigma)_x U_(0)^(dagger) =& cos^2(a') hat(sigma)_x - 2 cos(a') sin(a') hat(sigma)_y - sin^2(a') hat(sigma)_x \
  =& (cos^2(a') - sin^2(a')) hat(sigma)_x - 2 cos(a') sin(a') hat(sigma)_y.
$
Then with identities $cos(2x) = cos^2(x) - sin^2(x)$ and $sin(2x) = 2 cos(x) sin(x)$, we get
$
  U_(0) hat(sigma)_x U_(0)^(dagger) =& cos(2a') hat(sigma)_x - sin(2a') hat(sigma)_y
$
Finally, 
$
  H_("rot") = U_0 H U_0^(dagger) + i dot(U_0) U_0^(dagger) = a_1 s(t) (cos(a_0 t) hat(sigma)_x - sin(a_0 t) hat(sigma)_y)
$