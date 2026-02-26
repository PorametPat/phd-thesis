#import "@preview/drafting:0.2.2": inline-note, margin-note
#let caution-rect = rect.with(inset: 1em, radius: 0.5em)
#import "../utils.typ": pkg
#import "@preview/gentle-clues:1.2.0": *

= Next Steps Forward

#inline-note(rect: caution-rect, fill: orange.lighten(80%))[
  Justify the place of this work in the field, go back to discuss works reviewed in @sec:before-thejourney-begins
]

#let yes = text(fill: green.darken(20%), weight: "bold")[Yes]
#let no = text(fill: red.darken(20%), weight: "bold")[No]
#let unknown = text(fill: gray)[Unknown]

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, center, left, center),
    column-gutter: 2pt,
    row-gutter: 2pt,
    stroke: (x, y) => if y == 0 { (bottom: 1.5pt) } else { 0.5pt },
    fill: (x, y) => if y == 0 { rgb("#e9ecef") } else if calc.odd(y) { rgb("#f8f9fa") } else { white },

    table.header()[*Tool/Method*][*Data-efficient Approach*][*Flexibility*],

    [#pkg[`inspeqtor`] \ #text(size: 0.85em, fill: gray)[(this thesis)]],
    // [#yes],
    [@eig, #cite(<fosterVariationalBayesianOptimal2019>, form: "normal")], [*High*],

    [#pkg[`qibocal`] #cite(<pasqualeOpensourceFrameworkPerform2024>, form: "normal")],
    // [#yes],
    [#no], [*Low* \ #text(size: 0.85em, fill: gray)[(predefined experiments)]],

    [#pkg[`C3`]  #cite(<wittlerIntegratedToolSet2021>, form: "normal") \ #text(
        size: 0.8em,
        style: "italic",
        fill: gray,
      )[(last updated Jan 2024)]],
    // [#yes],
    [#no], [#unknown],

    [#pkg[QCtrl Boulder Opal] #cite(<BoulderOpal>, form: "normal") \ #text(size: 0.8em, fill: gray)[(closed-source)]],
    // [#yes],
    [OBSID #cite(<staceOptimizedBayesianSystem2024>, form: "normal")], [*High*],

    [#cite(<fidererNeuralNetworkHeuristicsAdaptive2021>, form: "prose")],
    // [#no],
    [Bayes Risk], [#unknown],

    [#cite(<sarraDeepBayesianExperimental2023>, form: "prose")],
    // [#no],
    [One step @eig #cite(<fosterUnifiedStochasticGradient2020>, form: "normal")], [#unknown],
  ),
  caption: [
    An attempt to compare related works to `inspeqtor`.
  ],
)


== Conclusion

=== Contributions
My contributions, and what are difficults about my works
+ Custom Deep Probabilistic Neural Network that work with `estimate_eig` algorithm. `flax` fails when vectorization is needed.
+ Implement the `boed` functions to `numpyro` which is implement only in `pyro` previously.
+ Design that make moving forward easier.
  + Store data in a human-readable format.
  + Functional programming design, plug and play with ease.
  + `Jax` for automatic-differentiation and better control at randomness.
+ The use of BNN making engineering the architecture easier. We can inspect the distribution of weight to identify which area can be improve.
+ The contributions lead to the Resource-efficient framework as,
  + We can confidently reduce size and simplify model using estimated model optimality
  + We can deal with SPAM noise without additional dataset use to characterize Graybox model.

== Outlook for the future


=== Synchro Signal

#task[
  + Disucss how to use data variance for model synchronization signal.
  + Other approach related to probabilistic model.
  + Other approaches, recalibration system #cite(<mazurekTailoredQuantumDevice2025>)
]
