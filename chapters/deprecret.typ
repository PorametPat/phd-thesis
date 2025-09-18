

#figure(
  diagram(
    debug: 0,
    spacing: 1.5em,
    // Left column
    node((0, 0), [`QubitInformation`], name: <qubit>),
    node((0, 1), [`ControlSequence`], name: <control>),
    node((0, 4), [`ExperimentConfig`], name: <config>),
    node((0, 5), [`ExperimentData`], name: <data>),

    // Center column
    node((1, 2), [Perform experiment], name: <exp>, fill: luma(98%), stroke: purple, corner-radius: 0.5em),
    node((1, 3), [`make_row`], name: <make>),
    node((1, 4), [Preprocess Dataframe], name: <preprocess>),

    // Right column
    node((2, 0), [Probabilistic Model], name: <model>),
    node((2, 1), [`estimate_eig`], name: <eig>),

    // Edges
    edge(<qubit>, <control>, "->"),
    edge(<control>, <config>, "->"),
    edge(<qubit>, "l,dddd,r", "->"),
    edge(<control>, "r,d", "->"),
    edge(<control>, "dd,r", "->"),
    edge(<exp>, <make>, "->"),
    edge(<make>, <preprocess>, "->"),
    edge(<config>, <data>, "->"),
    edge(<preprocess>, "d,l", "->"),

    edge(<model>, <eig>, "->"),
    edge(<eig>, "d,l", "-->", label: [optional], label-side: right, label-sep: 3em, label-anchor: "south"),
  ),
  caption: [
    In the experimental design and data aqusition phase, this diagram shows the order of data entity construction. For example, `ExperimentConfig` requires `QubitInformation` and `ControlSequence` during the initialization. The goal is to initialize `ExperimentData` to save and load dataset on local device for characterization phase.
  ],
  gap: 2em,
) <fig:data-module>



#grid(
  columns: 2,
  [
    #figure(
      diagram(
        debug: 0,
        node((0, 0), [Prepare experiment], name: <prepare>),
        edge("->"),
        node((0, 1), [Prior @pgm], name: <prior>),
        edge("->"),
        node((0, 2), [Estimate @eig \ and select next experiemnts], name: <eig>),
        edge("->"),
        node((0, 3), [Perform experiment], name: <exp>, fill: luma(98%), stroke: purple, corner-radius: 0.5em),
        edge("->"),
        node((0, 4), [Characterize @pgm], name: <characterize>),
        edge("->"),
        node((0, 5), [Benchmark @pgm], name: <benchmark>),
        edge("->"),
        node((0, 6), [Update prior \ with posterior], name: <update-posterior>),
        edge("->"),
        node((0, 7), [Terminate?], shape: fletcher.shapes.diamond, stroke: red, name: <terminate>),

        edge(<terminate>, "r,uuuuu,l", "->"),
        spacing: 2em,
      ),
      caption: [
        The chart of the subspace strategy.
      ],
      gap: 2em,
    ) <fig:subspace-flow>
  ],
  [
    #figure(
      diagram(
        debug: 0,
        node((0, 0), [Prepare experiment], name: <prepare>),
        edge("->"),
        node((0, 1), [Random new \ experiments], name: <eig>),
        edge("->"),
        node((0, 2), [Perform experiment], name: <exp>, fill: luma(98%), stroke: purple, corner-radius: 0.5em),
        edge("->"),
        node((0, 3), [Characterize @pgm], name: <characterize>),
        edge("->"),
        node((0, 4), [Benchmark @pgm], name: <benchmark>),
        edge("->"),
        node((0, 5), [Terminate?], shape: fletcher.shapes.diamond, stroke: red, name: <terminate>),

        edge(<terminate>, "r,uuuu,l", "->"),
        spacing: 2em,
      ),
      caption: [
        The chart of the subspace strategy.
      ],
      gap: 2em,
    ) <fig:random-flow>,
  ],
)