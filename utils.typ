#import "@preview/drafting:0.2.2": inline-note
#import "@preview/gentle-clues:1.2.0": *

#let style = (
  red: "#f43f5e",
  blue: "#6366f1",
  black: "#272822",
  green: "#A6E22E",
  white: "#F8F8F2",
)

#let spec(title: "Specification", ..args) = clue(
  ..args,
  accent-color: rgb(style.green),
  title: title,
  icon: emoji.clipboard,
)

// #let tldr

// Tags system
#let tag(body, fill: rgb("#f87171")) = box(
  text(body, fill: rgb("#ffffff")),
  fill: fill,
  inset: 0.3em,
  radius: 0.5em,
  baseline: 0.3em,
)

#let todoneedwrite(body) = inline-note(par-break: false, fill: rgb(style.red), stroke: rgb(style.red))[
  #text(fill: rgb(style.white))[need/write: #body]
]

#let todoneedcite = inline-note(par-break: false, fill: rgb(style.blue), stroke: rgb(style.blue))[
  #text(fill: rgb(style.white))[need/cite]
]

#let todocontinue(body) = inline-note(par-break: false, fill: rgb(style.blue), stroke: rgb(style.blue))[
  #text(fill: rgb(style.white))[Continue here: #body]
]

#let tldr(body) = box(fill: rgb(style.black), inset: 1em, radius: 0.5em, width: 100%)[
  #text(fill: rgb(style.green))[*TL;DR*] \
  #text(fill: rgb(style.white))[#body]
]

// Using gentle-clue version
#let tldr2(body) = clue(
  icon: emoji.flower.pink,
  title: [
    #text(
      fill: white,
    )[TL;DR]
  ],
  accent-color: gradient.linear(
    rgb(style.red),
    rgb(style.blue),
    dir: rtl,
  ),
  title-weight-delta: 300,
  radius: 0.5em,
)[#body]

#let implement-note(body) = box(
  stroke: rgb(style.red),
  inset: 0.5em,
  radius: 0.25em,
)[#body]

#let expectationvalue(..sink) = {
  let args = sink.pos() // array
  let expr = args.at(0, default: none)
  let func = args.at(1, default: none)

  if func == none {
    $lr(angle.l expr angle.r, size: #50%)$
  } else {
    $lr(angle.l func#h(0pt)mid(|)#h(0pt)expr#h(0pt)mid(|)#h(0pt)func angle.r)$
  }
}
#let expval = expectationvalue

#let _code(name, keyword, keyword-text, keyword-bg) = {
  [#box(
      fill: keyword-bg,
      inset: (x: 4pt, y: 2pt),
      radius: 3pt,
      baseline: 20%, // Adjust this to align with text baseline
      text(fill: keyword-text, weight: "medium", size: 0.9em)[#keyword],
    ) #text[#name]]
}

#let func(name) = _code(name, [func], rgb("#8259cd"), rgb("#b19cd937"))

#let modu(name) = _code(name, [modu], rgb("#4ad366"), rgb("#4ad36536"))

#let class(name) = _code(name, [class], rgb("#569cd6"), rgb("#569cd636"))

#let meth(name) = _code(name, [meth], rgb("#dcdcaa"), rgb("#dcdcaa36"))

#let pkg(name) = _code(name, [pkg], rgb("#ff6b6b"), rgb("#ff6b6b36"))


#let control = $bold(Theta)$