#import "@preview/glossarium:0.5.6": gls, glspl, make-glossary, print-glossary, register-glossary
#import "@preview/hydra:0.6.1": hydra
#import "utils.typ": style
#import "glossaries.typ": entry-list
#import "@preview/drafting:0.2.2": note-outline

#import "@preview/equate:0.3.2": equate

#show: equate.with(breakable: true, sub-numbering: true)
#set math.equation(numbering: "(1.1)")

#show: make-glossary
#register-glossary(entry-list)

// Customize the render behavior
// Make link looks good
#show link: set text(fill: rgb(style.purple))
#show ref: it => {
  text(it, fill: rgb(style.purple))
}

#show heading: set block(above: 1.5em, below: 1.5em)

#set par(leading: 0.75em, justify: true, first-line-indent: (
  amount: 1.5em,
))
#set text(
  // font: "New Computer Modern",
  size: 12pt,
)
#show heading.where(level: 1): it => {
  if it.body not in ([Contents], [Bibliography], [Acknowledgement], [Glossary], [List of Todos], [Appendix]) {
    pagebreak(weak: true)
    block(width: 100%)[
      Chapter #counter(heading).display() #it.body
    ]
  } else {
    it
  }
}
// Set the alignment of caption to the left
#show figure.caption: set align(left)

#set page(width: 21.59cm, height: 27.94cm)
// Set the page number to roman numbering and the margin for printing.
#set page(numbering: "I", margin: (inside: 1.5in, outside: 1in, y: 1in))

// Set the indentation of the ordered list.
#set enum(
  indent: 1em,
  body-indent: 1em,
)

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
// #set raw(theme: "theme.tmTheme")

#show: codly-init.with()
// #show raw.where(block: true): it => block(fill: rgb("#1d2433"), text(fill: rgb("#F8F8F2"), it))
#codly(zebra-fill: none, languages: (
  python: (
    name: [ python],
    color: white,
    icon: "🌸",
  ),
))

// Override math
#import "@preview/quick-maths:0.2.1": shorthands
#show: shorthands.with(
  ($<=$, sym.lt.eq.slant),
  ($>=$, sym.gt.eq.slant),
)

// Show the table of contents
#outline()
#pagebreak()
// Include the preface pages.
#include "chapters/acknowledgement.typ"

// Set the page number to numeric numbering.
#set page(
  numbering: "1",
  header: context {
    if calc.odd(here().page()) {
      counter(page).display()
      h(1fr)
      emph(hydra(1))
      // align(right, emph(hydra(1)))
      // align(left, counter(page).display())
    } else {
      emph(hydra(3))
      h(1fr)
      counter(page).display()
    }
    line(length: 100%)
  },
  footer: context {
    line(length: 100%)
    text()[_Poramet Pathumsoot's PhD Thesis_]
    h(1fr)
    let (current,) = counter(page).get()
    let (total,) = counter(page).final()
    let percent = current / total
    let n-outline = query(
      heading.where(
        level: 1,
        body: [List of Todos],
      ),
    ).at(0)
    link(n-outline.location())[TODOs 🚀]
    h(0.5cm)
    box(rect(height: 10pt, radius: 1em, stroke: none, fill: rgb("#e5e7eb"), width: 100pt, inset: 0em, outset: 0em)[
      #rect(
        height: 10pt,
        radius: 1em,
        stroke: none,
        fill: gradient.linear(
          rgb(style.red),
          rgb(style.blue),
          dir: rtl,
        ),
        width: percent * 100pt,
        inset: 0.1em,
        outset: 0em,
      )
    ])
  },
)

#note-outline(title: "List of Todos") <note>

// Setup appendix
#let appendix(body) = {
  set heading(numbering: "A", supplement: [Appendix])
  counter(heading).update(0)
  body
}

// Reset numbering frpom roman numbering.
#counter(page).update(1)
// Setup the heading and equation numbering
#set heading(numbering: "1.")
// #set math.equation(numbering: "(1.)")
// Include the body of the thesis
#include "chapters/introduction.typ"
#include "chapters/chapter_2.typ"
#include "chapters/chapter_3.typ"
#include "chapters/chapter_4.typ"
#include "chapters/chapter_5.typ" // <--- Layout did not converge within 5 attempts warning becasue `callisto`. 
#include "chapters/chapter_6.typ"

#show: appendix
#include "chapters/appendix.typ"

// Bibliography
#pagebreak()
#bibliography("references.bib", title: "Bibliography", style: "american-physics-society")

// Appendices and glossary
#pagebreak()
#heading(numbering: none, level: 1)[Glossary]
#print-glossary(entry-list)
