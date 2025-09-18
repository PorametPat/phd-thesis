#import "../utils.typ": style, tldr, tldr2

= Introduction

// #tldr2[Some content. #lorem(20)]

// #tldr[
//   Some content. #lorem(20)
// ]
//


== The problem

#lorem(100)

== How to read this thesis
#tldr[
  At the begining of some section, we use the #text(rgb("#A6E22E"))[*TL;DR*] blog to summarize the section. The aims is to convey the messages of the section to the reader as fast as possible for the impatient reader.
]

This thesis aims to explain and demonstrate the data-efficient characterization of quantum device framework packaging into `inspeqtor`. The design decisions and examples are discussed throughout the thesis.
Thus, the actual `python` code snippet is presented instead of a pseudo-code.



// #place(
//   dy: 10pt,
//   box(fill: rgb(style.black), inset: 1em, radius: 0.5em)[
//   #text(fill: rgb(style.green))[*TL;DR*] \
//   #text(fill: rgb(style.white))[#lorem(20)]
// ])
