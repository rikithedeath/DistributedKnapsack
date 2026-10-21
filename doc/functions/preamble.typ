#import "@preview/algorithmic:1.0.7"
#import algorithmic: Line, algorithm-figure, style-algorithm
#import "@preview/codelst:2.0.2": sourcecode

// Helpers for the algorithmic package
#let reqline(body) = Line(text(body, style: "italic"))
#let ensureline(body) = Line(text(body, style: "italic"))
#let states(..parts) = Line(parts.pos().join(","))

// Stili di documento. Le regole set/show definite al top-level di un modulo
// importato NON si propagano al documento che lo importa: per attivarle il
// documento radice deve avvolgere il contenuto con questo template.
//
// main.typ:
//   #import "functions/preamble.typ": *
//   #show: style-doc
#let style-doc(body) = {
  show: style-algorithm.with(hlines: (
    grid.hline(stroke: 0.5pt + luma(150)),
    grid.hline(stroke: 0.5pt + luma(150)),
    grid.hline(stroke: 0.5pt + luma(150)),
  ))
  set document(
    title: "Knapsack distribuito: implementazione, comparazione e analisi",
    author: "Matteo Ielacqua",
  )
  set text(lang: "it", region: "it", size: 11pt)
  set page(paper: "a4", margin: (x: 1in, y: 1in))
  set par(justify: true)
  set heading(numbering: "1.1.1")
  // Figure numerate capitolo.figura: la funzione di numbering viene valutata
  // nel context della figura, quindi counter(heading) (e i riferimenti @)
  // risolvono al capitolo in cui la figura si trova.
  set figure(numbering: (..nums) => context numbering(
    "1.1",
    counter(heading).get().first(),
    ..nums,
  ))
  // I contatori delle figure sono per-kind: ognuno va resettato a ogni capitolo.
  show heading.where(level: 1): it => {
    counter(figure.where(kind: image)).update(0)
    counter(figure.where(kind: table)).update(0)
    counter(figure.where(kind: "listing")).update(0)
    counter(figure.where(kind: "algorithm")).update(0)
    it
  }
  // Supplement in inglese per figure e tabelle (i riferimenti nel testo
  // usano lo stesso supplement della caption).
  show figure.where(kind: image): set figure(supplement: [Figura])
  show figure.where(kind: table): set figure(supplement: [Tabella])
  show figure.where(kind: "listing"): set figure(supplement: [Listato])
  show figure.where(kind: "algorithm"): set figure(supplement: [Algorithm])
  // Il numero di un riferimento va risolto alla posizione della figura
  // referenziata, non a quella del riferimento: senza questa regola un
  // @fig citato in un altro capitolo mostrerebbe il capitolo sbagliato
  // (cfr. typst#2980). it.supplement non è risolto quando è auto, quindi
  // il supplement lo deduciamo dal kind della figura referenziata.
  show ref: it => {
    let el = it.element
    if el != none and el.func() == figure {
      let loc = el.location()
      let chapter = counter(heading).at(loc).first()
      let num = counter(figure.where(kind: el.kind)).at(loc).first()
      let supp = if el.kind == image { [Figure] } else if el.kind == table { [Table] } else if el.kind == "listing" {
        [Listato]
      } else if el.kind == "algorithm" { [Algorithm] } else { it.supplement }
      [#supp~#numbering("1.1", chapter, num)]
    } else {
      it
    }
  }
  body
}
