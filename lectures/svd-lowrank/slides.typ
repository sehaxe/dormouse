// A twenty-minute lecture in the OpenCode design language.
// Cream canvas, monospaced type, hairline rules, ASCII markers, one dark card.
// Build: typst compile slides.typ slides.pdf
// Every number comes from numbers.json, produced by src/lecture_svd.py on a
// real 102,678,943-parameter checkpoint.

#set page(width: 297mm, height: 167mm, margin: 0pt, fill: rgb("#FDFCFC"))
#set text(font: "Noto Sans Mono", size: 12pt, fill: rgb("#201D1D"), hyphenate: false)
#set par(leading: 0.62em, justify: false)
#show math.equation: set text(font: ("New Computer Modern Math",), size: 14pt)

#let ink = rgb("#201D1D")
#let bodyink = rgb("#424245")
#let mute = rgb("#646262")
#let stone = rgb("#6E6E73")
#let ash = rgb("#9A9898")
#let line = rgb("#E7E3E3")
#let accent = rgb("#007AFF")
#let danger = rgb("#FF3B30")
#let success = rgb("#30D158")
#let card = rgb("#F1EEEE")

#let TOTAL = 10

#let kick(t) = text(size: 8.5pt, fill: stone, upper(t), tracking: 0.1em)
#let hl(t) = text(fill: accent, t)
#let bad(t) = text(fill: danger, t)
#let ok(t) = text(fill: success, t)
#let note(t) = text(size: 8.5pt, fill: stone, t)

// One slide: number, claim, hairline, then the body.
#let slide(num, title, body) = page(fill: rgb("#FDFCFC"))[
  #pad(left: 20mm, right: 20mm, top: 13mm)[
    #text(size: 8.5pt, fill: stone)[#num / #TOTAL]
    #v(3mm)
    #text(size: 19pt, weight: 700, fill: ink)[#title]
    #v(2mm)
    #line(length: 100%, stroke: 0.5pt + line)
    #v(5mm)
    #body
  ]
  #place(bottom + left, dx: 20mm, dy: -7mm, text(size: 7.5pt, fill: ash)[the rank of a matrix, and what it costs])
  #place(bottom + right, dx: -20mm, dy: -7mm, text(size: 7.5pt, fill: ash)[#num / #TOTAL])
]

#let dark-slide(num, title, body) = page(fill: rgb("#201D1D"))[
  #set text(fill: rgb("#FDFCFC"))
  #pad(left: 20mm, right: 20mm, top: 13mm)[
    #text(size: 8.5pt, fill: rgb("#9A9898"))[#num / #TOTAL]
    #v(3mm)
    #text(size: 19pt, weight: 700)[#title]
    #v(2mm)
    #line(length: 100%, stroke: 0.5pt + rgb("#302C2C"))
    #v(5mm)
    #body
  ]
  #place(bottom + right, dx: -20mm, dy: -7mm, text(size: 7.5pt, fill: rgb("#9A9898"))[#num / #TOTAL])
]

#let bullet(body) = grid(columns: (7mm, 1fr), align: (left + top, left + top), [#text(fill: accent)[[+]]], body)

// ---------------------------------------------------------------- 01 title
#page(fill: rgb("#FDFCFC"))[
  #pad(left: 20mm, right: 20mm, top: 26mm)[
    #kick[linear algebra · twenty minutes]
    #v(9mm)
    #text(size: 33pt, weight: 700)[A matrix can be replaced\ by two thin ones]
    #v(7mm)
    #line(length: 32mm, stroke: 1.4pt + accent)
    #v(8mm)
    #text(size: 15pt, fill: bodyink)[
      Eckart–Young says how much you lose.\
      It does not say what you lose it _for_.
    ]
    #v(1fr)
  ]
  #place(bottom + left, dx: 20mm, dy: -14mm, text(size: 8.5pt, fill: stone)[
    measured on a 102,678,943-parameter network\ trained on 4.2 GB of text · four weight matrices · three seeds
  ])
]

// ------------------------------------------------------------- 02 geometry
#slide(2, "A weight matrix is a shape")[
  A matrix $W in RR^(m times n)$ is a linear map. The honest picture is
  geometric: it sends the unit sphere to an #hl[ellipsoid].

  #v(4mm)
  #align(center)[
    #cetz.canvas({
      import cetz.draw: *
      set-style(stroke: (paint: ink, thickness: 1pt))
      circle((0, 0), radius: 0.85)
      line((-1.2, 0), (1.2, 0), stroke: (paint: line, thickness: 0.6pt, dash: "dashed"))
      content((0, -1.15), text(size: 8pt, fill: mute)[input: every direction equal])
      line((1.55, 0), (2.95, 0), stroke: (paint: accent, thickness: 1.4pt), mark: (end: "stealth", fill: accent))
      content((2.25, 0.2), $W$, size: 10pt)
      // output ellipsoid, rotated
      set-style(stroke: (paint: ink, thickness: 1pt))
      rotate((5.4, 0), angle: 22deg, {
        circle((0, 0), radius: 1.45, stroke: none)
        line((-1.45 * 1, 0), (1.45, 0))
      })
      content((6.9, -1.15), text(size: 8pt, fill: mute)[output: an ellipsoid])
    })
  ]

  #v(2mm)
  #note[the semi-axes are the singular values — geometry, before any learning]
]

// ------------------------------------------------------------------ 03 svd
#slide(3, "The decomposition is a sentence")[
  #align(center)[
    $ W = U Sigma V^top, quad Sigma = op("diag")(sigma_1 >= sigma_2 >= dots.c >= 0) $
  ]
  #v(2mm)
  #table(
    columns: (26mm, 1fr),
    stroke: none,
    inset: (y: 2.2mm),
    table.hline(stroke: 0.5pt + line),
    text(fill: mute)[factor], text(fill: mute)[what it does],
    table.hline(stroke: 0.5pt + line),
    [$V^top$], [turns the input onto the principal axes],
    [$Sigma$], [stretches axis $i$ by $sigma_i$],
    [$U$], [turns the result into output space],
    table.hline(stroke: 0.5pt + line),
  )
  #v(4mm)
  #hl[Rotate, stretch, rotate.] The $sigma_i$ say _how much_ the map does in each
  direction — and the natural hope is that most of them say _almost nothing_.

  #v(2mm)
  #note[that hope is the entire economic case for low-rank compression]
]

// -------------------------------------------------------------- 04 theorem
#slide(4, "Eckart–Young, 1936")[
  Let $W_r = sum_(i <= r) sigma_i u_i v_i^top$. Then for every matrix $B$ with
  $op("rank") B <= r$,
  #v(2mm)
  #align(center)[
    $ norm(W - W_r)_F <= norm(W - B)_F, quad norm(W - W_r)_F = sqrt(sum_(i > r) sigma_i^2) $
  ]
  #v(3mm)
  #bullet[the truncated SVD is the #hl[best] rank-$r$ approximation — optimal, not heuristic]
  #v(2.5mm)
  #bullet[its error is #hl[known in advance]: the tail of the spectrum]
  #v(2.5mm)
  #bullet[no assumption on the data, the training, or the matrix being nice]
  #v(2mm)
  #note[the strongest theorem here, and the one most often misread]
]

// ---------------------------------------------------------------- 05 proof
#slide(5, "Why it is true: no subspace does better")[
  #grid(columns: (1fr, 0.62fr), gutter: 8mm)[
    Any rank-$r$ candidate $B$ has a null space of dimension $>= n - r$, so it
    contains a unit vector $z$ with $B z = 0$. On that $z$:
    #v(1mm)
    $ norm(W - B)_F >= norm((W - B) z) = norm(W z) $
    #v(1mm)
    Choosing $z$ inside $op("span")(v_(r+1), dots, v_n)$ gives
    $norm(W z) >= sigma_(r+1)$, and summing over the remaining directions
    reproduces exactly $sqrt(sigma_(r+1)^2 + dots.c)$.
  ][
    #pad(top: 2mm)[
      #text(size: 8.5pt, fill: mute)[
        a direction the candidate cannot see is energy it cannot reproduce
      ]
    ]
  ]
]

// -------------------------------------------------------------- 06 spectra
#slide(6, "Act two: what do trained matrices look like?")[
  #align(center)[#image("figures/spectrum.png", width: 88%)]
  #v(-1mm)
  #note[four matrices from one checkpoint · five orders of magnitude · not one of them snaps to zero]
]

// ------------------------------------------------------------- 07 surprise
#dark-slide(7, "The expectation fails")[
  #grid(columns: (1fr, 0.9fr), gutter: 6mm)[
    #block(fill: rgb("#FDFCFC"), inset: 2mm, radius: 1.5mm)[
      #image("figures/mass.png", width: 100%)
    ]
  ][
    #table(
      columns: (auto, auto),
      stroke: none,
      inset: (y: 2mm),
      table.hline(stroke: 0.5pt + rgb("#302C2C")),
      text(size: 8.5pt, fill: rgb("#9A9898"))[matrix], text(size: 8.5pt, fill: rgb("#9A9898"))[mass at $r=64$],
      table.hline(stroke: 0.5pt + rgb("#302C2C")),
      [moe.down], [41%],
      [q_proj], [45%],
      [mem_dense], [#bad[28%]],
      [expert $v$], [100%],
      table.hline(stroke: 0.5pt + rgb("#302C2C")),
    )
    #v(4mm)
    Trained matrices are #bad[not] nearly low-rank.
    #v(3mm)
    #text(fill: rgb("#9A9898"))[
      Truncating a trained layer to rank 64 throws away more than half of it by
      the theorem's own measure.
    ]
  ]
]

// ----------------------------------------------------------- 08 resolution
#slide(8, "Compression is not reparameterisation")[
  The expert matrix is rank 64 #hl[by construction]: we never store a
  $64 times 2730$ matrix, we store two thin ones whose product has rank $<= 64$.
  #v(1mm)
  #align(center)[
    $ W = U V^top, quad U in RR^(64 times 64), thin V in RR^(64 times 2730) $
  ]
  #v(3mm)
  #grid(columns: (1fr, 1fr), gutter: 8mm)[
    #kick[compression]
    #v(2mm)
    Given a fixed $W$, how well can rank $r$ imitate it?
    #v(2mm)
    #text(fill: danger)[answer: badly]
  ][
    #kick[reparameterisation]
    #v(2mm)
    Restrict the layer to rank $r$ before training, and let descent search
    inside that family.
    #v(2mm)
    #text(fill: success)[answer: almost free]
  ]
  #v(4mm)
  Eckart–Young bounds the first. It says #hl[nothing] about the second — and the
  second is what gets deployed.
]

// ---------------------------------------------------------------- 09 price
#slide(9, "The price, measured")[
  #grid(columns: (1fr, 0.85fr), gutter: 7mm)[
    #image("figures/rank_ladder.png", width: 100%)
  ][
    #kick[three seeds per arm · identical budget]
    #v(3mm)
    #table(
      columns: (auto, auto, auto),
      stroke: none,
      inset: (y: 2.2mm),
      table.hline(stroke: 0.5pt + line),
      text(size: 8.5pt, fill: mute)[rank], text(size: 8.5pt, fill: mute)[$Delta$ bits/byte], text(size: 8.5pt, fill: mute)[storage],
      table.hline(stroke: 0.5pt + line),
      [64], [baseline], [$1.00 times$],
      [32], [$+0.027$], [$0.50 times$],
      [16], [$+0.136$], [$0.25 times$],
      table.hline(stroke: 0.5pt + line),
    )
    #v(4mm)
    Halving the rank costs about $2%$ of quality; quartering it costs $10%$.
    #v(3mm)
    The spectrum predicts the _shape_ of this curve. Only the experiment gives
    its #hl[height].
  ]
]

// ------------------------------------------------------------- 10 takeaway
#slide(10, "Three things to take away")[
  #bullet[#hl[a matrix is a shape] — the singular values are its semi-axes, and the
    truncated SVD is the optimal rank-$r$ replacement, with the error known in advance]
  #v(4mm)
  #bullet[#hl[trained matrices are not low-rank] — measured: rank 64 keeps 28–45% of
    the mass; the naive compression story is false for real weights]
  #v(4mm)
  #bullet[#hl[compress ≠ reparameterise] — truncating is expensive, training inside the
    family is nearly free, and only the first is covered by the theorem]
  #v(6mm)
  #line(length: 100%, stroke: 0.5pt + line)
  #v(3mm)
  #note[
    exercise — take any layer, compute $sigma_i$, predict the loss after truncation
    with $sqrt(sum_(i > r) sigma_i^2)$, then measure.\
    the ratio between prediction and reality is the interesting number.
  ]
]
