#import "@preview/cmarker:0.1.6"
#set document(title: "원형사 교본", author: "genkei-lab")
#set page(paper: "a4", margin: (x: 2cm, y: 2.2cm), numbering: "1",
  header: context { if counter(page).get().first() > 1 [#set text(8pt, fill: gray); 원형사 교본 — 튜토리얼 · 교재 · 커뮤니티 조사 #h(1fr) genkei-lab · 2026-09-24] })
#set text(lang: "ko", font: ("Noto Sans KR", "Malgun Gothic"), size: 9.5pt)
#set par(justify: true, leading: 0.75em)
#show heading.where(level: 1): it => { pagebreak(weak: true); block(below: 1em, text(20pt, weight: "bold", it.body)) }
#show heading.where(level: 2): it => { pagebreak(weak: true); block(above: 0.5em, below: 0.9em, text(14pt, weight: "bold", fill: rgb("#1f4e79"), it.body)) }
#show heading.where(level: 3): it => block(above: 1em, below: 0.6em, text(11pt, weight: "bold", it.body))
#show link: it => text(fill: rgb("#1f4e79"), it)
#show table: set text(8pt)
#set table(stroke: 0.4pt + luma(180), inset: 4pt)
#show table.cell.where(y: 0): set text(weight: "bold")
#show quote: it => block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%, it.body)

// 표지
#align(center + horizon)[
  #text(30pt, weight: "bold")[원형사 교본]
  #v(0.5em)
  #text(14pt)[피규어 원형사(原型師) 튜토리얼 · 교재 · 커뮤니티 조사]
  #v(2em)
  #text(10pt, fill: gray)[genkei-lab · 2026-09-24 · `/sc:research`]
  #v(1em)
  #block(width: 75%, text(9pt)[소체 → 겹 → 분할 → 다보 → 출력 준비 → 표면 처리 → 복제 → 양산 → 도색.
  AI 가 원형사 프로세스를 따라 하게 하기 위한 근거를 교과서 목차 · 실무 블로그 · 업계 자료에서 모았다.
  확실도 ●● 높음 · ● 중간 · ○ 낮음.])
]
#pagebreak()
#outline(title: [차례], depth: 2)

#cmarker.render(read("research_원형사교본_20260924.md"), h1-level: 1)
