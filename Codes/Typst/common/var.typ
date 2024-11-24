// EXO
#set page(height: 100pt)
#let c = counter("exo")
#let exo(tlt, txt) = block[
  #c.step()
  #rect(fill: red, radius: 5pt)[*Task #context c.display(): #tlt *] 
  #rect(fill: luma(221))[#txt]
]

// SOLUTION
#let solution(sol) = block[
  #rect(fill: olive)[#sol]
]

// PERSONALIZE FIGURE
/*
#let fig(imgLoc, imgCap) = figure(
  image(#str(<imgLoc>), width: 100%),
  caption: [#imgCap],
)
*/
//#fig["<< IMAGE_NAME.EXT >>"][<< CAPTION >>]

// TEST SCENARIO
#let test(tst) = [
#box(
	height: 25pt,
	image("attention.png", width: 10%)
)
#tst
]

/* TEMPLATE 
#exo[Title][Content.]

```julia
# WRITE YOUR CODE HERE
```

#test[Some test]
*/