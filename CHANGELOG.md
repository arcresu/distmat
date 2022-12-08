# Changelog

## distmat 0.3.0

  * `DistMatrix` and `SquareMatrix` can optionally store a list of labels
    corresponding to the underlying elements (e.g. taxa). There are new methods
    to get values by label. Some APIs changes as a result, for example the file
    parsers now just return a matrix rather than a `(labels, matrix)` pair.
  * `DistMatrix` and `SquareMatrix` can be constructed from iterators over
    pairs of labels associated with distances using `from_labelled_distances`.
