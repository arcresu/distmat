# Changelog

## UNRELEASED

  * The matrix types previously required that `D: Copy` in order to do anything
    useful. With this release, the APIs have been converted to relax this bound
    and instead the iterators and accessors return values by reference.
    The exception is the rowwise iterators and the symmetric accessors of
    `DistMatrix`, which still require that `D: Copy + Default` due to a
    lifetime issue.
  * The matrix types gained by-reference iterators over the underlying data
    vectors (`iter()` and `iter_mut()`).

## distmat 0.3.0

  * `DistMatrix` and `SquareMatrix` can optionally store a list of labels
    corresponding to the underlying elements (e.g. taxa). There are new methods
    to get values by label. Some APIs changes as a result, for example the file
    parsers now just return a matrix rather than a `(labels, matrix)` pair.
  * `DistMatrix` and `SquareMatrix` can be constructed from iterators over
    pairs of labels associated with distances using `from_labelled_distances`.
