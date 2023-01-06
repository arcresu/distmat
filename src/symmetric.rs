//! A symmetric matrix stored as the lower triangle.

use std::io::Read;
use std::ops::Sub;
use std::path::Path;
use std::slice;

use crate::builder::{DataError, DistBuilder};
use crate::formats::{
    parse_phylip_lt, parse_tabular_lt, PhylipDialect, PhylipError, Separator, TabularError,
};
pub use crate::square::Labels;
use crate::{open_file, AbsDiff};

/// Stores the lower triangle of a matrix.
///
/// This type stores only `n * (n-1) / 2` elements instead of the full `n * n` elements, so is more
/// memory efficient at the expense of a less straightforward mapping between matrix coordinates
/// and storage index.
///
/// Elements of the matrix can be accessed either using coordinates in the lower
/// triangle with [`get`](DistMatrix::get), or any non-diagonal coordinate with
/// [`get_symmetric`](DistMatrix::get_symmetric).
///
/// The rowwise iterators are available when `D: Copy + Default` as they yield copies of values.
#[derive(PartialEq, Eq, Clone)]
#[cfg_attr(test, derive(Debug))]
pub struct DistMatrix<D> {
    pub(crate) data: Vec<D>,
    pub(crate) size: usize,
    pub(crate) labels: Option<Vec<String>>,
}

/// Iterator over matrix rows; see [`DistMatrix::iter_rows`].
pub struct Rows<'a, D> {
    matrix: &'a DistMatrix<D>,
    row: std::ops::Range<usize>,
}

/// Copying iterator over a row of the matrix; see [`DistMatrix::iter_rows`].
pub struct Row<'a, D> {
    matrix: &'a DistMatrix<D>,
    row: usize,
    span: Span<'a, D>,
}

/// Iterator over matrix coordinates; see [`DistMatrix::iter_coords`].
pub struct Coordinates {
    i: usize,
    j: usize,
    n_m1: usize,
}

pub type Iter<'a, D> = std::slice::Iter<'a, D>;
pub type IterMut<'a, D> = std::slice::IterMut<'a, D>;

/// Build a matrix from its values.
///
/// The length of the source iterator should be `n * (n - 1) / 2` for some `n: usize`.
/// The diagonal and upper triangle are omitted, and the remaining lower triangle should be
/// provided in column major order.
///
/// Panics if the iterator contains the wrong number of entries.
impl<D> FromIterator<D> for DistMatrix<D> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = D>>(iter: I) -> Self {
        let data: Vec<D> = iter.into_iter().collect();
        data.into()
    }
}

/// Build a matrix from its values.
///
/// The length of the source vector should be `n * (n - 1) / 2` for some `n: usize`.
/// The diagonal and upper triangle are omitted, and the remaining lower triangle should be
/// provided in column major order.
///
/// Panics if the vector contains the wrong number of entries.
impl<D> From<Vec<D>> for DistMatrix<D> {
    fn from(data: Vec<D>) -> Self {
        let size = n_items(data.len());
        assert_eq!(n_entries(size), data.len());
        DistMatrix {
            data,
            size,
            labels: None,
        }
    }
}

impl<D> DistMatrix<D> {
    /// Build a distance matrix from pairwise distances of points where distance is taken to be the
    /// absolute difference as defined by the [`AbsDiff`] trait.
    ///
    /// See [`from_pw_distances_with`](DistMatrix::from_pw_distances_with) for a version that
    /// allows for an arbitrary distance measure.
    pub fn from_pw_distances<'a, T>(points: &'a [T]) -> Self
    where
        &'a T: Sub<&'a T, Output = D> + AbsDiff<&'a T>,
    {
        Coordinates::new(points.len())
            .map(|(i, j)| {
                let t1 = &points[i];
                let t2 = &points[j];
                t1.abs_diff(t2)
            })
            .collect()
    }

    /// Build a distance matrix from pairwise distances of points where distance is measured using
    /// the provided closure.
    pub fn from_pw_distances_with<T, F: FnMut(&T, &T) -> D>(points: &[T], mut dist_fn: F) -> Self {
        Coordinates::new(points.len())
            .map(|(i, j)| {
                let t1 = &points[i];
                let t2 = &points[j];
                dist_fn(t1, t2)
            })
            .collect()
    }

    /// Build a matrix from an iterator of labelled distances.
    ///
    /// The iterator can be in any order so long as by the end of iteration there are exactly the
    /// correct entries. An error will be returned if any entry is duplicated, or if there are the
    /// wrong number of entries.
    pub fn from_labelled_distances<S, I>(iter: I) -> Result<DistMatrix<D>, DataError>
    where
        S: AsRef<str>,
        I: IntoIterator<Item = (S, S, D)>,
    {
        let builder: DistBuilder<D> = std::iter::FromIterator::from_iter(iter);
        builder.try_into()
    }

    /// Retrieve an element by reference from the lower triangle of the distance matrix.
    ///
    /// Returns a value if `row < col < n` where `n` is the number of rows in the distance matrix,
    /// otherwise returns `None`.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&D> {
        if row >= self.size || col >= self.size || row >= col {
            return None;
        }
        Some(&self.data[index_for(self.size, row, col)])
    }

    /// Retrieve an element by mutable reference from the lower triangle of the distance matrix.
    ///
    /// Returns a value if `row < col < n` where `n` is the number of rows in the distance matrix,
    /// otherwise returns `None`.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut D> {
        if row >= self.size || col >= self.size || row >= col {
            return None;
        }
        Some(&mut self.data[index_for(self.size, row, col)])
    }

    /// Retrieve an element by reference from the distance matrix.
    ///
    /// If either of the indices exceeds the size of the distance matrix or `row == col`, then
    /// `None` is returned.
    pub fn get_symmetric(&self, row: usize, col: usize) -> Option<&D> {
        if row >= self.size || col >= self.size {
            return None;
        }
        Some(&self.data[index_for(self.size, row.min(col), row.max(col))])
    }

    #[inline]
    fn label_to_index<S: AsRef<str>>(&self, label: S) -> Option<usize> {
        self.labels
            .as_ref()?
            .into_iter()
            .position(|l| l == label.as_ref())
    }

    /// Retrieve an element by reference from the lower triangle of the distance matrix.
    ///
    /// Equivalent to [`get`](DistMatrix::get) after converting the labels to indices, except that
    /// `None` will additionally be returned if either `row` or `col` are not known labels.
    pub fn get_by_name<S: AsRef<str>>(&self, row: S, col: S) -> Option<&D> {
        let row = self.label_to_index(row)?;
        let col = self.label_to_index(col)?;
        self.get(row, col)
    }

    /// Retrieve an element by reference from the distance matrix.
    ///
    /// Equivalent to [`get_symmetric`](DistMatrix::get_symmetric) after converting the labels to
    /// indices, except that `None` will additionally be returned if either `row` or `col` are not
    /// known labels.
    pub fn get_symmetric_by_name<S: AsRef<str>>(&self, row: S, col: S) -> Option<&D> {
        let row = self.label_to_index(row)?;
        let col = self.label_to_index(col)?;
        self.get_symmetric(row, col)
    }

    /// Decompose into the stored labels and data.
    ///
    /// The order is compatible with R's `dist` objects.
    #[inline]
    pub fn into_inner(self) -> (Option<Vec<String>>, Vec<D>) {
        (self.labels, self.data)
    }

    /// Iterate by reference over all values in the matrix.
    ///
    /// The order corresponds to that of [`iter_coords`](DistMatrix::iter_coords).
    pub fn iter(&self) -> Iter<D> {
        self.data.iter()
    }

    /// Iterate by mutable reference over all values in the matrix.
    ///
    /// The order corresponds to that of [`iter_coords`](DistMatrix::iter_coords).
    pub fn iter_mut(&mut self) -> IterMut<D> {
        self.data.iter_mut()
    }

    /// Iterate over coordinates as `(row, column)`.
    ///
    /// This does not include the diagonal or upper triangle entries.
    #[inline]
    pub fn iter_coords(&self) -> Coordinates {
        Coordinates::new(self.size)
    }

    /// Iterator over labels for the underlying elements.
    ///
    /// If no labels are configured for this matrix, the iterator will be empty.
    /// See [`Labels::has_labels`] and [`set_labels`](DistMatrix::set_labels).
    #[inline]
    pub fn iter_labels(&self) -> Labels {
        Labels(self.labels.as_ref().map(|labs| labs.iter()))
    }

    /// Replace the element labels.
    ///
    /// Panics if the number of labels is not the same as `self.size()`.
    #[inline]
    pub fn set_labels(&mut self, new_labels: Vec<String>) {
        assert_eq!(new_labels.len(), self.size);
        self.labels = Some(new_labels);
    }

    /// Remove the element labels.
    #[inline]
    pub fn clear_labels(&mut self) {
        self.labels = None;
    }

    /// Convert distances using the provided function.
    #[inline]
    pub fn map_with<T, F: FnMut(&D) -> T>(&self, mapper: F) -> DistMatrix<T> {
        DistMatrix {
            data: self.data.iter().map(mapper).collect(),
            size: self.size,
            labels: self.labels.clone(),
        }
    }

    /// The number of rows/cols in the full matrix.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl DistMatrix<f32> {
    /// Parse a distance matrix from `reader` in PHYLIP lower triangle format.
    ///
    /// See [`formats`](crate::formats) for details.
    #[inline]
    pub fn from_phylip<R: Read>(
        reader: R,
        dialect: PhylipDialect,
    ) -> Result<DistMatrix<f32>, PhylipError> {
        parse_phylip_lt(reader, dialect)
    }

    /// Load a distance matrix from the file at `path` in PHYLIP lower triangle format.
    ///
    /// The file can optionally be compressed with `gzip`.
    ///
    /// See [`formats`](crate::formats) for details.
    #[inline]
    pub fn from_phylip_file<P: AsRef<Path>>(
        path: P,
        dialect: PhylipDialect,
    ) -> Result<DistMatrix<f32>, PhylipError> {
        let reader = open_file(path)?;
        parse_phylip_lt(reader, dialect)
    }

    /// Parse a distance matrix from `reader` in tabular lower triangle format.
    ///
    /// Only the long shape is supported.
    ///
    /// See [`formats`](crate::formats) for details.
    #[inline]
    pub fn from_tabular<R: Read>(
        reader: R,
        separator: Separator,
    ) -> Result<DistMatrix<u32>, TabularError> {
        parse_tabular_lt(reader, separator)
    }

    /// Load a distance matrix from the file at `path` in tabular lower triangle format.
    ///
    /// The file can optionally be compressed with `gzip`.
    /// Only the long shape is supported.
    ///
    /// See [`formats`](crate::formats) for details.
    #[inline]
    pub fn from_tabular_file<P: AsRef<Path>>(
        path: P,
        separator: Separator,
    ) -> Result<DistMatrix<u32>, TabularError> {
        let reader = open_file(path)?;
        parse_tabular_lt(reader, separator)
    }
}

impl<D: Copy> DistMatrix<&D> {
    /// Maps a `DistMatrix<&D>` to a `DistMatrix<D>` by copying values.
    #[inline]
    pub fn copied(self) -> DistMatrix<D> {
        DistMatrix {
            data: self.data.into_iter().copied().collect(),
            size: self.size,
            labels: self.labels,
        }
    }
}

impl<D: Copy> DistMatrix<D> {
    /// Copy a subset of the distance matrix corresponding to the specified row/column positions.
    ///
    /// Panics if any of the positions are out of range.
    pub fn subset(&self, positions: &[usize]) -> Self {
        if positions.is_empty() {
            return DistMatrix {
                data: Vec::new(),
                size: 1,
                labels: None,
            };
        }

        let mut positions = positions.to_vec();
        positions.sort_unstable();
        positions.dedup();
        assert!(*positions.last().unwrap() < self.size);

        let data: Vec<D> = Coordinates::new(self.size)
            .zip(&self.data)
            .filter_map(|((i, j), &v)| {
                let keep =
                    positions.binary_search(&i).is_ok() && positions.binary_search(&j).is_ok();
                keep.then_some(v)
            })
            .collect();
        let size = positions.len();
        assert_eq!(size, n_items(data.len()));

        let labels = self.labels.as_ref().map(|labs| {
            let mut new_labs = Vec::with_capacity(positions.len());
            for pos in positions {
                new_labs.push(labs[pos].clone());
            }
            new_labs
        });

        DistMatrix { data, size, labels }
    }
}

impl<D: Copy + Default> DistMatrix<D> {
    /// Iterate over rows of the distance matrix.
    #[inline]
    pub fn iter_rows(&self) -> Rows<D> {
        Rows {
            matrix: self,
            row: 0..self.size,
        }
    }

    /// Iterate over columns of the distance matrix.
    ///
    /// Since this is a symmetric matrix, this is the same as [`iter_rows`](DistMatrix::iter_rows).
    #[inline]
    pub fn iter_cols(&self) -> Rows<D> {
        self.iter_rows()
    }

    /// Iterator over pairwise distances from the specified point to all points in order, including
    /// itself.
    #[inline]
    pub fn iter_from_point(&self, idx: usize) -> Row<D> {
        Row {
            matrix: self,
            row: idx,
            span: Span::Initial,
        }
    }

    /// Iterator over pairwise distances from all points (including itself) to the specified point
    /// in order.
    ///
    /// Since this is a symmetric matrix, this is the same as
    /// [`iter_from_point`](DistMatrix::iter_from_point).
    #[inline]
    pub fn iter_to_point(&self, idx: usize) -> Row<D> {
        self.iter_from_point(idx)
    }
}

impl<'a, D> Iterator for Rows<'a, D> {
    type Item = Row<'a, D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.row.next().map(|row| Row {
            matrix: self.matrix,
            row,
            span: Span::Initial,
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.matrix.size, Some(self.matrix.size))
    }
}

impl<'a, D> ExactSizeIterator for Rows<'a, D> {}

impl<'a, D> DoubleEndedIterator for Rows<'a, D> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.row.next_back().map(|row| Row {
            matrix: self.matrix,
            row,
            span: Span::Initial,
        })
    }
}

#[derive(Debug)]
enum Span<'a, D> {
    Initial,
    ShrinkingStride(usize, usize),
    Diagonal,
    Contig(slice::Iter<'a, D>),
}

impl<'a, D: Copy + Default> Iterator for Row<'a, D> {
    type Item = D;

    fn next(&mut self) -> Option<Self::Item> {
        match self.span {
            Span::Initial => {
                if self.row == 0 {
                    self.span = Span::Diagonal;
                    return self.next();
                } else if self.row == 1 {
                    self.span = Span::Diagonal;
                } else {
                    self.span = Span::ShrinkingStride(self.row - 1, self.matrix.size - 2);
                };
                Some(self.matrix.data[self.row - 1])
            }
            Span::ShrinkingStride(cursor, stride) => {
                let cursor = cursor + stride;
                self.span = if stride <= self.matrix.size - self.row {
                    Span::Diagonal
                } else {
                    Span::ShrinkingStride(cursor, stride - 1)
                };
                Some(self.matrix.data[cursor])
            }
            Span::Diagonal => {
                let start = index_for(self.matrix.size, self.row, self.row + 1);
                let end = start + self.matrix.size - self.row - 1;
                self.span = Span::Contig(self.matrix.data[start..end].iter());
                Some(D::default())
            }
            Span::Contig(ref mut iter) => iter.next().copied(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.matrix.size, Some(self.matrix.size))
    }
}

impl<'a, D: Copy + Default> ExactSizeIterator for Row<'a, D> {}

impl Coordinates {
    #[inline]
    pub(super) fn new(n: usize) -> Self {
        Self {
            i: 0,
            j: 0,
            n_m1: n.saturating_sub(1),
        }
    }
}

impl Iterator for Coordinates {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.j >= self.n_m1 {
            self.i += 1;
            self.j = self.i;
        }
        if self.i >= self.n_m1 {
            return None;
        }
        self.j += 1;
        Some((self.i, self.j))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = n_entries(self.n_m1 + 1);
        (l, Some(l))
    }
}

impl ExactSizeIterator for Coordinates {}

/// Interpret the values as though they were describing the upper triangle or equivalently as
/// though the lower triangle were filled in row major order instead of column major.
pub(crate) fn flip_order<D: Copy>(data: &[D], size: usize) -> Vec<D> {
    assert_eq!(n_entries(size), data.len());
    TransCoordIter::new(size).map(|x| data[x]).collect()
}

#[cfg_attr(test, derive(Debug))]
struct TransCoordIter {
    val: usize,
    row: usize,
    height: usize,
    stride: usize,
    top: usize,
    top_stride: usize,
}

impl TransCoordIter {
    fn new(size: usize) -> Self {
        assert!(size > 0);
        TransCoordIter {
            val: 0,
            row: 0,
            height: size - 1,
            stride: 0,
            top: 0,
            top_stride: 2,
        }
    }
}

impl Iterator for TransCoordIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.height <= 1 {
            return None;
        }

        if self.row >= self.height {
            self.height -= 1;
            self.row = 1;
            self.top += self.top_stride;
            self.val = self.top;
            self.stride = self.top_stride;
            self.top_stride += 1;
            return Some(self.val);
        }

        self.val += self.stride;
        self.stride += 1;
        self.row += 1;
        Some(self.val)
    }
}

/// Converts dist vector length into the number of items (size of full dist matrix).
///
/// An n by n distance matrix is stored as a vector with length `n*(n-1)/2` encoding
/// the lower triangle. Given the length of such a vector, solve the inverse of that
/// expression to get back the number of items
fn n_items(dist_length: usize) -> usize {
    (((8 * dist_length + 1) as f64).sqrt() as usize + 1) / 2
}

pub(crate) const fn n_entries(n: usize) -> usize {
    n * n.saturating_sub(1) / 2
}

/// Gets the index of the data vector corresponding to the given row and column.
///
/// Assumes that `i < j < n` where `n` is the number of rows in the matrix.
const fn index_for(n: usize, i: usize, j: usize) -> usize {
    n * i - (i + 1) * i / 2 + j - i - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_labels<I: IntoIterator<Item = &'static str>>(labs: I) -> Vec<String> {
        labs.into_iter().map(|x| x.to_owned()).collect()
    }

    #[test]
    fn test_n_items() {
        assert_eq!(n_items(0), 1);
        assert_eq!(n_items(1), 2);
        assert_eq!(n_items(3), 3);
        assert_eq!(n_items(6), 4);
        assert_eq!(n_items(10), 5);
        assert_eq!(n_items(15), 6);
        assert_eq!(n_items(4950), 100);
    }

    #[test]
    fn test_dist_iter() {
        let iter = Coordinates::new(5);
        assert_eq!(
            iter.collect::<Vec<_>>(),
            vec![
                (0, 1),
                (0, 2),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 3),
                (1, 4),
                (2, 3),
                (2, 4),
                (3, 4)
            ]
        );
    }

    #[test]
    fn test_index_for() {
        assert_eq!(index_for(5, 0, 1), 0);
        assert_eq!(index_for(5, 0, 2), 1);
        assert_eq!(index_for(5, 0, 3), 2);
        assert_eq!(index_for(5, 0, 4), 3);
        assert_eq!(index_for(5, 1, 2), 4);
        assert_eq!(index_for(5, 1, 3), 5);
        assert_eq!(index_for(5, 1, 4), 6);
        assert_eq!(index_for(5, 2, 3), 7);
        assert_eq!(index_for(5, 2, 4), 8);
        assert_eq!(index_for(5, 3, 4), 9);
    }

    #[test]
    fn test_iter_row() {
        // 0 1 2 3 4  ->
        // 1 0 1 2 3  ->  1
        // 2 1 0 1 2  ->  2 1
        // 3 2 1 0 1  ->  3 2 1
        // 4 3 2 1 0  ->  4 3 2 1
        let m: DistMatrix<u32> = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1].into_iter().collect();

        let mut r0 = m.iter_from_point(0);
        assert_eq!(r0.next(), Some(0));
        assert_eq!(r0.next(), Some(1));
        assert_eq!(r0.next(), Some(2));
        assert_eq!(r0.next(), Some(3));
        assert_eq!(r0.next(), Some(4));
        assert_eq!(r0.next(), None);

        let mut r1 = m.iter_from_point(1);
        assert_eq!(r1.next(), Some(1));
        assert_eq!(r1.next(), Some(0));
        assert_eq!(r1.next(), Some(1));
        assert_eq!(r1.next(), Some(2));
        assert_eq!(r1.next(), Some(3));
        assert_eq!(r1.next(), None);

        let mut r2 = m.iter_from_point(2);
        assert_eq!(r2.next(), Some(2));
        assert_eq!(r2.next(), Some(1));
        assert_eq!(r2.next(), Some(0));
        assert_eq!(r2.next(), Some(1));
        assert_eq!(r2.next(), Some(2));
        assert_eq!(r2.next(), None);

        // 0 1 2 3 4 5  ->
        // 1 0 1 2 3 4  ->  1
        // 2 1 0 1 2 3  ->  2 1
        // 3 2 1 0 1 2  ->  3 2 1
        // 4 3 2 1 0 1  ->  4 3 2 1
        // 5 4 3 2 1 0  ->  5 4 3 2 1
        let m: DistMatrix<u32> = [1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 1]
            .into_iter()
            .collect();

        let mut r2 = m.iter_from_point(2);
        assert_eq!(r2.next(), Some(2));
        assert_eq!(r2.next(), Some(1));
        assert_eq!(r2.next(), Some(0));
        assert_eq!(r2.next(), Some(1));
        assert_eq!(r2.next(), Some(2));
        assert_eq!(r2.next(), Some(3));
        assert_eq!(r2.next(), None);
    }

    #[test]
    fn test_getters() {
        let m: DistMatrix<u32> = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1].into_iter().collect();
        assert_eq!(m.get(0, 3), Some(&3));
        assert_eq!(m.get(3, 0), None);
        assert_eq!(m.get_symmetric(3, 0), Some(&3));
    }

    #[test]
    fn test_getters_by_name() {
        let mut m: DistMatrix<u32> = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1].into_iter().collect();
        m.set_labels(mk_labels(["A", "B", "C", "D", "E"]));
        assert_eq!(m.get_by_name("A", "D"), Some(&3));
        assert_eq!(m.get_by_name("D", "A"), None);
        assert_eq!(m.get_symmetric_by_name("D", "A"), Some(&3));
    }

    #[test]
    fn test_iter_rows() {
        fn expect_row(row: Option<Row<u32>>, reference: Vec<u32>) {
            assert!(row.is_some());
            assert_eq!(row.unwrap().collect::<Vec<u32>>(), reference);
        }

        // 0 1 2 3 4  ->
        // 1 0 1 2 3  ->  1
        // 2 1 0 1 2  ->  2 1
        // 3 2 1 0 1  ->  3 2 1
        // 4 3 2 1 0  ->  4 3 2 1
        let m: DistMatrix<u32> = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1].into_iter().collect();

        let mut rows = m.iter_rows();
        expect_row(rows.next(), vec![0, 1, 2, 3, 4]);
        expect_row(rows.next(), vec![1, 0, 1, 2, 3]);
        expect_row(rows.next(), vec![2, 1, 0, 1, 2]);
        expect_row(rows.next(), vec![3, 2, 1, 0, 1]);
        expect_row(rows.next(), vec![4, 3, 2, 1, 0]);
        assert!(rows.next().is_none());
    }

    #[test]
    fn test_transpose_coords() {
        let coords: Vec<usize> = TransCoordIter::new(5).collect();
        assert_eq!(coords, vec![0, 1, 3, 6, 2, 4, 7, 5, 8, 9]);

        let coords: Vec<usize> = TransCoordIter::new(6).collect();
        assert_eq!(
            coords,
            vec![0, 1, 3, 6, 10, 2, 4, 7, 11, 5, 8, 12, 9, 13, 14]
        );
    }

    #[test]
    fn test_flip_order() {
        let lower = vec![1, 2, 3, 4, 1, 2, 3, 1, 2, 1];
        let upper = vec![1, 2, 1, 3, 2, 1, 4, 3, 2, 1];
        assert_eq!(flip_order(&upper, 5), lower);
    }

    #[test]
    fn test_from_pw_distances() {
        let m = DistMatrix::from_pw_distances(&[1, 6, 2, 5]);
        assert_eq!(m.data, vec![5, 1, 4, 4, 1, 3]);
    }

    #[test]
    fn test_from_pw_distances_with() {
        let m = DistMatrix::from_pw_distances_with(&[1, 6, 2, 5], |x, y| x.abs_diff(y) + 1);
        assert_eq!(m.data, vec![6, 2, 5, 5, 2, 4]);
    }

    #[test]
    fn test_from_iter() {
        let m: DistMatrix<u32> = DistMatrix::from_pw_distances(&[1u32, 6, 2, 5]);
        let m2: DistMatrix<u32> = m.data.clone().into_iter().collect();
        assert_eq!(m, m2);
    }

    #[test]
    fn test_builder() {
        let dists = vec![("A", "B", 5), ("A", "C", 1), ("C", "B", 4)];
        let m = DistMatrix::from_labelled_distances(dists.into_iter()).unwrap();
        let mut m2 = DistMatrix::<u32>::from_pw_distances(&[1_u32, 6, 2]);
        m2.set_labels(mk_labels(["A", "B", "C"]));
        assert_eq!(m, m2);
    }

    #[test]
    fn test_subset() {
        let dist1 = DistMatrix::from_pw_distances(&[1, 2, 6, 8, 3, 1, 9, 3, 4]);
        let dist2 = DistMatrix::from_pw_distances(&[1, 6, 8, 9, 3]);
        assert_eq!(dist1.subset(&[0, 2, 3, 6, 7]), dist2);
    }

    #[test]
    fn test_from_file() {
        let m = DistMatrix::from_tabular_file("tests/long_lt.dat", Separator::Char('\t')).unwrap();
        let labels: Vec<&str> = m.iter_labels().collect();
        assert_eq!(labels, vec!["seq1", "seq2", "seq3", "seq4"]);
        assert_eq!(m.size(), 4);
    }
}
