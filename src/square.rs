//! An arbitrary square matrix.

use std::io::Read;
use std::iter::StepBy;
use std::ops::{Range, Sub};
use std::path::Path;
use std::slice::{self, ChunksExact};

use crate::builder::{DataError, DistBuilder};
use crate::formats::{
    parse_phylip, parse_tabular, PhylipDialect, PhylipError, Separator, TabularError, TabularShape,
};
use crate::symmetric::{n_entries, Coordinates};
use crate::{open_file, AbsDiff, DistMatrix};

/// Stores a full distance matrix in row-major order.
///
/// This type is necessary to represent a distance measure that is not symmetric, and is useful
/// when the performance of accessing or iterating over elements is more important than the memory
/// usage.
///
/// The rowwise/colwise iterators are available when `D: Copy` as they yield copies of values.
#[derive(PartialEq, Eq, Clone)]
#[cfg_attr(test, derive(Debug))]
pub struct SquareMatrix<D> {
    pub(crate) data: Vec<D>,
    pub(crate) size: usize,
    pub(crate) labels: Option<Vec<String>>,
}

/// Iterator over matrix rows; see [`SquareMatrix::iter_rows`].
pub struct Rows<'a, D>(ChunksExact<'a, D>);

/// Copying iterator over a row of the matrix; see [`SquareMatrix::iter_rows`].
pub struct Row<'a, D>(slice::Iter<'a, D>);

/// Iterator over matrix columns; see [`SquareMatrix::iter_cols`].
pub struct Columns<'a, D> {
    matrix: &'a SquareMatrix<D>,
    column: Range<usize>,
}

/// Copying iterator over a column of the matrix; see [`SquareMatrix::iter_cols`].
pub struct Column<'a, D>(StepBy<slice::Iter<'a, D>>);

/// Iterator by reference over element labels; see [`SquareMatrix::iter_labels`].
pub struct Labels<'a>(pub(crate) Option<slice::Iter<'a, String>>);

pub type Iter<'a, D> = std::slice::Iter<'a, D>;
pub type IterMut<'a, D> = std::slice::IterMut<'a, D>;

/// Build a matrix from the values in row major order.
/// The length of the source iterator should be `n * n` for some `n: usize`.
impl<D> FromIterator<D> for SquareMatrix<D> {
    fn from_iter<I: IntoIterator<Item = D>>(iter: I) -> Self {
        let data: Vec<D> = iter.into_iter().collect();
        let size = (data.len() as f64).sqrt() as usize;
        assert_eq!(size * size, data.len());
        SquareMatrix {
            data,
            size,
            labels: None,
        }
    }
}

impl<D> SquareMatrix<D> {
    /// Build a distance matrix from pairwise distances of points where distance is taken to be the
    /// absolute difference as defined by the [`AbsDiff`] trait.
    ///
    /// See [`from_pw_distances_with`](SquareMatrix::from_pw_distances_with) for a version that
    /// allows for an arbitrary distance measure.
    pub fn from_pw_distances<'a, T>(points: &'a [T]) -> Self
    where
        &'a T: Sub<&'a T, Output = D> + AbsDiff<&'a T>,
    {
        let size = points.len();
        (0..size)
            .flat_map(|i| {
                (0..size).map(move |j| {
                    let t1 = &points[i];
                    let t2 = &points[j];
                    t1.abs_diff(t2)
                })
            })
            .collect()
    }

    /// Build a distance matrix from pairwise distances of points where distance is measured using
    /// the provided closure.
    pub fn from_pw_distances_with<T, F: FnMut(&T, &T) -> D>(points: &[T], mut dist_fn: F) -> Self {
        let size = points.len();
        let mut data = Vec::with_capacity(size * size);
        for i in 0..size {
            for j in 0..size {
                let t1 = &points[i];
                let t2 = &points[j];
                data.push(dist_fn(t1, t2))
            }
        }
        SquareMatrix {
            data,
            size,
            labels: None,
        }
    }

    /// Build a matrix from an iterator of labelled distances.
    ///
    /// The iterator can be in any order so long as by the end of iteration there are exactly the
    /// correct entries. An error will be returned if any entry is duplicated, or if there are the
    /// wrong number of entries.
    pub fn from_labelled_distances<S, I>(iter: I) -> Result<SquareMatrix<D>, DataError>
    where
        S: AsRef<str>,
        I: IntoIterator<Item = (S, S, D)>,
    {
        let builder: DistBuilder<D> = std::iter::FromIterator::from_iter(iter);
        builder.try_into()
    }

    /// Retrieve an element from the distance matrix.
    ///
    /// Returns `None` if either `row` or `col` are out of range.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&D> {
        (row <= self.size && col <= self.size).then(|| &self.data[index_for(self.size, row, col)])
    }

    #[inline]
    fn label_to_index<S: AsRef<str>>(&self, label: S) -> Option<usize> {
        self.labels
            .as_ref()?
            .into_iter()
            .position(|l| l == label.as_ref())
    }

    /// Retrieve an element from the distance matrix.
    ///
    /// Equivalent to [`get`](SquareMatrix::get) after converting the labels to indices, except that
    /// `None` will additionally be returned if either `row` or `col` are not known labels.
    pub fn get_by_name<S: AsRef<str>>(&self, row: S, col: S) -> Option<&D> {
        let row = self.label_to_index(row)?;
        let col = self.label_to_index(col)?;
        self.get(row, col)
    }

    /// Decompose into the stored labels and data.
    #[inline]
    pub fn into_inner(self) -> (Option<Vec<String>>, Vec<D>) {
        (self.labels, self.data)
    }

    /// Iterate by reference over all values in the matrix in row-major order.
    pub fn iter(&self) -> Iter<D> {
        self.data.iter()
    }

    /// Iterate by mutable reference over all values in the matrix in row-major order.
    pub fn iter_mut(&mut self) -> IterMut<D> {
        self.data.iter_mut()
    }

    /// Iterator by reference over labels for the underlying elements.
    ///
    /// If no labels are configured for this matrix, the iterator will be empty.
    /// See [Labels::has_labels()] and [set_labels()](DistMatrix::set_labels).
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

    /// Retain only the lower triangle of the matrix.
    ///
    /// See also [`lower_triangle`](SquareMatrix::lower_triangle).
    pub fn into_lower_triangle(self) -> DistMatrix<D>
    where
        D: std::fmt::Debug,
    {
        let mut data = self.data;
        let size = self.size;
        let labels = self.labels;
        for (idx_dest, (i, j)) in Coordinates::new(self.size).enumerate() {
            let idx_src = index_for(size, j, i);
            data.swap(idx_src, idx_dest);
        }
        data.truncate(n_entries(size));
        DistMatrix { data, size, labels }
    }
}

impl SquareMatrix<u32> {
    /// Parse a distance matrix from `reader` in a tabular format.
    ///
    /// See [`formats`](crate::formats) for details.
    #[inline]
    pub fn from_tabular<R: Read>(
        reader: R,
        separator: Separator,
        shape: TabularShape,
    ) -> Result<SquareMatrix<u32>, TabularError> {
        parse_tabular(reader, separator, shape)
    }

    /// Load a distance matrix from the file at `path` in a tabular format.
    ///
    /// The file can optionally be compressed with `gzip`.
    ///
    /// See [`formats`](crate::formats) for details.
    #[inline]
    pub fn from_tabular_file<P: AsRef<Path>>(
        path: P,
        separator: Separator,
        shape: TabularShape,
    ) -> Result<SquareMatrix<u32>, TabularError> {
        let reader = open_file(path)?;
        parse_tabular(reader, separator, shape)
    }
}

impl SquareMatrix<f32> {
    /// Parse a distance matrix from `reader` in PHYLIP square format.
    ///
    /// See [`formats`](crate::formats) for details.
    #[inline]
    pub fn from_phylip<R: Read>(
        reader: R,
        dialect: PhylipDialect,
    ) -> Result<SquareMatrix<f32>, PhylipError> {
        parse_phylip(reader, dialect)
    }

    /// Load a distance matrix from the file at `path` in PHYLIP square format.
    ///
    /// The file can optionally be compressed with `gzip`.
    ///
    /// See [`formats`](crate::formats) for details.
    #[inline]
    pub fn from_phylip_file<P: AsRef<Path>>(
        path: P,
        dialect: PhylipDialect,
    ) -> Result<SquareMatrix<f32>, PhylipError> {
        let reader = open_file(path)?;
        parse_phylip(reader, dialect)
    }
}

impl<D: Copy> SquareMatrix<D> {
    /// Retain only the lower triangle of the matrix.
    ///
    /// See also [`into_lower_triangle`](SquareMatrix::into_lower_triangle).
    pub fn lower_triangle(&self) -> DistMatrix<D> {
        let mut m: DistMatrix<D> = Coordinates::new(self.size)
            .map(|(i, j)| self.data[index_for(self.size, j, i)])
            .collect();
        m.labels = self.labels.clone();
        m
    }

    /// Iterator over pairwise distances from the specified point to all points in order, including
    /// itself.
    #[inline]
    pub fn iter_from_point(&self, idx: usize) -> Row<D> {
        let start = idx * self.size;
        let end = start + self.size;
        Row(self.data[start..end].iter())
    }

    /// Iterator over pairwise distances from all points (including itself) to the specified point
    /// in order.
    #[inline]
    pub fn iter_to_point(&self, idx: usize) -> Column<D> {
        Column(self.data[idx..].iter().step_by(self.size))
    }

    /// Iterate over rows of the distance matrix.
    #[inline]
    pub fn iter_rows(&self) -> Rows<D> {
        Rows(self.data.chunks_exact(self.size))
    }

    /// Iterate over columns of the distance matrix.
    #[inline]
    pub fn iter_cols(&self) -> Columns<D> {
        Columns {
            matrix: self,
            column: 0..self.size,
        }
    }
}

/// Mirror the lower triangle and fill in the diagonal with the default value.
impl<D: Copy + Default> From<DistMatrix<D>> for SquareMatrix<D> {
    #[inline]
    fn from(matrix: DistMatrix<D>) -> Self {
        matrix.iter_rows().flatten().collect()
    }
}

impl<'a, D> Iterator for Rows<'a, D> {
    type Item = Row<'a, D>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|slice| Row(slice.iter()))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, D> ExactSizeIterator for Rows<'a, D> {}

impl<'a, D> DoubleEndedIterator for Rows<'a, D> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|slice| Row(slice.iter()))
    }
}

impl<'a, D> Iterator for Columns<'a, D> {
    type Item = Column<'a, D>;

    fn next(&mut self) -> Option<Self::Item> {
        self.column
            .next()
            .map(|col| Column(self.matrix.data[col..].iter().step_by(self.matrix.size)))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.matrix.size, Some(self.matrix.size))
    }
}

impl<'a, D> ExactSizeIterator for Columns<'a, D> {}

impl<'a, D> DoubleEndedIterator for Columns<'a, D> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.column
            .next_back()
            .map(|col| Column(self.matrix.data[col..].iter().step_by(self.matrix.size)))
    }
}

impl<'a, D> Row<'a, D> {
    /// Return the row as a slice.
    pub fn as_slice(&self) -> &'a [D] {
        self.0.as_slice()
    }
}

impl<'a, D: Copy> Iterator for Row<'a, D> {
    type Item = D;

    #[inline]
    fn next(&mut self) -> Option<D> {
        self.0.next().copied()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a, D: Copy> ExactSizeIterator for Row<'a, D> {}

impl<'a, D: Copy> DoubleEndedIterator for Row<'a, D> {
    #[inline]
    fn next_back(&mut self) -> Option<D> {
        self.0.next_back().copied()
    }
}

impl<'a, D: Copy> Iterator for Column<'a, D> {
    type Item = D;

    #[inline]
    fn next(&mut self) -> Option<D> {
        self.0.next().copied()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let l = self.0.len();
        (l, Some(l))
    }
}

impl<'a, D: Copy> ExactSizeIterator for Column<'a, D> {}

impl<'a> Iterator for Labels<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        self.0
            .as_mut()
            .and_then(|inner| inner.next().map(String::as_str))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.0 {
            Some(ref inner) => inner.size_hint(),
            None => (0, Some(0)),
        }
    }
}

impl<'a> ExactSizeIterator for Labels<'a> {}

impl<'a> Labels<'a> {
    /// Does the matrix have labels?
    #[inline]
    pub fn has_labels(&self) -> bool {
        self.0.is_some()
    }
}

/// Gets the index of the data vector corresponding to the given row and column.
pub(crate) const fn index_for(n: usize, i: usize, j: usize) -> usize {
    n * i + j
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_labels<I: IntoIterator<Item = &'static str>>(labs: I) -> Vec<String> {
        labs.into_iter().map(|x| x.to_owned()).collect()
    }

    #[test]
    fn test_from_pw_distances() {
        //    1  6  2  5
        //   ------------
        // 1| 0  5  1  4
        // 6| 5  0  4  1
        // 2| 1  4  0  3
        // 5| 4  1  3  0
        let m = SquareMatrix::<u32>::from_pw_distances(&[1_u32, 6, 2, 5]);
        assert_eq!(m.data, vec![0, 5, 1, 4, 5, 0, 4, 1, 1, 4, 0, 3, 4, 1, 3, 0]);

        let m = SquareMatrix::<i32>::from_pw_distances(&[1_i32, 6, 2, 5]);
        assert_eq!(m.data, vec![0, 5, 1, 4, 5, 0, 4, 1, 1, 4, 0, 3, 4, 1, 3, 0]);
    }

    #[test]
    fn test_from_pw_distances_with() {
        //    1   6   2   5
        //   ---------------
        // 1| 0  -5  -1  -4
        // 6| 5   0   4   1
        // 2| 1  -4   0  -3
        // 5| 4  -1   3   0
        let m = SquareMatrix::from_pw_distances_with(&[1_i32, 6, 2, 5], |x, y| x - y);
        assert_eq!(
            m.data,
            vec![0, -5, -1, -4, 5, 0, 4, 1, 1, -4, 0, -3, 4, -1, 3, 0]
        );
    }

    #[test]
    fn test_from_iter() {
        let m: SquareMatrix<u32> = SquareMatrix::from_pw_distances(&[1u32, 6, 2, 5]);
        let m2: SquareMatrix<u32> = m.data.clone().into_iter().collect();
        assert_eq!(m, m2);
    }

    #[test]
    fn test_builder() {
        let dists = vec![
            ("A", "A", 0),
            ("A", "B", 5),
            ("A", "C", 1),
            ("B", "A", 5),
            ("B", "B", 0),
            ("B", "C", 4),
            ("C", "A", 1),
            ("C", "B", 4),
            ("C", "C", 0),
        ];
        let m = SquareMatrix::from_labelled_distances(dists.into_iter()).unwrap();
        let mut m2 = SquareMatrix::<u32>::from_pw_distances(&[1_u32, 6, 2]);
        m2.set_labels(mk_labels(["A", "B", "C"]));
        assert_eq!(m, m2);
    }

    #[test]
    fn test_getters() {
        let mut m = SquareMatrix::from_pw_distances_with(&[1_i32, 6, 2, 5], |x, y| x - y);
        m.set_labels(mk_labels(["A", "B", "C", "D"]));

        assert_eq!(m.get(1, 2), Some(&4));
        assert_eq!(m.get(2, 1), Some(&-4));
        assert_eq!(m.get_by_name("B", "C"), Some(&4));
        assert_eq!(m.get_by_name("C", "B"), Some(&-4));
    }

    #[test]
    fn test_iter_row() {
        let m = SquareMatrix::from_pw_distances_with(&[1_i32, 6, 2, 5], |x, y| x - y);

        let mut r0 = m.iter_from_point(0);
        assert_eq!(r0.next(), Some(0));
        assert_eq!(r0.next(), Some(-5));
        assert_eq!(r0.next(), Some(-1));
        assert_eq!(r0.next(), Some(-4));
        assert_eq!(r0.next(), None);

        let mut r1 = m.iter_from_point(1);
        assert_eq!(r1.next(), Some(5));
        assert_eq!(r1.next(), Some(0));
        assert_eq!(r1.next(), Some(4));
        assert_eq!(r1.next(), Some(1));
        assert_eq!(r1.next(), None);

        let m = SquareMatrix::<u32>::from_pw_distances(&[1u32, 6, 2, 5, 10]);

        let mut r2 = m.iter_from_point(2);
        assert_eq!(r2.next(), Some(1));
        assert_eq!(r2.next(), Some(4));
        assert_eq!(r2.next(), Some(0));
        assert_eq!(r2.next(), Some(3));
        assert_eq!(r2.next(), Some(8));
        assert_eq!(r2.next(), None);
    }

    #[test]
    fn test_iter_col() {
        let m = SquareMatrix::from_pw_distances_with(&[1_i32, 6, 2, 5], |x, y| x - y);

        let mut r0 = m.iter_to_point(0);
        assert_eq!(r0.next(), Some(0));
        assert_eq!(r0.next(), Some(5));
        assert_eq!(r0.next(), Some(1));
        assert_eq!(r0.next(), Some(4));
        assert_eq!(r0.next(), None);

        let mut r1 = m.iter_to_point(1);
        assert_eq!(r1.next(), Some(-5));
        assert_eq!(r1.next(), Some(0));
        assert_eq!(r1.next(), Some(-4));
        assert_eq!(r1.next(), Some(-1));
        assert_eq!(r1.next(), None);

        let m = SquareMatrix::<u32>::from_pw_distances(&[1u32, 6, 2, 5, 10]);

        let mut r2 = m.iter_to_point(2);
        assert_eq!(r2.next(), Some(1));
        assert_eq!(r2.next(), Some(4));
        assert_eq!(r2.next(), Some(0));
        assert_eq!(r2.next(), Some(3));
        assert_eq!(r2.next(), Some(8));
        assert_eq!(r2.next(), None);
    }

    #[test]
    fn test_iter_rows() {
        fn expect_row(row: Option<Row<i32>>, reference: Vec<i32>) {
            assert!(row.is_some());
            assert_eq!(row.unwrap().collect::<Vec<i32>>(), reference);
        }

        let m = SquareMatrix::from_pw_distances_with(&[1_i32, 6, 2, 5], |x, y| x - y);
        let mut rows = m.iter_rows();
        expect_row(rows.next(), vec![0, -5, -1, -4]);
        expect_row(rows.next(), vec![5, 0, 4, 1]);
        expect_row(rows.next(), vec![1, -4, 0, -3]);
        expect_row(rows.next(), vec![4, -1, 3, 0]);
        assert!(rows.next().is_none());
    }

    #[test]
    fn test_iter_cols() {
        fn expect_col(col: Option<Column<i32>>, reference: Vec<i32>) {
            assert!(col.is_some());
            assert_eq!(col.unwrap().collect::<Vec<i32>>(), reference);
        }

        let m = SquareMatrix::from_pw_distances_with(&[1_i32, 6, 2, 5], |x, y| x - y);
        let mut cols = m.iter_cols();
        expect_col(cols.next(), vec![0, 5, 1, 4]);
        expect_col(cols.next(), vec![-5, 0, -4, -1]);
        expect_col(cols.next(), vec![-1, 4, 0, 3]);
        expect_col(cols.next(), vec![-4, 1, -3, 0]);
        assert!(cols.next().is_none());
    }

    #[test]
    fn test_from_sym() {
        let m = DistMatrix::<u32>::from_pw_distances(&[1_u32, 2, 3, 4, 5]);
        let m1 = SquareMatrix::<u32>::from_pw_distances(&[1_u32, 2, 3, 4, 5]);
        let m2: SquareMatrix<u32> = m.into();

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_to_sym() {
        #[rustfmt::skip]
        let m: SquareMatrix<u32> = [
            0, 0, 0,  0, 0,
            1, 0, 0,  0, 0,
            2, 5, 0,  0, 0,
            3, 6, 8,  0, 0,
            4, 7, 9, 10, 0,
        ].into_iter().collect();
        let m1: DistMatrix<u32> = m.lower_triangle();
        let m2: DistMatrix<u32> = (1..=10).collect();

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_to_sym_inplace() {
        #[rustfmt::skip]
        let m: SquareMatrix<u32> = [
            0, 0, 0,  0, 0,
            1, 0, 0,  0, 0,
            2, 5, 0,  0, 0,
            3, 6, 8,  0, 0,
            4, 7, 9, 10, 0,
        ].into_iter().collect();
        let m1: DistMatrix<u32> = m.into_lower_triangle();
        let m2: DistMatrix<u32> = (1..=10).collect();

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_from_file() {
        let m = SquareMatrix::from_tabular_file(
            "tests/snp-dists/default.dat",
            Separator::Char('\t'),
            TabularShape::Wide,
        )
        .unwrap();
        let labels: Vec<&str> = m.iter_labels().collect();
        assert_eq!(labels, vec!["seq1", "seq2", "seq3", "seq4"]);
        assert_eq!(m.size(), 4);

        let m = SquareMatrix::from_tabular_file(
            "tests/snp-dists/default.dat.gz",
            Separator::Char('\t'),
            TabularShape::Wide,
        )
        .unwrap();
        let labels: Vec<&str> = m.iter_labels().collect();
        assert_eq!(labels, vec!["seq1", "seq2", "seq3", "seq4"]);
        assert_eq!(m.size(), 4);
    }
}
