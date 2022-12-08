//! Data types and file formats for distance matrix data.
//!
//! Distance matrix data types:
//!   * [`DistMatrix`] stores only the lower diagonal so works for symmetric distance measures. It
//!     is more memory efficient and is compatible with R's `dist` objects.
//!   * [`SquareMatrix`] stores the full matrix and is more efficient to access.
//!
//! Both matrix types have their data stored in a single `Vec<D>`, and are generic
//! over `D`. They provide various iterators and accessors to retrieve elements.
//! They can optionally store labels (e.g. taxon labels from bioinformatic distance matrix formats).
//!
//! There are different ways to load your data into matrix types:
//!
//!   * using a pairwise distance measure: `from_pw_distances` or `from_pw_distances_with`,
//!   * from an iterator over pairs of labels and distances with `from_labelled_distances`,
//!   * from a vector stored in the correct order with `collect`,
//!   * from a file (see [`formats`]): tabular (e.g. `snp-dists`) or PHYLIP,
//!   * `let square: SquareMatrix<D> = dist.into();` where `dist: DistMatrix<D>`,
//!   * `let dist: DistMatrix<D> = square.lower_triangle();` where `square: SquareMatrix<D>`.

use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::ops::Sub;
use std::path::Path;

mod builder;
pub mod formats;
pub mod square;
pub mod symmetric;

pub use builder::DataError;
pub use square::SquareMatrix;
pub use symmetric::DistMatrix;

/// Implemented for types with an obvious default meaning of absolute distance between points.
pub trait AbsDiff<Rhs = Self>: Sized + Sub<Rhs> {
    /// Absolute difference with `other`.
    fn abs_diff(self, other: Rhs) -> <Self as Sub<Rhs>>::Output;
}

impl<T: PartialOrd<T> + Sub<T>> AbsDiff<T> for T {
    fn abs_diff(self, other: Self) -> <Self as Sub>::Output {
        if self > other {
            self - other
        } else {
            other - self
        }
    }
}

impl<'a, T> AbsDiff<&'a T> for T
where
    T: PartialOrd<&'a T> + Sub<&'a T>,
    &'a T: Sub<T, Output = <T as Sub<&'a T>>::Output>,
{
    fn abs_diff(self, other: &'a T) -> <Self as Sub<&'a T>>::Output {
        if self > other {
            self - other
        } else {
            other - self
        }
    }
}

const GZIP_HEADER: [u8; 2] = [0x1f, 0x8b];

pub(crate) fn open_file<P: AsRef<Path>>(path: P) -> io::Result<Box<dyn io::Read>> {
    let mut file = File::open(path)?;

    let mut buffer = [0_u8; 2];
    file.read_exact(&mut buffer)?;
    file.seek(SeekFrom::Start(0))?;

    if buffer == GZIP_HEADER {
        Ok(Box::new(GzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}
