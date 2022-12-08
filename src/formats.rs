//! Read distance matrix data to different file formats.
//!
//! If you have a file produced by a tool for which none of the options here work,
//! please report an issue describing how the file was made and an example of the format.
//!
//! ## PHYLIP
//! Parse distance matrix in the PHYLIP format.
//!
//! See [`PhylipDialect`] for a description of the taxon label dialects.
//! PHYLIP distance matrix data can be stored in two shapes:
//!
//!   * square: the full distance matrix is represented, e.g.:
//!
//!     ```{txt}
//!     3
//!     taxon1 0 1 2
//!     taxon2 1 0 1
//!     taxon3 2 1 0
//!     ```
//!
//!     See [`SquareMatrix::from_phylip`](crate::SquareMatrix::from_phylip).
//!
//!   * lower triangle: only the lower triangle is represented, e.g.:
//!
//!     ```{txt}
//!     3
//!     taxon1
//!     taxon2 1
//!     taxon3 2 1
//!     ```
//!
//!     The diagonal terms are assumed to be zero and the distance measure is symmetric.
//!
//!     See [`DistMatrix::from_phylip`](crate::DistMatrix::from_phylip).
//!
//!
//! ## Tabular
//! Parses distance matrix files that are CSV/TSV tabular data.
//!
//! This includes the various output types produced by
//! [snp-dists](https://github.com/tseemann/snp-dists).
//!
//! See [`TabularShape`] for a description of the accepted shapes.
//!
//! See [`SquareMatrix::from_tabular`](crate::SquareMatrix::from_tabular) and
//! [`DistMatrix::from_tabular`](crate::DistMatrix::from_tabular).

mod phylip;
pub(crate) use phylip::{parse as parse_phylip, parse_lt as parse_phylip_lt};
pub use phylip::{PhylipDialect, PhylipError};

mod tabular;
pub(crate) use tabular::{parse as parse_tabular, parse_lt as parse_tabular_lt};
pub use tabular::{Separator, TabularError, TabularShape};
