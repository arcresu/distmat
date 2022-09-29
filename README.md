# distmat

[![crates-io-v](https://img.shields.io/crates/v/distmat)](https://crates.io/crates/distmat)
![crates-io-l](https://img.shields.io/crates/l/distmat)
[![docs-rs](https://img.shields.io/docsrs/distmat)](https://docs.rs/distmat)

> **Distance matrix data types and file formats**

Matrix types specialised for storing pairwise distance data, and parsers for
some common file formats for storing such data.

```rust
use distmat::{DistMatrix, SquareMatrix};
use distmat::formats::{PhylipDialect, Separator, TabularShape};

fn main() {
    // A symmetric matrix stored as the lower triangle:
    //   _1__5__3
    // 1|
    // 5| 4
    // 3| 2  2
    let matrix1 = DistMatrix::from_pw_distances(&[1, 5, 3]);
    assert_eq!(matrix1.get_symmetric(1, 2), Some(2));

    // A square matrix stored in row major order:
    //   _1___5___3
    // 1| 0  -4  -2
    // 5| 4   0   2
    // 3| 2  -2   0
    let matrix2 = SquareMatrix::from_pw_distances_with(&[1, 5, 3], |x, y| x - y);
    let mut total = 0;
    for row in matrix2.iter_rows() {
        total += row.sum();
    }

    let (_labels, _matrix) =
        SquareMatrix::from_tabular_file("snp-dists.dat", Separator::Char('\t'), TabularShape::Wide).unwrap();
    let (_labels, _matrix) =
        SquareMatrix::from_phylip_file("phylip.dist", PhylipDialect::Strict).unwrap();
    let (_labels, _matrix) =
        DistMatrix::from_phylip_file("phylip_lt.dist", PhylipDialect::Relaxed).unwrap();
}

```


## Purpose
Goals:

  * Read and write pairwise distance data from any reasonable formats,
    especially those used in bioinformatics.
  * Provide a convenient API to interact with distance data.

Non-goals:

  * Linear algebra. There are many linear algebra libraries available with
    matrix data structures. At most distmat will help you export your data to
    these libraries.
  * Algorithms. You can provide a closure to distmat to construct a distance
    matrix, but any specialised algorithms or distance measures are best
    implemented elsewhere.


## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).
