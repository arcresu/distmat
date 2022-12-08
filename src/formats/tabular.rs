use std::io::Read;
use std::io::{BufRead, BufReader};
use thiserror::Error;

use crate::builder::{DataError, DistBuilder};
use crate::symmetric::flip_order;
use crate::{DistMatrix, SquareMatrix};

#[derive(Clone, Copy, Debug)]
pub enum Separator {
    /// Values are separated by a single character.
    Char(char),

    /// Values are separated by any amount of ASCII whitespace.
    Whitespace,
}

pub enum TabularShape {
    /// Wide tabular data has rows and columns corresponding to the matrix entries.
    /// The first row and the first column should contain the taxa labels in the
    /// same order. The top leftmost cell is ignored.
    ///
    /// This file uses `Separator::Char(',')` and has 3 taxa:
    ///
    /// ```{txt}
    /// ,A,B,C
    /// A,0,1,2
    /// B,1,0,1
    /// C,2,1,0
    /// ```
    Wide,

    /// Long tabular data has exactly 3 columns: the two taxa and the distance.
    /// A header row is optional and ignored. This shape can either represent a
    /// complete square matrix with `N * N` rows, or the lower triangle with
    /// `N * (N - 1) / 2` rows.
    ///
    /// This file uses `Separator::Whitespace` and represents only the lower
    /// triangle for a matrix with 3 taxa:
    ///
    /// ```{txt}
    /// from    to   dist
    /// A       B    1
    /// A       C    2
    /// B       C    1
    /// ```
    ///
    /// This file represents a complete matrix:
    ///
    /// ```{txt}
    /// from,to,d
    /// A,A,0
    /// A,B,1
    /// A,C,2
    /// B,A,1
    /// B,B,0
    /// B,C,1
    /// C,A,2
    /// C,B,1
    /// C,C,0
    /// ```
    ///
    /// Note that [SquareMatrix::from_labelled_distances] and [DistMatrix::from_labelled_distances]
    /// construct matrix types from an iterator of this shape if your data is not stored in
    /// a tabular file.
    Long,
}

#[derive(Error, Debug)]
pub enum TabularError {
    /// An underlying I/O error occurred.
    #[error("unable to read distance matrix file")]
    Io(#[from] std::io::Error),

    #[error("unable to read header row with taxa labels")]
    Header,

    #[error("the file contained no data (empty or header had no delimeters)")]
    NoData,

    #[error("matrix row {0} (label '{1}') had {2} entries when {3} were expected")]
    RowWidth(usize, String, usize, usize),

    #[error("expected 3 columns: {0}")]
    ColsTruncated(String),

    #[error("matrix row {0} had label '{1}' but '{2}' was expected")]
    RowOrder(usize, String, String),

    #[error("row did not start with a label: {0}")]
    Label(String),

    #[error("reached end of file while expecting {0} more matrix rows")]
    RowsTruncated(usize),

    #[error("data has incorrect shape")]
    Data(#[from] DataError),

    /// Unable to parse a numeric value.
    #[error("expected integer found `{0}': {1}")]
    Numeric(String, std::num::ParseIntError),
}

/// Parse a distance matrix in a square format.
pub fn parse<R: Read>(
    reader: R,
    separator: Separator,
    shape: TabularShape,
) -> Result<SquareMatrix<u32>, TabularError> {
    let (labels, data, size) = match shape {
        TabularShape::Wide => parse_wide(reader, separator)?,
        TabularShape::Long => parse_long(reader, separator, false)?,
    };
    let labels = Some(labels);
    let matrix = SquareMatrix { data, size, labels };
    Ok(matrix)
}

/// Parse a distance matrix where only the lower triangle is represented.
pub fn parse_lt<R: Read>(reader: R, separator: Separator) -> Result<DistMatrix<u32>, TabularError> {
    let (labels, data, size) = parse_long(reader, separator, true)?;
    let labels = Some(labels);
    let data = flip_order(&data, size);
    let matrix = DistMatrix { data, size, labels };
    Ok(matrix)
}

fn parse_wide<R: Read>(
    reader: R,
    separator: Separator,
) -> Result<(Vec<String>, Vec<u32>, usize), TabularError> {
    let labels;
    let mut data;

    {
        let mut br = BufReader::new(reader);
        let mut buf = String::new();

        //read the header row
        br.read_line(&mut buf).map_err(|_| TabularError::Header)?;
        let (_, rest) = separator.split_label(&buf)?;
        labels = separator.split_str(rest.trim_end());
        if labels.is_empty() {
            return Err(TabularError::NoData);
        }
        data = Vec::with_capacity(labels.len() * labels.len());

        let mut row = 0;

        loop {
            row += 1;
            buf.clear();
            let n = br.read_line(&mut buf)?;
            if n > 0 {
                let (label, rest) = separator.split_label(&buf)?;
                if label != labels[row - 1] {
                    return Err(TabularError::RowOrder(
                        row,
                        label.to_owned(),
                        labels[row - 1].clone(),
                    ));
                }

                let n_read = separator.split_u32(rest.trim_end(), &mut data)?;
                if n_read != labels.len() {
                    return Err(TabularError::RowWidth(
                        row,
                        label.to_owned(),
                        n_read,
                        labels.len(),
                    ));
                }
            } else {
                break; // EOF
            }
        }

        if row < labels.len() {
            return Err(TabularError::RowsTruncated(labels.len() - row));
        }
    }

    let size = labels.len();
    Ok((labels, data, size))
}

fn parse_long<R: Read>(
    reader: R,
    separator: Separator,
    lower_triangle: bool,
) -> Result<(Vec<String>, Vec<u32>, usize), TabularError> {
    let builder = parse_long_impl(reader, separator)?;
    let labels = builder.labels.clone();
    let size = labels.len();

    if lower_triangle {
        let matrix: DistMatrix<u32> = builder.try_into()?;
        Ok((labels, matrix.data, size))
    } else {
        let matrix: SquareMatrix<u32> = builder.try_into()?;
        Ok((labels, matrix.data, size))
    }
}

fn parse_long_impl<R: Read>(
    reader: R,
    separator: Separator,
) -> Result<DistBuilder<u32>, TabularError> {
    let mut builder = DistBuilder::<u32>::new();

    let mut br = BufReader::new(reader);
    let mut buf = String::new();

    let mut row = 0;
    let mut header_seen = false;

    loop {
        row += 1;
        buf.clear();
        let n = br.read_line(&mut buf)?;
        if n > 0 {
            let parts = separator.split_3(buf.trim_end());
            if row == 1 && !header_seen {
                if let Err(TabularError::Numeric(_, _)) = parts {
                    row = 0;
                    header_seen = true;
                    continue;
                }
            }

            let (name1, name2, distance) = parts?;
            builder.add(name1, name2, distance)?;
        } else {
            break; // EOF
        }
    }

    Ok(builder)
}

impl Separator {
    fn split_str(&self, line: &str) -> Vec<String> {
        match self {
            Separator::Char(c) => line.split(*c).map(str::to_owned).collect(),
            Separator::Whitespace => line.split_ascii_whitespace().map(str::to_owned).collect(),
        }
    }

    fn split_label<'a>(&self, line: &'a str) -> Result<(&'a str, &'a str), TabularError> {
        match self {
            Separator::Char(c) => line
                .split_once(*c)
                .ok_or_else(|| TabularError::Label(line.to_owned())),
            Separator::Whitespace => {
                let (label, rest) = line
                    .split_once(|x| char::is_ascii_whitespace(&x))
                    .ok_or_else(|| TabularError::Label(line.to_owned()))?;
                Ok((label, rest.trim_start()))
            }
        }
    }

    fn split_u32(&self, line: &str, data: &mut Vec<u32>) -> Result<usize, TabularError> {
        let orig_size = data.len();

        match self {
            Separator::Char(c) => {
                for number in line.trim_end().split(*c) {
                    data.push(
                        number
                            .parse()
                            .map_err(|e| TabularError::Numeric(number.to_owned(), e))?,
                    );
                }
            }
            Separator::Whitespace => {
                for number in line.trim_end().split_ascii_whitespace() {
                    data.push(
                        number
                            .parse()
                            .map_err(|e| TabularError::Numeric(number.to_owned(), e))?,
                    );
                }
            }
        }

        Ok(data.len() - orig_size)
    }

    fn split_3<'a>(&self, line: &'a str) -> Result<(&'a str, &'a str, u32), TabularError> {
        let (p1, p2, p3) = match self {
            Separator::Char(c) => extract_3(line, line.split(*c))?,
            Separator::Whitespace => extract_3(line, line.split_ascii_whitespace())?,
        };

        let p3 = p3
            .parse()
            .map_err(|e| TabularError::Numeric(p3.to_owned(), e))?;
        Ok((p1, p2, p3))
    }
}

fn extract_3<'a>(
    line: &'a str,
    mut splitter: impl Iterator<Item = &'a str>,
) -> Result<(&'a str, &'a str, &'a str), TabularError> {
    let p1 = splitter
        .next()
        .ok_or_else(|| TabularError::ColsTruncated(line.to_owned()))?;
    let p2 = splitter
        .next()
        .ok_or_else(|| TabularError::ColsTruncated(line.to_owned()))?;
    let p3 = splitter
        .next()
        .ok_or_else(|| TabularError::ColsTruncated(line.to_owned()))?;
    if splitter.next().is_some() {
        return Err(TabularError::ColsTruncated(line.to_owned()));
    }
    Ok((p1, p2, p3))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_labels() -> Vec<String> {
        vec![
            "seq1".to_owned(),
            "seq2".to_owned(),
            "seq3".to_owned(),
            "seq4".to_owned(),
        ]
    }

    fn expected_data() -> Vec<u32> {
        vec![
            // seq1 seq2 seq3 seq4
            0, 1, 2, 3, // seq1
            1, 0, 3, 4, // seq2
            2, 3, 0, 4, // seq3
            3, 4, 4, 0, // seq4
        ]
    }

    #[test]
    fn test_wide() {
        let f = include_bytes!("../../tests/snp-dists/default.dat");
        let (labels, data, _size) = parse_wide(f.as_slice(), Separator::Char('\t')).unwrap();
        assert_eq!(labels, expected_labels());
        assert_eq!(data, expected_data());
    }

    #[test]
    fn test_version() {
        let f = include_bytes!("../../tests/snp-dists/version.dat");
        let (labels, data, _size) = parse_wide(f.as_slice(), Separator::Char('\t')).unwrap();
        assert_eq!(labels, expected_labels());
        assert_eq!(data, expected_data());
    }

    #[test]
    fn test_comma() {
        let f = include_bytes!("../../tests/snp-dists/comma.dat");
        let (labels, data, _size) = parse_wide(f.as_slice(), Separator::Char(',')).unwrap();
        assert_eq!(labels, expected_labels());
        assert_eq!(data, expected_data());
    }

    #[test]
    fn test_melt() {
        let f = include_bytes!("../../tests/snp-dists/melt.dat");
        let (labels, data, _size) = parse_long(f.as_slice(), Separator::Char('\t'), false).unwrap();
        assert_eq!(labels, expected_labels());
        assert_eq!(data, expected_data());
    }

    #[test]
    fn test_melt_comma() {
        let f = include_bytes!("../../tests/snp-dists/melt-comma.dat");
        let (labels, data, _size) = parse_long(f.as_slice(), Separator::Char(','), false).unwrap();
        assert_eq!(labels, expected_labels());
        assert_eq!(data, expected_data());
    }

    #[test]
    fn test_melt_lt() {
        let f = include_bytes!("../../tests/long_lt.dat");
        let (labels, data, _size) = parse_long(f.as_slice(), Separator::Char('\t'), true).unwrap();
        assert_eq!(labels, expected_labels());
        assert_eq!(data, vec![1, 2, 3, 3, 4, 4]);
    }
}
