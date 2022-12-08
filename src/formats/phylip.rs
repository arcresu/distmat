use std::io::Read;
use std::io::{BufRead, BufReader};
use thiserror::Error;

use crate::symmetric::flip_order;
use crate::{DistMatrix, SquareMatrix};

/// Dialect used for taxon labels.
///
/// The known variants of the PHYLIP format differ in how they separate the
/// taxon labels from the distance data. In either format the matrix entries
/// are separated from each other by any amount of whitespace (including new
/// lines).
#[derive(Clone, Copy, Debug)]
pub enum PhylipDialect {
    /// The traditional format where taxon labels occupy the first 10
    /// characters of each line. Longer labels are not possible, and shorter
    /// labels must be padded with spaces to reach at least 10 characters.
    /// Taxon labels may contain spaces.
    ///
    /// The following file has 3 taxa: `"taxon 1"`, `"taxon 2"`, and `"taxon 3"`:
    /// ```{txt}
    /// 3
    /// taxon 1    0 1 2
    /// taxon 2    1 0 1
    /// taxon 3    2 1 0
    /// ```
    Strict,

    /// The relaxed dialect. Taxon labels may not contain spaces, but may be of
    /// any length. They are separated from the distace data by any number of
    /// spaces.
    ///
    /// The following file has 3 taxa `"taxon"`, `"taxon_very_long"`, and
    /// `"taxon-other-delim"`:
    /// ```{txt}
    /// 3
    /// taxon 0 1 2
    /// taxon_very_long    1 0 1
    /// taxon-other-delim 2 1 0
    /// ```
    Relaxed,
}

/// An error in parsing the distance matrix file.
#[derive(Error, Debug)]
pub enum PhylipError {
    /// An underlying I/O error occurred.
    #[error("unable to read distance matrix file")]
    Io(#[from] std::io::Error),

    #[error("unable to read header row with taxa count")]
    Header,

    #[error("row missing label")]
    Label,

    #[error("matrix row {0} had at least {1} entries when {2} were expected")]
    Extra(usize, usize, usize),

    #[error("reached end of file while expecting {0} more matrix rows")]
    RowsTruncated(usize),

    /// Unable to parse a numeric value.
    #[error("expected floating point number found `{0}': {1}")]
    Numeric(String, std::num::ParseFloatError),
}

/// Parse a distance matrix in a square format.
pub fn parse<R: Read>(reader: R, dialect: PhylipDialect) -> Result<SquareMatrix<f32>, PhylipError> {
    let (labels, data, size) = parse_impl(reader, dialect, false)?;
    let labels = Some(labels);
    let matrix = SquareMatrix { data, size, labels };
    Ok(matrix)
}

/// Parse a distance matrix where only the lower triangle is represented.
pub fn parse_lt<R: Read>(
    reader: R,
    dialect: PhylipDialect,
) -> Result<DistMatrix<f32>, PhylipError> {
    let (labels, data, size) = parse_impl(reader, dialect, true)?;
    let labels = Some(labels);
    let matrix = DistMatrix { data, size, labels };
    Ok(matrix)
}

fn parse_impl<R: Read>(
    reader: R,
    dialect: PhylipDialect,
    lower_triangle: bool,
) -> Result<(Vec<String>, Vec<f32>, usize), PhylipError> {
    let mut br = BufReader::new(reader);
    let mut buf = String::new();

    br.read_line(&mut buf)?;
    let size: usize = buf.trim().parse().map_err(|_| PhylipError::Header)?;

    let mut labels = Vec::<String>::with_capacity(size);
    let mut data = Vec::<f32>::with_capacity(size * size);
    let mut expected_entries = 0;

    loop {
        buf.clear();
        let n = br.read_line(&mut buf)?;
        if n > 0 {
            let n_read;

            if expected_entries == 0 {
                let (label, new_data) = parse_label(&buf, dialect)?;
                expected_entries = if lower_triangle { labels.len() } else { size };
                labels.push(label.to_owned());
                n_read = parse_data(new_data, &mut data)?;
            } else {
                n_read = parse_data(&buf, &mut data)?;
            }

            if n_read > expected_entries {
                return Err(PhylipError::Extra(
                    labels.len(),
                    expected_entries + n_read,
                    size,
                ));
            } else {
                expected_entries -= n_read;
            }
        } else {
            // EOF
            let remaining = size - labels.len();
            if remaining > 0 {
                return Err(PhylipError::RowsTruncated(remaining));
            }

            break;
        }
    }

    if lower_triangle {
        data = flip_order(&data, size);
    }

    Ok((labels, data, size))
}

fn parse_label(line: &str, dialect: PhylipDialect) -> Result<(&str, &str), PhylipError> {
    match dialect {
        PhylipDialect::Strict => {
            if line.len() >= 10 {
                Ok((line[0..10].trim_end(), line[10..].trim_start()))
            } else {
                Err(PhylipError::Label)
            }
        }

        PhylipDialect::Relaxed => match line.split_once(|x| char::is_ascii_whitespace(&x)) {
            Some((label, rest)) => Ok((label, rest.trim_start())),
            None => Err(PhylipError::Label),
        },
    }
}

fn parse_data(line: &str, data: &mut Vec<f32>) -> Result<usize, PhylipError> {
    let orig_size = data.len();

    for number in line.trim_end().split_ascii_whitespace() {
        data.push(
            number
                .parse()
                .map_err(|e| PhylipError::Numeric(number.to_owned(), e))?,
        );
    }

    Ok(data.len() - orig_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected_matrix() -> SquareMatrix<f32> {
        let mut matrix: SquareMatrix<f32> = [
            0.0000, 1.6866, 1.7198, 1.6606, 1.5243, 1.6043, 1.5905, 1.6866, 0.0000, 1.5232, 1.4841,
            1.4465, 1.4389, 1.4629, 1.7198, 1.5232, 0.0000, 0.7115, 0.5958, 0.6179, 0.5583, 1.6606,
            1.4841, 0.7115, 0.0000, 0.4631, 0.5061, 0.4710, 1.5243, 1.4465, 0.5958, 0.4631, 0.0000,
            0.3484, 0.3083, 1.6043, 1.4389, 0.6179, 0.5061, 0.3484, 0.0000, 0.2692, 1.5905, 1.4629,
            0.5583, 0.4710, 0.3083, 0.2692, 0.0000,
        ]
        .into_iter()
        .collect();
        matrix.set_labels(vec![
            "Bovine".to_owned(),
            "Mouse".to_owned(),
            "Gibbon".to_owned(),
            "Orang".to_owned(),
            "Gorilla".to_owned(),
            "Chimp".to_owned(),
            "Human".to_owned(),
        ]);
        matrix
    }

    #[test]
    fn test_square() {
        let f = include_bytes!("../../tests/phylip/square.dist");

        let matrix = parse(f.as_slice(), PhylipDialect::Strict).unwrap();
        assert_eq!(matrix, expected_matrix());

        let matrix = parse(f.as_slice(), PhylipDialect::Relaxed).unwrap();
        assert_eq!(matrix, expected_matrix());
    }

    #[test]
    fn test_square_multiline() {
        let f = include_bytes!("../../tests/phylip/square_multiline.dist");
        let matrix = parse(f.as_slice(), PhylipDialect::Relaxed).unwrap();
        assert_eq!(matrix, expected_matrix());
    }

    #[test]
    fn test_lower_triangle() {
        let f = include_bytes!("../../tests/phylip/lower.dist");
        let matrix = parse_lt(f.as_slice(), PhylipDialect::Relaxed).unwrap();
        assert_eq!(matrix, expected_matrix().lower_triangle());
    }

    #[test]
    fn test_lower_triangle_multiline() {
        let f = include_bytes!("../../tests/phylip/lower_multiline.dist");
        let matrix = parse_lt(f.as_slice(), PhylipDialect::Strict).unwrap();

        let mut m_exp: DistMatrix<f32> = [
            1.7043, 2.0235, 2.1378, 1.5232, 1.8261, 1.9182, 2.0039, 1.9431, 1.9663, 2.0593, 1.6664,
            1.732, 1.7101, 1.1901, 1.3287, 1.2423, 1.2508, 1.2536, 1.3066, 1.2827, 1.3296, 1.2005,
            1.346, 1.3757, 1.3956, 1.2905, 1.3199, 1.3887, 1.4658, 1.4826, 1.4502, 1.8708, 1.5356,
            1.4577, 1.7803, 1.6661, 1.7878, 1.3137, 1.3788, 1.3826, 1.4543, 1.6683, 1.6606, 1.5935,
            1.7119, 1.7599, 1.0642, 1.1124, 0.9832, 1.0629, 0.9228, 1.0681, 0.9127, 1.0635, 1.0557,
            0.1022, 0.2061, 0.3895, 0.8035, 0.7239, 0.7278, 0.7899, 0.6933, 0.2681, 0.393, 0.7109,
            0.729, 0.7412, 0.8742, 0.7118, 0.3665, 0.8132, 0.7894, 0.8763, 0.8868, 0.7589, 0.7858,
            0.714, 0.7966, 0.8288, 0.8542, 0.7095, 0.5959, 0.6213, 0.5612, 0.4604, 0.5065, 0.47,
            0.3502, 0.3097, 0.2712,
        ]
        .into_iter()
        .collect();
        m_exp.set_labels(vec![
            "Mouse".to_owned(),
            "Bovine".to_owned(),
            "Lemur".to_owned(),
            "Tarsier".to_owned(),
            "Squir Monk".to_owned(),
            "Jpn Macaq".to_owned(),
            "Rhesus Mac".to_owned(),
            "Crab-E.Mac".to_owned(),
            "BarbMacaq".to_owned(),
            "Gibbon".to_owned(),
            "Orang".to_owned(),
            "Gorilla".to_owned(),
            "Chimp".to_owned(),
            "Human".to_owned(),
        ]);

        assert_eq!(matrix, m_exp);
    }

    #[test]
    fn test_label_parsing() {
        assert_eq!(
            parse_label("1234 67890 0.12345", PhylipDialect::Strict).unwrap(),
            ("1234 67890", "0.12345")
        );
        assert_eq!(
            parse_label("1234 67890 0.12345", PhylipDialect::Relaxed).unwrap(),
            ("1234", "67890 0.12345")
        );
        assert_eq!(
            parse_label("1234       0.12345", PhylipDialect::Strict).unwrap(),
            ("1234", "0.12345")
        );
        assert_eq!(
            parse_label("1234567 0.12345", PhylipDialect::Strict).unwrap(),
            ("1234567 0.", "12345")
        );
        assert_eq!(
            parse_label("1234567 0.12345", PhylipDialect::Relaxed).unwrap(),
            ("1234567", "0.12345")
        );
        assert!(parse_label("12345 0.1", PhylipDialect::Strict).is_err());
    }
}
