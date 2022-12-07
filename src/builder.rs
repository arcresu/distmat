use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use thiserror::Error;

use crate::symmetric::Coordinates;
use crate::{DistMatrix, SquareMatrix};

#[derive(Error, Debug)]
pub enum DataError {
    #[error("missing entry for distance between '{0}' and  '{1}'")]
    Missing(String, String),

    #[error("duplicate entry for distance between '{0}' and  '{1}'")]
    Duplicate(String, String),
}

pub(crate) struct DistBuilder<D> {
    pub(crate) labels: Vec<String>,
    dists: BTreeMap<(u32, u32), D>,
    label_to_id: BTreeMap<String, u32>,
}

impl<D: Copy> DistBuilder<D> {
    pub fn new() -> Self {
        DistBuilder {
            labels: Vec::new(),
            dists: BTreeMap::new(),
            label_to_id: BTreeMap::new(),
        }
    }

    pub fn add<S: AsRef<str>>(
        &mut self,
        label1: S,
        label2: S,
        distance: D,
    ) -> Result<(), DataError> {
        let label1 = label1.as_ref();
        let label2 = label2.as_ref();

        let id1 = *self
            .label_to_id
            .entry(label1.to_owned())
            .or_insert_with(|| {
                self.labels.push(label1.to_owned());
                self.labels.len() as u32 - 1
            });

        let id2 = *self
            .label_to_id
            .entry(label2.to_owned())
            .or_insert_with(|| {
                self.labels.push(label2.to_owned());
                self.labels.len() as u32 - 1
            });

        match self.dists.entry((id1, id2)) {
            Entry::Vacant(e) => {
                e.insert(distance);
            }
            Entry::Occupied(_) => {
                return Err(DataError::Duplicate(label1.to_owned(), label2.to_owned()));
            }
        }

        Ok(())
    }
}

impl<D: Copy> TryFrom<DistBuilder<D>> for SquareMatrix<D> {
    type Error = DataError;

    fn try_from(builder: DistBuilder<D>) -> Result<SquareMatrix<D>, DataError> {
        let size = builder.labels.len();
        let mut data = Vec::with_capacity(size * size);
        for i in 0..size {
            for j in 0..size {
                let dist: D = *builder.dists.get(&(i as u32, j as u32)).ok_or_else(|| {
                    DataError::Missing(builder.labels[i].clone(), builder.labels[j].clone())
                })?;
                data.push(dist);
            }
        }
        Ok(SquareMatrix { data, size })
    }
}

impl<D: Copy> TryFrom<DistBuilder<D>> for DistMatrix<D> {
    type Error = DataError;

    fn try_from(builder: DistBuilder<D>) -> Result<DistMatrix<D>, DataError> {
        let size = builder.labels.len();
        let mut data = Vec::with_capacity(size * (size - 1) / 2);
        for (i, j) in Coordinates::new(size) {
            let dist1 = builder.dists.get(&(i as u32, j as u32));
            let dist2 = builder.dists.get(&(j as u32, i as u32));

            if dist1.is_some() && dist2.is_some() {
                return Err(DataError::Duplicate(
                    builder.labels[i].clone(),
                    builder.labels[j].clone(),
                ));
            }

            let dist: D = *dist1.or(dist2).ok_or_else(|| {
                DataError::Missing(builder.labels[i].clone(), builder.labels[j].clone())
            })?;
            data.push(dist);
        }
        Ok(DistMatrix { data, size })
    }
}

impl<S, D: Copy> FromIterator<(S, S, D)> for DistBuilder<D>
where
    S: AsRef<str>,
{
    fn from_iter<I: IntoIterator<Item = (S, S, D)>>(iter: I) -> Self {
        let mut builder = DistBuilder::<D>::new();
        for (label1, label2, distance) in iter {
            builder.add(label1, label2, distance).unwrap();
        }
        builder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_builder() {
        let mut builder = DistBuilder::<u32>::new();
        builder.add("A", "A", 0).unwrap();
        builder.add("A", "B", 5).unwrap();
        builder.add("A", "C", 1).unwrap();

        builder.add("C", "A", 1).unwrap();
        builder.add("C", "B", 4).unwrap();
        builder.add("C", "C", 0).unwrap();

        builder.add("B", "A", 5).unwrap();
        builder.add("B", "B", 0).unwrap();
        builder.add("B", "C", 4).unwrap();

        let m: SquareMatrix<u32> = builder.try_into().unwrap();
        let m2 = SquareMatrix::<u32>::from_pw_distances(&[1u32, 6, 2]);
        assert_eq!(m, m2);
    }

    #[test]
    fn test_square_builder_dup() {
        let mut builder = DistBuilder::<u32>::new();
        builder.add("A", "A", 0).unwrap();
        assert!(builder.add("A", "A", 5).is_err());
    }

    #[test]
    fn test_square_builder_incomplete() {
        let mut builder = DistBuilder::<u32>::new();
        builder.add("A", "A", 0).unwrap();
        builder.add("A", "B", 1).unwrap();
        assert!(TryInto::<SquareMatrix<u32>>::try_into(builder).is_err());
    }

    #[test]
    fn test_sym_builder() {
        let mut builder = DistBuilder::<u32>::new();
        builder.add("A", "B", 5).unwrap();
        builder.add("C", "A", 1).unwrap();
        builder.add("C", "B", 4).unwrap();

        let m: DistMatrix<u32> = builder.try_into().unwrap();
        let m2 = DistMatrix::<u32>::from_pw_distances(&[1u32, 6, 2]);
        assert_eq!(m, m2);
    }

    struct Dist {
        a: String,
        b: String,
        distance: u32,
    }

    #[test]
    fn test_from_iter() {
        let dists = vec![
            Dist {
                a: "A".to_string(),
                b: "B".to_string(),
                distance: 5,
            },
            Dist {
                a: "C".to_string(),
                b: "A".to_string(),
                distance: 1,
            },
            Dist {
                a: "C".to_string(),
                b: "B".to_string(),
                distance: 4,
            },
        ];

        let m: DistBuilder<u32> = dists.into_iter().map(|x| (x.a, x.b, x.distance)).collect();
        let m: DistMatrix<u32> = m.try_into().unwrap();
        let m2 = DistMatrix::<u32>::from_pw_distances(&[1u32, 6, 2]);
        assert_eq!(m, m2);
    }
}
