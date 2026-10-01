// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::aggregate_fn::fns::min_max::MinMaxResult;
use crate::arrays::Primitive;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;

/// Value-range evidence produced by direct core validation or the mask index producer.
///
/// The enclosing owner binds this private token to its immutable input. Aggregate kernels,
/// transformations, imported metadata, and scalar seeds cannot mint or propagate it.
pub(super) struct VerifiedIntegerBounds {
    values: Option<MinMaxResult>,
}

impl VerifiedIntegerBounds {
    pub(super) fn from_producer(values: Option<MinMaxResult>) -> Self {
        Self { values }
    }

    pub(super) fn validate(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        vortex_ensure!(
            array.dtype().is_int(),
            "Bounds validation requires an integer input"
        );
        let primitive = array.as_opt::<Primitive>().ok_or_else(|| {
            vortex_error::vortex_err!("Bounds validation requires a canonical primitive input")
        })?;
        vortex_ensure!(
            array.dtype().as_ptype() == primitive.ptype(),
            "Integer physical and logical types must match"
        );
        let validity = primitive.validity()?.execute_mask(array.len(), ctx)?;
        let values = match_each_integer_ptype!(primitive.ptype(), |T| {
            let values = primitive.as_slice::<T>();
            let mut bounds: Option<(T, T)> = None;
            let mut include = |value: T| {
                bounds = Some(bounds.map_or((value, value), |(min, max)| {
                    (min.min(value), max.max(value))
                }));
            };
            match &validity {
                Mask::AllTrue(_) => values.iter().copied().for_each(&mut include),
                Mask::AllFalse(_) => {}
                Mask::Values(mask) => mask.bit_buffer().for_each_set_index(|i| include(values[i])),
            }
            bounds.map(|(min, max)| MinMaxResult {
                min: Scalar::primitive(min, Nullability::NonNullable),
                max: Scalar::primitive(max, Nullability::NonNullable),
            })
        });
        Ok(Self { values })
    }

    pub(super) fn values(&self) -> Option<&MinMaxResult> {
        self.values.as_ref()
    }

    pub(super) fn fits(&self, dtype: &DType) -> bool {
        self.values
            .as_ref()
            .is_none_or(|bounds| bounds.min.cast(dtype).is_ok() && bounds.max.cast(dtype).is_ok())
    }
}
