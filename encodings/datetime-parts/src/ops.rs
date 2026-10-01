// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::dtype::DType;
use vortex_array::extension::datetime::Timestamp;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::DateTimeParts;
use crate::array::DateTimePartsSlots;
use crate::timestamp;
use crate::timestamp::TimestampParts;

impl OperationsVTable<DateTimeParts> for DateTimeParts {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, DateTimeParts>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let DType::Extension(ext) = array.dtype().clone() else {
            vortex_panic!(
                "DateTimePartsArray must have extension dtype, found {}",
                array.dtype()
            );
        };

        let Some(options) = ext.metadata_opt::<Timestamp>() else {
            vortex_panic!(Compute: "must decode TemporalMetadata from extension metadata");
        };

        // The array's validity is its days', so a null days value is the null row.
        let days = state
            .slot(DateTimePartsSlots::DAYS)?
            .ok_or_else(|| vortex_err!("DateTimeParts days slot is missing"))?
            .execute_scalar(index, ctx)?;
        let Some(days) = days.as_primitive().as_::<i32>() else {
            return Ok(Scalar::null(DType::Extension(ext)));
        };
        let seconds: i32 = state
            .slot(DateTimePartsSlots::SECONDS)?
            .ok_or_else(|| vortex_err!("DateTimeParts seconds slot is missing"))?
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<i32>()
            .vortex_expect("seconds fits in i32");
        let subseconds: i32 = state
            .slot(DateTimePartsSlots::SUBSECONDS)?
            .ok_or_else(|| vortex_err!("DateTimeParts subseconds slot is missing"))?
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<i32>()
            .vortex_expect("subseconds fits in i32");

        let ts = timestamp::combine(
            TimestampParts {
                days,
                seconds,
                subseconds,
            },
            options.unit,
        );

        Ok(Scalar::extension::<Timestamp>(
            options.clone(),
            Scalar::primitive(ts, ext.storage_dtype().nullability()),
        ))
    }

    fn scalar_at(
        array: ArrayView<'_, DateTimeParts>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
