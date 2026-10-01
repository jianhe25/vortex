// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::Dict;
use crate::ExecutionCtx;
use crate::ProbeState;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::dict::DictSlots;
use crate::scalar::Scalar;

impl OperationsVTable<Dict> for Dict {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Dict>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        // A null code is a null row; a null value is too. The codes and values probes are kept
        // by a repeated read, so a compressed codes child keeps its preparation between rows.
        let Some(code) = state
            .slot(DictSlots::CODES)?
            .ok_or_else(|| vortex_err!("Dict codes slot is missing"))?
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<usize>()
        else {
            return Ok(Scalar::null(array.dtype().clone()));
        };
        let value = state
            .slot(DictSlots::VALUES)?
            .ok_or_else(|| vortex_err!("Dict values slot is missing"))?
            .execute_scalar(code, ctx)?;
        // The values' dtype differs from the array's only by nullability.
        Scalar::try_new(array.dtype().clone(), value.into_value())
    }

    fn scalar_at(
        array: ArrayView<'_, Dict>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
