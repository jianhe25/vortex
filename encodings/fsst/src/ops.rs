// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::FSST;
use crate::FSSTSlots;

impl OperationsVTable<FSST> for FSST {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, FSST>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        // The validity is the codes validity slot, which a repeated read resolves once.
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        // The row's codes are the bytes between its two offsets. Reading the offsets through
        // their probe rather than through a rebuilt codes array lets a compressed offsets child
        // keep its preparation between rows.
        let mut offsets = state
            .slot(FSSTSlots::CODES_OFFSETS)?
            .ok_or_else(|| vortex_err!("FSST codes offsets slot is missing"))?;
        let start = usize::try_from(&offsets.execute_scalar(index, ctx)?)?;
        let end = usize::try_from(&offsets.execute_scalar(index + 1, ctx)?)?;
        let codes = array
            .codes_bytes()
            .as_slice()
            .get(start..end)
            .ok_or_else(|| vortex_err!("FSST codes offsets {start}..{end} are out of bounds"))?;

        let decoded_buffer = ByteBuffer::from(array.decompressor().decompress(codes));
        Ok(varbin_scalar(decoded_buffer, array.dtype()))
    }

    fn scalar_at(
        array: ArrayView<'_, FSST>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
