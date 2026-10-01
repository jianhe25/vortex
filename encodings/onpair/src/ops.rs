// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::OnPairSlots;
use crate::array::dict_view;
use crate::decode::collect_widened;

impl OperationsVTable<OnPair> for OnPair {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, OnPair>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        // The validity is the validity slot, which a repeated read resolves once.
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        // A row owns a variable-length run of the flat `codes` stream; the per-row
        // `codes_offsets` boundaries map the row index to that run. Read just this row's two
        // boundaries and its codes through their probes, so a repeated read keeps the children's
        // preparation between rows, and decode only that run: never the whole column.
        let mut codes_offsets = state
            .slot(OnPairSlots::CODES_OFFSETS)?
            .ok_or_else(|| vortex_err!("OnPair codes_offsets slot is missing"))?;
        let row_start = code_boundary(codes_offsets.execute_scalar(index, ctx)?, index)?;
        let row_end = code_boundary(codes_offsets.execute_scalar(index + 1, ctx)?, index + 1)?;
        vortex_ensure!(
            row_start <= row_end,
            "OnPair codes_offsets must be nondecreasing, got {row_start} > {row_end} at row {index}"
        );

        vortex_ensure!(
            row_end <= array.codes().len(),
            "OnPair codes_offsets[{}] is {row_end}, past the {} codes",
            index + 1,
            array.codes().len()
        );

        // A one-off read decodes the run in bulk. A repeated read goes code by code through
        // the retained codes probe, which keeps the codes child's own preparation between rows
        // instead of executing a fresh slice of it for each.
        let codes: Buffer<u16> = if state.retained().is_none() {
            collect_widened::<u16>(&array.codes().slice(row_start..row_end)?, ctx)?
        } else {
            let mut codes_probe = state
                .slot(OnPairSlots::CODES)?
                .ok_or_else(|| vortex_err!("OnPair codes slot is missing"))?;
            (row_start..row_end)
                .map(|code_index| {
                    codes_probe
                        .execute_scalar(code_index, ctx)?
                        .as_primitive()
                        .as_opt::<u16>()
                        .flatten()
                        .ok_or_else(|| {
                            vortex_err!("OnPair codes[{code_index}] is null or not a u16")
                        })
                })
                .collect::<VortexResult<Vec<u16>>>()?
                .into()
        };
        let dict = dict_view(array, ctx)?;

        // The per-row decoded length is recorded in the `uncompressed_lengths`
        // child, so read it directly instead of asking the decoder to compute it.
        let len = state
            .slot(OnPairSlots::UNCOMPRESSED_LENGTHS)?
            .ok_or_else(|| vortex_err!("OnPair uncompressed_lengths slot is missing"))?
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_opt::<usize>()
            .flatten()
            .ok_or_else(|| {
                vortex_err!("OnPair uncompressed_lengths[{index}] is null, negative, or too large")
            })?;
        // The stored length controls allocation; each code emits 1 to MAX_TOKEN_SIZE bytes.
        vortex_ensure!(
            codes.len() <= len && len <= codes.len().saturating_mul(onpair::MAX_TOKEN_SIZE),
            "OnPair row {index} recorded length {len} is impossible for {} codes",
            codes.len()
        );
        let mut buf: Vec<u8> = Vec::with_capacity(len);
        let written =
            match onpair::try_decode_into(codes.as_slice(), dict, buf.spare_capacity_mut()) {
                Ok(written) => written,
                Err(_) => vortex_bail!("OnPair row {index} exceeds its recorded length"),
            };
        vortex_ensure!(
            written == len,
            "OnPair row {index} decoded {written} bytes, recorded {len}"
        );
        // SAFETY: `try_decode_into` initialised exactly `written` bytes.
        unsafe { buf.set_len(written) };
        Ok(varbin_scalar(ByteBuffer::from(buf), array.dtype()))
    }

    fn scalar_at(
        array: ArrayView<'_, OnPair>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

/// A `codes_offsets` boundary as a code index.
fn code_boundary(scalar: Scalar, index: usize) -> VortexResult<usize> {
    scalar
        .as_primitive()
        .as_opt::<usize>()
        .flatten()
        .ok_or_else(|| vortex_err!("OnPair codes_offsets[{index}] is null, negative, or too large"))
}
