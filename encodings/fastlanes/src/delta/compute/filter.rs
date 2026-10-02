// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::Delta as FastLanesDelta;
use fastlanes::FastLanes;
use fastlanes::Transpose;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::filter::FilterKernel;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Delta;
use crate::delta::array::DeltaArrayExt;
use crate::delta::array::DeltaArraySlotsExt;
use crate::delta::array::delta_decompress::decode_chunk;

/// Above this density, decoding the whole array and filtering the result is as fast as decoding
/// chunk by chunk and gathering, and the generic path is simpler.
const MAX_FUSED_DENSITY: f64 = 0.5;

/// Gathers the selected values one chunk at a time, decoding only the chunks that hold a selected
/// value and never materializing the full decoded array.
impl FilterKernel for Delta {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Mask::Values(values) = mask else {
            return Ok(None);
        };
        if values.density() > MAX_FUSED_DENSITY {
            return Ok(None);
        }

        let ptype = array.dtype().as_ptype();
        let validity = array.validity()?.filter(mask)?;
        let filtered = match_each_unsigned_integer_ptype!(ptype.to_unsigned(), |U| {
            const LANES: usize = U::LANES;
            let buffer = gather::<U, LANES>(array, values.indices(), ctx)?;
            PrimitiveArray::new(buffer, validity)
        });
        Ok(Some(filtered.reinterpret_cast(ptype).into_array()))
    }
}

fn gather<U, const LANES: usize>(
    array: ArrayView<'_, Delta>,
    indices: &[usize],
    ctx: &mut ExecutionCtx,
) -> VortexResult<Buffer<U>>
where
    U: NativePType + FastLanesDelta + Transpose,
{
    let bases = array
        .bases()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .reinterpret_cast(U::PTYPE);
    let deltas = array
        .deltas()
        .clone()
        .execute::<PrimitiveArray>(ctx)?
        .reinterpret_cast(U::PTYPE);
    let (bases, deltas) = (bases.as_slice::<U>(), deltas.as_slice::<U>());

    let offset = array.offset();
    let mut output = BufferMut::<U>::with_capacity(indices.len());
    let mut transposed = [U::default(); 1024];
    let mut values = [U::default(); 1024];
    let mut decoded_chunk = None;
    // Indices are sorted, so each chunk is decoded at most once.
    for &index in indices {
        let position = offset + index;
        let chunk = position / 1024;
        if decoded_chunk != Some(chunk) {
            decode_chunk::<U, LANES>(bases, deltas, chunk, &mut transposed, &mut values);
            decoded_chunk = Some(chunk);
        }
        output.push(values[position % 1024]);
    }
    Ok(output.freeze())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::Delta;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case::sparse_scattered(Mask::from_indices(3000, (0..3000).step_by(97)))]
    #[case::one_run(Mask::from_slices(3000, vec![(1020, 1100)]))]
    #[case::last_chunk(Mask::from_indices(3000, [2047, 2048, 2999]))]
    fn filter_matches_decoded(#[case] mask: Mask) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_option_iter(
            (0..3000i64).map(|v| (v % 11 != 0).then_some(v * 7 - 9_000)),
        );
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?.into_array();

        let actual = delta.filter(mask.clone())?.execute::<PrimitiveArray>(&mut ctx)?;
        let expected = primitive.into_array().filter(mask)?.execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn filter_on_slice() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter((0..3000u32).map(|v| v * 3));
        let delta = Delta::try_from_primitive_array(&primitive, &mut ctx)?
            .into_array()
            .slice(700..2900)?;
        let mask = Mask::from_indices(delta.len(), (0..delta.len()).step_by(13));

        let actual = delta.filter(mask.clone())?.execute::<PrimitiveArray>(&mut ctx)?;
        let expected = primitive
            .into_array()
            .slice(700..2900)?
            .filter(mask)?
            .execute::<PrimitiveArray>(&mut ctx)?;
        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }
}
