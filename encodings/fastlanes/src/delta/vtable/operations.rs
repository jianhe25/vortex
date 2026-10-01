// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ProbeState;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::Delta;
use crate::FL_CHUNK_SIZE;
use crate::delta::DeltaArrayExt;

/// The last decoded 1,024-element chunk, retained so reads in the same chunk skip the decode.
#[derive(Default)]
pub struct DeltaProbeState {
    chunk: Option<(usize, ArrayRef)>,
}

impl OperationsVTable<Delta> for Delta {
    type ProbeState = DeltaProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, Delta>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let Some(retained) = state.retained() else {
            // A one-off read decodes just this element.
            return array
                .array()
                .slice(index..index + 1)?
                .execute::<PrimitiveArray>(ctx)?
                .into_array()
                .execute_scalar(0, ctx);
        };
        // Every element of a chunk is the running sum of its lane's deltas, so decoding one
        // costs as much as decoding the chunk. Decode the whole chunk once and keep it.
        let physical = array.offset() + index;
        let chunk = physical / FL_CHUNK_SIZE;
        let chunk_start = (chunk * FL_CHUNK_SIZE).saturating_sub(array.offset());
        if !matches!(&retained.chunk, Some((kept, _)) if *kept == chunk) {
            let chunk_end = ((chunk + 1) * FL_CHUNK_SIZE - array.offset()).min(array.len());
            let decoded = array
                .array()
                .slice(chunk_start..chunk_end)?
                .execute::<PrimitiveArray>(ctx)?
                .into_array();
            retained.chunk = Some((chunk, decoded));
        }
        let (_, decoded) = retained
            .chunk
            .as_ref()
            .ok_or_else(|| vortex_err!("Delta chunk was just decoded"))?;
        decoded.execute_scalar(index - chunk_start, ctx)
    }

    fn scalar_at(
        array: ArrayView<'_, Delta>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::compute::conformance::binary_numeric::test_binary_numeric_array;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::Delta;
    use crate::DeltaArray;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = array_session();
        crate::initialize(&session);
        session
    });

    fn da(array: &PrimitiveArray) -> DeltaArray {
        Delta::try_from_primitive_array(array, &mut SESSION.create_execution_ctx())
            .vortex_expect("Delta array construction should succeed")
    }

    /// A retained probe keeps one decoded chunk; reads that cross chunks, come back, and go
    /// through a sliced offset all agree with one-off reads, nulls included.
    #[test]
    fn probe_reads_across_chunks_and_offsets() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let expected: Vec<Option<u32>> = (0u32..2500)
            .map(|i| (i % 7 != 3).then_some(i * 3))
            .collect();
        let delta = da(&PrimitiveArray::from_option_iter(expected.iter().copied())).into_array();
        let sliced = delta.slice(1000..2200)?;
        let mut whole = delta.repeated_probe();
        let mut part = sliced.repeated_probe();
        for index in [0, 1023, 1024, 2499, 1024, 5, 2048, 2047, 3] {
            let scalar = Scalar::from(expected[index]);
            assert_eq!(delta.execute_scalar(index, &mut ctx)?, scalar);
            assert_eq!(whole.execute_scalar(index, &mut ctx)?, scalar);
        }
        for index in [0, 23, 24, 1199, 1048, 1047, 0] {
            let scalar = Scalar::from(expected[1000 + index]);
            assert_eq!(sliced.execute_scalar(index, &mut ctx)?, scalar);
            assert_eq!(part.execute_scalar(index, &mut ctx)?, scalar);
        }
        assert!(part.execute_scalar(1200, &mut ctx).is_err());
        Ok(())
    }

    #[test]
    fn test_slice_non_jagged_array_first_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(10..250).unwrap();
        let expected = PrimitiveArray::from_iter(10u32..250).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_second_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_two() {
        let delta = da(&(0u32..2048).collect());

        let actual = delta.slice(1000..1048).unwrap();
        let expected = PrimitiveArray::from_iter(1000u32..1048).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_four() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(2040..2050).unwrap();
        let expected = PrimitiveArray::from_iter(2040u32..2050).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_whole() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(0..4096).unwrap();
        let expected = PrimitiveArray::from_iter(0u32..4096).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_non_jagged_array_empty() {
        let delta = da(&(0u32..4096).collect());

        let actual = delta.slice(0..0).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(4096..4096).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(1024..1024).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_jagged_array_second_chunk_of_two() {
        let delta = da(&(0u32..2000).collect());

        let actual = delta.slice(1024 + 10..1024 + 250).unwrap();
        let expected = PrimitiveArray::from_iter((1024 + 10u32)..(1024 + 250)).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_jagged_array_empty() {
        let delta = da(&(0u32..4000).collect());

        let actual = delta.slice(0..0).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(4000..4000).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());

        let actual = delta.slice(1024..1024).unwrap();
        let expected = PrimitiveArray::from_iter(Vec::<u32>::new()).into_array();
        assert_arrays_eq!(actual, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(10..1013).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![10u32, 11]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(1034..1050).unwrap();
        let sliced_again = sliced.slice(0..2).unwrap();

        let expected = PrimitiveArray::from_iter(vec![1034u32, 1035]).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_non_jagged() {
        let delta = da(&(0u32..2048).collect());

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_jagged() {
        let delta = da(&(0u32..2000).collect());

        let sliced = delta.slice(1010..1050).unwrap();
        let sliced_again = sliced.slice(5..20).unwrap();

        let expected = PrimitiveArray::from_iter(1015u32..1030).into_array();
        assert_arrays_eq!(sliced_again, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    fn test_scalar_at_non_jagged_array() {
        let delta = da(&(0u32..2048).collect()).into_array();

        let expected = PrimitiveArray::from_iter(0u32..2048).into_array();
        assert_arrays_eq!(delta, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_non_jagged_array_oob() {
        let delta = da(&(0u32..2048).collect()).into_array();
        delta
            .execute_scalar(2048, &mut SESSION.create_execution_ctx())
            .unwrap();
    }
    #[test]
    fn test_scalar_at_jagged_array() {
        let delta = da(&(0u32..2000).collect()).into_array();

        let expected = PrimitiveArray::from_iter(0u32..2000).into_array();
        assert_arrays_eq!(delta, expected, &mut SESSION.create_execution_ctx());
    }

    #[test]
    #[should_panic]
    fn test_scalar_at_jagged_array_oob() {
        let delta = da(&(0u32..2000).collect()).into_array();
        delta
            .execute_scalar(2000, &mut SESSION.create_execution_ctx())
            .unwrap();
    }

    #[rstest]
    // Basic delta arrays
    #[case::delta_u32((0u32..100).collect())]
    #[case::delta_u64((0..100).map(|i| i as u64 * 10).collect())]
    // Large arrays (multiple chunks)
    #[case::delta_large_u32((0u32..2048).collect())]
    #[case::delta_large_u64((0u64..2048).collect())]
    // Single element
    #[case::delta_single(PrimitiveArray::new(buffer![42u32], Validity::NonNullable))]
    // Signed inputs (added with signed-delta support).
    #[case::delta_i32_crossing_zero((-100i32..100).collect())]
    #[case::delta_i64_negative((0i64..100).map(|i| -i * 10).collect())]
    #[case::delta_large_i32((-1024i32..1024).collect())]
    #[case::delta_single_negative(PrimitiveArray::new(buffer![-42i32], Validity::NonNullable))]
    fn test_delta_consistency(#[case] array: PrimitiveArray) {
        test_array_consistency(
            &da(&array).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
    }

    #[rstest]
    #[case::delta_u8_basic(PrimitiveArray::new(buffer![1u8, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u16_basic(PrimitiveArray::new(buffer![1u16, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u32_basic(PrimitiveArray::new(buffer![1u32, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u64_basic(PrimitiveArray::new(buffer![1u64, 1, 1, 1, 1], Validity::NonNullable))]
    #[case::delta_u32_large(PrimitiveArray::new(buffer![1u32; 100], Validity::NonNullable))]
    #[case::delta_i8_basic(PrimitiveArray::new(buffer![-1i8, -1, -1, -1, -1], Validity::NonNullable))]
    #[case::delta_i32_basic(PrimitiveArray::new(buffer![-1i32, -1, -1, -1, -1], Validity::NonNullable))]
    fn test_delta_binary_numeric(#[case] array: PrimitiveArray) {
        test_binary_numeric_array(
            &da(&array).into_array(),
            &mut SESSION.create_execution_ctx(),
        );
    }
}
