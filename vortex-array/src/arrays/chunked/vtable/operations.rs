// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::ProbeState;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Chunked;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::arrays::chunked::ChunkedSlots;
use crate::scalar::Scalar;

/// The chunk the previous read landed in, retained so a read in the same chunk skips the search
/// over the chunk offsets.
#[derive(Default)]
pub struct ChunkedProbeState {
    last_chunk: Option<usize>,
}

impl OperationsVTable<Chunked> for Chunked {
    type ProbeState = ChunkedProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, Chunked>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let offsets = array.chunk_offset_values();
        let (mut retained, mut children) = state.split();
        let chunk_index = match retained.as_deref().and_then(|state| state.last_chunk) {
            Some(last) if offsets[last] <= index && index < offsets[last + 1] => last,
            _ => {
                let (chunk_index, _) = array.find_chunk_idx(index)?;
                if let Some(state) = retained.as_mut() {
                    state.last_chunk = Some(chunk_index);
                }
                chunk_index
            }
        };
        // Chunks share the outer dtype, so a null row comes back from the chunk's probe as a
        // null of the right dtype; a repeated read keeps one probe per chunk it has visited.
        children
            .slot(ChunkedSlots::CHUNKS_OFFSET + chunk_index)?
            .ok_or_else(|| vortex_err!("Chunked chunk {chunk_index} slot is missing"))?
            .execute_scalar(index - offsets[chunk_index], ctx)
    }

    fn scalar_at(
        array: ArrayView<'_, Chunked>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ChunkedArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;

    fn chunked_array() -> ChunkedArray {
        ChunkedArray::try_new(
            vec![
                buffer![1u64, 2, 3].into_array(),
                buffer![4u64, 5, 6].into_array(),
                buffer![7u64, 8, 9].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap()
    }

    #[rstest]
    #[case::middle(2..5, &[3u64, 4, 5])]
    #[case::begin(1..3, &[2u64, 3])]
    #[case::aligned(3..6, &[4u64, 5, 6])]
    #[case::many_aligned(0..6, &[1u64, 2, 3, 4, 5, 6])]
    #[case::end(7..8, &[8u64])]
    #[case::exactly_end(6..9, &[7u64, 8, 9])]
    fn slice(#[case] range: Range<usize>, #[case] expected: &[u64]) {
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(
            chunked_array().slice(range).unwrap(),
            PrimitiveArray::from_iter(expected.iter().copied()),
            &mut ctx
        );
    }

    #[test]
    fn slice_empty() {
        let chunked = ChunkedArray::try_new(vec![], PType::U32.into()).unwrap();
        let sliced = chunked.slice(0..0).unwrap();

        assert!(sliced.is_empty());
    }

    /// Nulls come back through the chunk's probe, on the one-off and the retained path, and the
    /// retained chunk hint survives reads that leave and re-enter a chunk or cross an empty one.
    #[test]
    fn probe_reads_nulls_across_chunks() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_option_iter([Some(1u64), None]).into_array(),
                PrimitiveArray::from_option_iter(std::iter::empty::<Option<u64>>()).into_array(),
                PrimitiveArray::from_option_iter([None, Some(4u64)]).into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::Nullable),
        )?
        .into_array();
        let expected = [Some(1u64), None, None, Some(4)];
        let mut repeated = array.repeated_probe();
        for index in (0..expected.len()).rev().chain(0..expected.len()) {
            let scalar = Scalar::from(expected[index]);
            assert_eq!(array.execute_scalar(index, &mut ctx)?, scalar);
            assert_eq!(repeated.execute_scalar(index, &mut ctx)?, scalar);
        }
        assert!(repeated.execute_scalar(expected.len(), &mut ctx).is_err());
        Ok(())
    }

    /// Validity reads through one retained probe resolve each chunk's validity in its own kind:
    /// all valid, all null, and a validity array.
    #[test]
    fn probe_validity_across_chunk_validity_kinds() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_option_iter([Some(1u64), Some(2)]).into_array(),
                PrimitiveArray::from_option_iter([None::<u64>, None]).into_array(),
                PrimitiveArray::from_option_iter([None, Some(6u64)]).into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::Nullable),
        )?
        .into_array();
        let expected = [true, true, false, false, false, true];
        let mut repeated = array.repeated_probe();
        for index in (0..expected.len()).rev().chain(0..expected.len()) {
            assert_eq!(
                array.probe().execute_is_valid(index, &mut ctx)?,
                expected[index]
            );
            assert_eq!(repeated.execute_is_valid(index, &mut ctx)?, expected[index]);
            assert_eq!(
                repeated.execute_scalar(index, &mut ctx)?.is_null(),
                !expected[index]
            );
        }
        Ok(())
    }

    #[test]
    fn scalar_at_empty_children_both_sides() {
        let mut ctx = array_session().create_execution_ctx();
        let array = ChunkedArray::try_new(
            vec![
                Buffer::<u64>::empty().into_array(),
                Buffer::<u64>::empty().into_array(),
                buffer![1u64, 2].into_array(),
                Buffer::<u64>::empty().into_array(),
                Buffer::<u64>::empty().into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();
        assert_arrays_eq!(array, PrimitiveArray::from_iter([1u64, 2]), &mut ctx);
    }

    #[test]
    fn scalar_at_empty_children_trailing() {
        let mut ctx = array_session().create_execution_ctx();
        let array = ChunkedArray::try_new(
            vec![
                buffer![1u64, 2].into_array(),
                Buffer::<u64>::empty().into_array(),
                Buffer::<u64>::empty().into_array(),
                buffer![3u64, 4].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();
        assert_arrays_eq!(array, PrimitiveArray::from_iter([1u64, 2, 3, 4]), &mut ctx);
    }

    #[test]
    fn scalar_at_empty_children_leading() {
        let mut ctx = array_session().create_execution_ctx();
        let array = ChunkedArray::try_new(
            vec![
                Buffer::<u64>::empty().into_array(),
                Buffer::<u64>::empty().into_array(),
                buffer![1u64, 2].into_array(),
                buffer![3u64, 4].into_array(),
            ],
            DType::Primitive(PType::U64, Nullability::NonNullable),
        )
        .unwrap();
        assert_arrays_eq!(array, PrimitiveArray::from_iter([1u64, 2, 3, 4]), &mut ctx);
    }
}
