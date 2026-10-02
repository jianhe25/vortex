// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::IntegerPType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::CompareKernel;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::FSST;
use crate::FSSTArrayExt;
use crate::FSSTArraySlotsExt;
impl CompareKernel for FSST {
    fn compare(
        lhs: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        match rhs.as_constant() {
            Some(constant) => compare_fsst_constant(lhs, &constant, operator, ctx),
            // Otherwise, fall back to the default comparison behavior.
            _ => Ok(None),
        }
    }
}

/// Specialized compare function implementation used when performing against a constant
fn compare_fsst_constant(
    left: ArrayView<'_, FSST>,
    right: &Scalar,
    operator: CompareOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let is_rhs_empty = match right.dtype() {
        DType::Binary(_) => right
            .as_binary()
            .is_empty()
            .vortex_expect("RHS should not be null"),
        DType::Utf8(_) => right
            .as_utf8()
            .is_empty()
            .vortex_expect("RHS should not be null"),
        _ => vortex_bail!("VarBinArray can only have type of Binary or Utf8"),
    };
    if is_rhs_empty {
        let buffer = match operator {
            // Every possible value is gte ""
            CompareOperator::Gte => BitBuffer::new_set(left.len()),
            // No value is lt ""
            CompareOperator::Lt => BitBuffer::new_unset(left.len()),
            _ => left
                .uncompressed_lengths()
                .binary(
                    ConstantArray::new(
                        Scalar::zero_value(left.uncompressed_lengths().dtype()),
                        left.uncompressed_lengths().len(),
                    )
                    .into_array(),
                    operator.into(),
                )?
                .execute(ctx)?,
        };

        return Ok(Some(
            BoolArray::new(
                buffer,
                left.array()
                    .validity()?
                    .union_nullability(right.dtype().nullability()),
            )
            .into_array(),
        ));
    }

    // The following section only supports Eq/NotEq
    if !matches!(operator, CompareOperator::Eq | CompareOperator::NotEq) {
        return Ok(None);
    }

    let needle: &[u8] = match right.dtype() {
        DType::Utf8(_) => right
            .as_utf8()
            .value()
            .vortex_expect("Expected non-null scalar")
            .as_str()
            .as_bytes(),
        DType::Binary(_) => right
            .as_binary()
            .value()
            .vortex_expect("Expected non-null scalar")
            .as_slice(),
        _ => unreachable!("FSSTArray can only have string or binary data type"),
    };

    // FSST encodes greedily with a fixed symbol table, so a value's codes are a function of the
    // value alone and two values are equal exactly when their codes are. Compressing the literal
    // once lets every row be compared in the compressed domain.
    let encoded_needle = left.compressor().compress(needle);

    let uncompressed_lengths = left
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let codes_offsets = left.codes_offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let codes_bytes = left.codes_bytes().as_slice();

    let eq = match_each_integer_ptype!(uncompressed_lengths.ptype(), |L| {
        match_each_integer_ptype!(codes_offsets.ptype(), |O| {
            eq_codes_to_constant::<L, O>(
                uncompressed_lengths.as_slice::<L>(),
                codes_offsets.as_slice::<O>(),
                codes_bytes,
                needle.len(),
                &encoded_needle,
            )
        })
    });
    let buffer = match operator {
        CompareOperator::Eq => eq,
        CompareOperator::NotEq => !eq,
        _ => unreachable!("operator is Eq or NotEq"),
    };

    Ok(Some(
        BoolArray::new(
            buffer,
            left.array()
                .validity()?
                .union_nullability(right.dtype().nullability()),
        )
        .into_array(),
    ))
}

/// Whether each row equals the literal whose FSST encoding is `encoded_needle`.
///
/// Rows whose uncompressed length or code length differ from the literal's are rejected before
/// any code bytes are read. Null rows carry an uncompressed length of zero and possibly empty
/// code ranges; their answers are masked out by validity.
fn eq_codes_to_constant<L: IntegerPType, O: IntegerPType>(
    uncompressed_lengths: &[L],
    codes_offsets: &[O],
    codes_bytes: &[u8],
    needle_len: usize,
    encoded_needle: &[u8],
) -> BitBuffer {
    BitBuffer::collect_bool(uncompressed_lengths.len(), |idx| {
        // SAFETY: `collect_bool` yields idx < len, and an FSST array has len + 1 code offsets.
        let uncompressed_len: usize = unsafe { uncompressed_lengths.get_unchecked(idx) }.as_();
        if uncompressed_len != needle_len {
            return false;
        }
        let start: usize = unsafe { codes_offsets.get_unchecked(idx) }.as_();
        let end: usize = unsafe { codes_offsets.get_unchecked(idx + 1) }.as_();
        end.wrapping_sub(start) == encoded_needle.len()
            && codes_bytes
                .get(start..end)
                .is_some_and(|codes| codes == encoded_needle)
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_compare_fsst() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let lhs = VarBinArray::from_iter(
            [
                Some("hello"),
                None,
                Some("world"),
                None,
                Some("this is a very long string"),
            ],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let compressor = fsst_train_compressor(&lhs, &mut ctx)?;
        let lhs = fsst_compress(&lhs, &compressor, &mut ctx)?;

        let rhs = ConstantArray::new("world", lhs.len());

        // Ensure fastpath for Eq exists, and returns correct answer
        let equals = lhs
            .clone()
            .into_array()
            .binary(rhs.clone().into_array(), Operator::Eq)?
            .execute::<BoolArray>(&mut ctx)?;

        assert_eq!(equals.dtype(), &DType::Bool(Nullability::Nullable));

        assert_arrays_eq!(
            &equals,
            &BoolArray::from_iter([Some(false), None, Some(true), None, Some(false)]),
            &mut ctx
        );

        // Ensure fastpath for Eq exists, and returns correct answer
        let not_equals = lhs
            .clone()
            .into_array()
            .binary(rhs.into_array(), Operator::NotEq)?
            .execute::<BoolArray>(&mut ctx)?;

        assert_eq!(not_equals.dtype(), &DType::Bool(Nullability::Nullable));
        assert_arrays_eq!(
            &not_equals,
            &BoolArray::from_iter([Some(true), None, Some(false), None, Some(true)]),
            &mut ctx
        );

        // Ensure null constants are handled correctly.
        let null_rhs =
            ConstantArray::new(Scalar::null(DType::Utf8(Nullability::Nullable)), lhs.len());
        let equals_null = lhs
            .clone()
            .into_array()
            .binary(null_rhs.clone().into_array(), Operator::Eq)?;
        assert_arrays_eq!(
            &equals_null,
            &BoolArray::from_iter([None::<bool>, None, None, None, None]),
            &mut ctx
        );

        let noteq_null = lhs
            .into_array()
            .binary(null_rhs.into_array(), Operator::NotEq)?;
        assert_arrays_eq!(
            &noteq_null,
            &BoolArray::from_iter([None::<bool>, None, None, None, None]),
            &mut ctx
        );
        Ok(())
    }

    #[rstest]
    #[case::present("abcabc", 0..9)]
    #[case::same_length_absent("abcabd", 0..9)]
    #[case::prefix_of_values("abc", 0..9)]
    #[case::longer_than_values("abcabcabcabcabc", 0..9)]
    #[case::sliced("abcabc", 3..8)]
    #[cfg_attr(miri, ignore)]
    fn test_compare_fsst_eq_matches_uncompressed(
        #[case] needle: &str,
        #[case] range: std::ops::Range<usize>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = VarBinArray::from_iter(
            [
                Some("abcabc"),
                Some("abcabd"),
                None,
                Some("abc"),
                Some("abcabc"),
                Some("xyzxyz"),
                Some("abcabcabc"),
                Some("abcabc"),
                Some(""),
            ],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let compressor = fsst_train_compressor(&values, &mut ctx)?;
        let fsst = fsst_compress(&values, &compressor, &mut ctx)?
            .into_array()
            .slice(range.clone())?;
        let values = values.slice(range)?;

        for op in [Operator::Eq, Operator::NotEq] {
            let rhs = ConstantArray::new(needle, fsst.len()).into_array();
            let actual = fsst
                .clone()
                .binary(rhs.clone(), op)?
                .execute::<BoolArray>(&mut ctx)?;
            let expected = values
                .clone()
                .binary(rhs, op)?
                .execute::<BoolArray>(&mut ctx)?;
            assert_arrays_eq!(&actual, &expected, &mut ctx);
        }
        Ok(())
    }
}
