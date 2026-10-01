// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ALPRD;
use crate::ALPRDArrayExt;
use crate::ALPRDSlots;

impl OperationsVTable<ALPRD> for ALPRD {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, ALPRD>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        // The array's validity is its left parts', so a null left code is the null row.
        let left_code = state
            .slot(ALPRDSlots::LEFT_PARTS)?
            .ok_or_else(|| vortex_err!("ALPRD left parts slot is missing"))?
            .execute_scalar(index, ctx)?;
        let Some(left_code) = left_code.as_primitive().as_::<u16>() else {
            return Ok(Scalar::null(array.dtype().clone()));
        };

        // The left value is either a code into the dictionary or an exception held in the
        // patches, whose non-null values mark exception positions.
        let maybe_patched_value = match array.left_parts_patches() {
            Some(patches) => patches.get_patched(index)?,
            None => None,
        };
        let left = match maybe_patched_value {
            Some(patched_value) => patched_value
                .as_primitive()
                .as_::<u16>()
                .vortex_expect("patched values must be non-null"),
            None => array.left_parts_dictionary()[left_code as usize],
        };
        let right = state
            .slot(ALPRDSlots::RIGHT_PARTS)?
            .ok_or_else(|| vortex_err!("ALPRD right parts slot is missing"))?
            .execute_scalar(index, ctx)?;

        // combine left and right values
        Ok(if array.dtype().as_ptype() == PType::F32 {
            let right: u32 = right.as_primitive().as_::<u32>().vortex_expect("non-null");
            let packed = f32::from_bits((left as u32) << array.right_bit_width() | right);
            Scalar::primitive(packed, array.dtype().nullability())
        } else {
            let right: u64 = right.as_primitive().as_::<u64>().vortex_expect("non-null");
            let packed = f64::from_bits(((left as u64) << array.right_bit_width()) | right);
            Scalar::primitive(packed, array.dtype().nullability())
        })
    }

    fn scalar_at(
        array: ArrayView<'_, ALPRD>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::scalar::Scalar;
    use vortex_session::VortexSession;

    use crate::ALPRDArrayExt;
    use crate::ALPRDFloat;
    use crate::RDEncoder;
    use crate::RDEncoderExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_slice<T: ALPRDFloat + NativePType>(#[case] a: T, #[case] b: T, #[case] outlier: T) {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter([a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(array.as_view());

        assert!(encoded.left_parts_patches().is_some());
        assert_arrays_eq!(encoded, array, &mut ctx);
    }

    #[rstest]
    #[case(0.1f32, 0.2f32, 3e25f32)]
    #[case(0.1f64, 0.2f64, 3e100f64)]
    fn test_scalar_at<T: ALPRDFloat + NativePType + Into<Scalar>>(
        #[case] a: T,
        #[case] b: T,
        #[case] outlier: T,
    ) {
        let mut ctx = SESSION.create_execution_ctx();
        let array = PrimitiveArray::from_iter([a, b, outlier]);
        let encoded = RDEncoder::new(&[a, b]).encode(array.as_view());
        assert!(encoded.left_parts_patches().is_some());
        assert_arrays_eq!(encoded, array, &mut ctx);
    }

    /// Nulls, dictionary-coded values and exceptions come back on the one-off and the retained
    /// path.
    #[test]
    fn probe_reads_nulls_values_and_exceptions() {
        let mut ctx = SESSION.create_execution_ctx();
        let (a, b, outlier) = (0.1f64, 0.2f64, 3e100f64);
        let expected = [Some(a), None, Some(outlier), Some(b), None];
        let array = PrimitiveArray::from_option_iter(expected);
        let encoded = RDEncoder::new(&[a, b]).encode(array.as_view()).into_array();
        assert!(encoded.as_::<crate::ALPRD>().left_parts_patches().is_some());
        let mut probe = encoded.repeated_probe();
        for index in [4, 2, 1, 0, 3, 2] {
            let scalar = Scalar::from(expected[index]);
            assert_eq!(encoded.execute_scalar(index, &mut ctx).unwrap(), scalar);
            assert_eq!(probe.execute_scalar(index, &mut ctx).unwrap(), scalar);
        }
    }

    #[test]
    fn nullable_scalar_at() {
        let mut ctx = SESSION.create_execution_ctx();
        let a = 0.1f64;
        let b = 0.2f64;
        let outlier = 3e100f64;
        let array = PrimitiveArray::from_option_iter([Some(a), Some(b), Some(outlier)]);
        let encoded = RDEncoder::new(&[a, b]).encode(array.as_view());
        assert!(encoded.left_parts_patches().is_some());
        assert_arrays_eq!(
            encoded,
            PrimitiveArray::from_option_iter([Some(a), Some(b), Some(outlier)]),
            &mut ctx
        );
    }
}
