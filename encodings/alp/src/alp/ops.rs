// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPFloat;
use crate::ALPSlots;
use crate::match_each_alp_float_ptype;

impl OperationsVTable<ALP> for ALP {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, ALP>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        // The array's validity is its encoded child's, so a null encoded value is the null row,
        // whatever the patches hold there.
        let encoded = state
            .slot(ALPSlots::ENCODED)?
            .ok_or_else(|| vortex_err!("ALP encoded slot is missing"))?
            .execute_scalar(index, ctx)?;
        if encoded.is_null() {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        if let Some(patches) = array.patches()
            && let Some(patch) = patches.get_patched(index)?
        {
            return patch.cast(array.dtype());
        }

        Ok(match_each_alp_float_ptype!(array.dtype().as_ptype(), |T| {
            let encoded: <T as ALPFloat>::ALPInt =
                (&encoded).try_into().vortex_expect("invalid ALPInt");
            Scalar::primitive(
                <T as ALPFloat>::decode_single(encoded, array.exponents()),
                array.dtype().nullability(),
            )
        }))
    }

    fn scalar_at(
        array: ArrayView<'_, ALP>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ALPArrayExt;
    use crate::alp_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    /// Nulls, encoded values and patched exceptions come back on the one-off and the retained
    /// path.
    #[test]
    fn probe_reads_nulls_values_and_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let expected = [
            Some(1.25f64),
            None,
            Some(1e300),
            Some(-3.5),
            None,
            Some(0.0),
        ];
        let array = PrimitiveArray::from_option_iter(expected);
        let encoded = alp_encode(array.as_view(), None, &mut ctx)?;
        assert!(encoded.patches().is_some());
        let encoded = encoded.into_array();
        let mut probe = encoded.repeated_probe();
        for index in [5, 2, 1, 0, 3, 4, 2] {
            let scalar = Scalar::from(expected[index]);
            assert_eq!(encoded.execute_scalar(index, &mut ctx)?, scalar);
            assert_eq!(probe.execute_scalar(index, &mut ctx)?, scalar);
        }
        Ok(())
    }
}
