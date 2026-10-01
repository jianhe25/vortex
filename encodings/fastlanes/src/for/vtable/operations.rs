// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::FoR;
use crate::FL_CHUNK_SIZE;
use crate::r#for::array::FoRArrayExt;
use crate::r#for::array::FoRSlots;

impl OperationsVTable<FoR> for FoR {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, FoR>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        // The array's validity is its encoded child's, so a null encoded value is the null row
        // and the chunk's reference need not be read.
        let encoded = state
            .slot(FoRSlots::ENCODED)?
            .ok_or_else(|| vortex_err!("FoR encoded slot is missing"))?
            .execute_scalar(index, ctx)?;
        let encoded = encoded.as_primitive();
        let chunk = (usize::from(array.offset()) + index) / FL_CHUNK_SIZE;

        Ok(match_each_integer_ptype!(array.ptype(), |P| {
            match encoded.typed_value::<P>() {
                Some(value) => {
                    let reference = state
                        .slot(FoRSlots::REFERENCES)?
                        .ok_or_else(|| vortex_err!("FoR references slot is missing"))?
                        .execute_scalar(chunk, ctx)?;
                    let reference = reference
                        .as_primitive()
                        .typed_value::<P>()
                        .vortex_expect("FoRArray Reference value cannot be null");
                    Scalar::primitive::<P>(
                        value.wrapping_add(reference),
                        array.dtype().nullability(),
                    )
                }
                None => Scalar::null(array.dtype().clone()),
            }
        }))
    }

    fn scalar_at(
        array: ArrayView<'_, FoR>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

#[cfg(test)]
mod test {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::FoRData;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn for_scalar_at() {
        let mut ctx = SESSION.create_execution_ctx();
        let for_arr = FoRData::encode(
            PrimitiveArray::from_iter([-100, 1100, 1500, 1900]),
            &mut ctx,
        )
        .unwrap();
        let expected = PrimitiveArray::from_iter([-100, 1100, 1500, 1900]);
        assert_arrays_eq!(for_arr, expected, &mut ctx);
    }

    #[test]
    fn for_probe_reads_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let expected = [Some(-100i32), None, Some(1500), None];
        let for_arr =
            FoRData::encode(PrimitiveArray::from_option_iter(expected), &mut ctx)?.into_array();
        let mut probe = for_arr.repeated_probe();
        for (index, value) in expected.into_iter().enumerate() {
            assert_eq!(
                for_arr.execute_scalar(index, &mut ctx)?,
                Scalar::from(value)
            );
            assert_eq!(probe.execute_scalar(index, &mut ctx)?, Scalar::from(value));
        }
        Ok(())
    }
}
