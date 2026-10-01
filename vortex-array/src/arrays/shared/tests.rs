// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::Canonical;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::arrays::SharedArray;
use crate::arrays::shared::SharedArrayExt;
use crate::hash::ArrayEq;
use crate::hash::EqMode;
use crate::scalar::Scalar;
use crate::validity::Validity;

#[test]
fn shared_array_caches_on_canonicalize() -> VortexResult<()> {
    let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
    let shared = SharedArray::new(array);

    let session = crate::array_session();
    let mut ctx = session.create_execution_ctx();

    let first = shared.get_or_compute(|source| source.clone().execute::<Canonical>(&mut ctx))?;

    // Second call should return cached without invoking the closure.
    let second = shared.get_or_compute(|_| panic!("should not execute twice"))?;

    assert!(first.array_eq(&second, EqMode::Value));

    Ok(())
}

/// A retained probe reads through the source until the array is materialised, then through the
/// cached result, passing nulls through either way.
#[test]
fn repeated_probe_follows_materialisation() -> VortexResult<()> {
    let session = crate::array_session();
    let mut ctx = session.create_execution_ctx();
    let source = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
    let shared = SharedArray::new(source);
    let array = shared.clone().into_array();

    let mut probe = array.repeated_probe();
    assert_eq!(probe.execute_scalar(0, &mut ctx)?, Scalar::from(Some(1i32)));
    assert!(probe.execute_scalar(1, &mut ctx)?.is_null());

    shared.get_or_compute(|source| source.clone().execute::<Canonical>(&mut ctx))?;
    assert_eq!(probe.execute_scalar(2, &mut ctx)?, Scalar::from(Some(3i32)));
    assert!(probe.execute_scalar(1, &mut ctx)?.is_null());
    assert_eq!(array.execute_scalar(0, &mut ctx)?, Scalar::from(Some(1i32)));
    Ok(())
}
