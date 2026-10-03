// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_buffer::buffer;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::compute::conformance::filter::test_filter_conformance;
use crate::compute::conformance::mask::test_mask_conformance;
use crate::compute::conformance::search_sorted::rstest_reuse::apply;
use crate::compute::conformance::search_sorted::search_sorted_conformance;
use crate::compute::conformance::search_sorted::*;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSorted;
use crate::search_sorted::SearchSortedPrimitiveArray;
use crate::search_sorted::SearchSortedSide;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

#[apply(search_sorted_conformance)]
fn test_search_sorted_primitive(
    #[case] array: ArrayRef,
    #[case] value: i32,
    #[case] side: SearchSortedSide,
    #[case] expected: SearchResult,
) -> vortex_error::VortexResult<()> {
    let res = SearchSortedPrimitiveArray::<i32>::new(&array, &mut SESSION.create_execution_ctx())
        .search_sorted(&value, side)?;
    assert_eq!(res, expected);
    Ok(())
}

#[test]
fn test_mask_primitive_array() {
    test_mask_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::NonNullable).into_array(),
        &mut array_session().create_execution_ctx(),
    );
    test_mask_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllValid).into_array(),
        &mut array_session().create_execution_ctx(),
    );
    test_mask_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllInvalid).into_array(),
        &mut array_session().create_execution_ctx(),
    );
    test_mask_conformance(
        &PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4],
            Validity::Array(BoolArray::from_iter([true, false, true, false, true]).into_array()),
        )
        .into_array(),
        &mut array_session().create_execution_ctx(),
    );
}

#[test]
fn test_filter_primitive_array() {
    // Test various sizes
    test_filter_conformance(
        &PrimitiveArray::new(buffer![42i32], Validity::NonNullable).into_array(),
        &mut array_session().create_execution_ctx(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1], Validity::NonNullable).into_array(),
        &mut array_session().create_execution_ctx(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::NonNullable).into_array(),
        &mut array_session().create_execution_ctx(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4, 5, 6, 7], Validity::NonNullable).into_array(),
        &mut array_session().create_execution_ctx(),
    );

    // Test with validity
    test_filter_conformance(
        &PrimitiveArray::new(buffer![0, 1, 2, 3, 4], Validity::AllValid).into_array(),
        &mut array_session().create_execution_ctx(),
    );
    test_filter_conformance(
        &PrimitiveArray::new(
            buffer![0, 1, 2, 3, 4, 5],
            Validity::Array(
                BoolArray::from_iter([true, false, true, false, true, true]).into_array(),
            ),
        )
        .into_array(),
        &mut array_session().create_execution_ctx(),
    );
}

#[test]
fn test_map_each_with_validity() -> vortex_error::VortexResult<()> {
    let f = |(v, valid): (i32, bool)| if valid { v * 10 } else { -1 };
    let cases = [
        (Validity::NonNullable, vec![10, 20, 30, 40]),
        (Validity::AllValid, vec![10, 20, 30, 40]),
        (Validity::AllInvalid, vec![-1, -1, -1, -1]),
        (
            Validity::Array(BoolArray::from_iter([true, false, false, true]).into_array()),
            vec![10, -1, -1, 40],
        ),
    ];
    for (validity, expected) in cases {
        let mut ctx = SESSION.create_execution_ctx();
        let mapped = PrimitiveArray::new(buffer![1i32, 2, 3, 4], validity)
            .map_each_with_validity::<i32, i32, _>(&mut ctx, f)?;
        assert_eq!(mapped.as_slice::<i32>(), expected.as_slice());
    }

    // An empty input maps to an empty output for every validity variant.
    for validity in [
        Validity::NonNullable,
        Validity::AllValid,
        Validity::AllInvalid,
    ] {
        let mut ctx = SESSION.create_execution_ctx();
        let mapped = PrimitiveArray::new(vortex_buffer::Buffer::<i32>::empty(), validity)
            .map_each_with_validity::<i32, i32, _>(&mut ctx, f)?;
        assert!(mapped.as_slice::<i32>().is_empty());
    }
    Ok(())
}
