// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::sync::LazyLock;

use prost::Message;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::expr::display::ExprDisplay;
use crate::proto::expr as pb;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnRef;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;

/// Expression that represents a literal scalar value.
#[derive(Clone)]
pub struct Literal;

impl Literal {
    /// Binds `scalar` into a literal scalar fn.
    ///
    /// Boolean scalars reuse a shared, pre-built fn instead of allocating a new one.
    pub fn scalar_fn(scalar: Scalar) -> ScalarFnRef {
        match cached_bool_literal(&scalar) {
            Some(cached) => cached.scalar_fn.clone(),
            None => Literal.bind(scalar),
        }
    }

    /// Creates a literal expression for `scalar`.
    ///
    /// Boolean scalars reuse a shared, pre-built node instead of allocating a new one.
    pub fn expr(scalar: Scalar) -> Expression {
        match cached_bool_literal(&scalar) {
            Some(cached) => cached.expr.clone(),
            None => Literal.new_expr(scalar, []),
        }
    }

    /// Creates a bound literal expression for `scalar`.
    ///
    /// Boolean scalars reuse a shared, pre-built node instead of allocating a new one.
    pub fn bound_expr(scalar: Scalar) -> BoundExpression {
        match cached_bool_literal(&scalar) {
            Some(cached) => cached.bound_expr.clone(),
            None => Literal
                .try_new_bound_expr(scalar, [])
                .vortex_expect("literal expressions are always well-typed"),
        }
    }
}

/// The pre-built literal nodes for one boolean scalar.
struct BoolLiteral {
    scalar_fn: ScalarFnRef,
    expr: Expression,
    bound_expr: BoundExpression,
}

/// A boolean scalar takes one of only five values (`false` and `true` at either nullability, plus
/// the nullable null), so their literal nodes are built once and shared. Creating a boolean literal
/// is then a reference-count increment rather than a heap allocation for the erased scalar fn and
/// another for the empty child list.
static BOOL_LITERALS: LazyLock<[BoolLiteral; 5]> = LazyLock::new(|| {
    [
        Scalar::bool(false, Nullability::NonNullable),
        Scalar::bool(true, Nullability::NonNullable),
        Scalar::bool(false, Nullability::Nullable),
        Scalar::bool(true, Nullability::Nullable),
        Scalar::null(DType::Bool(Nullability::Nullable)),
    ]
    .map(|scalar| {
        let scalar_fn = Literal.bind(scalar);
        let expr = Expression::try_new(scalar_fn.clone(), [])
            .vortex_expect("literal expressions have no children");
        let bound_expr = BoundExpression::try_new(scalar_fn.clone(), [])
            .vortex_expect("literal expressions are always well-typed");
        BoolLiteral {
            scalar_fn,
            expr,
            bound_expr,
        }
    })
});

/// Returns the shared literal nodes for `scalar` if it is a boolean, or `None` otherwise.
fn cached_bool_literal(scalar: &Scalar) -> Option<&'static BoolLiteral> {
    let DType::Bool(nullability) = scalar.dtype() else {
        return None;
    };
    let value = match scalar.value() {
        None => None,
        Some(ScalarValue::Bool(value)) => Some(*value),
        Some(_) => return None,
    };
    let index = match (nullability, value) {
        (Nullability::NonNullable, Some(false)) => 0,
        (Nullability::NonNullable, Some(true)) => 1,
        (Nullability::Nullable, Some(false)) => 2,
        (Nullability::Nullable, Some(true)) => 3,
        (Nullability::Nullable, None) => 4,
        (Nullability::NonNullable, None) => return None,
    };
    Some(&BOOL_LITERALS[index])
}

impl ScalarFnVTable for Literal {
    type Options = Scalar;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.literal");
        *ID
    }

    fn serialize(&self, instance: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::LiteralOpts {
                value: Some(instance.into()),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let ops = pb::LiteralOpts::decode(_metadata)?;
        Scalar::from_proto(
            ops.value
                .as_ref()
                .ok_or_else(|| vortex_err!("Literal metadata missing value"))?,
            session,
        )
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(0)
    }

    fn child_name(&self, _instance: &Self::Options, _child_idx: usize) -> ChildName {
        unreachable!()
    }

    fn fmt_sql(
        &self,
        scalar: &Scalar,
        _expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", scalar)
    }

    fn return_dtype(&self, options: &Self::Options, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(options.dtype().clone())
    }

    fn execute(
        &self,
        scalar: &Scalar,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(ConstantArray::new(scalar.clone(), args.row_count()).into_array())
    }

    fn validity(
        &self,
        scalar: &Scalar,
        _expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(Literal::expr(scalar.is_valid().into())))
    }

    fn is_strict(&self, _instance: &Self::Options) -> bool {
        true
    }

    fn is_infallible(&self, _instance: &Self::Options) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::Literal;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::bound;
    use crate::expr::lit;
    use crate::expr::test_harness;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTableExt;

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();

        assert_eq!(
            lit(10).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(
            lit(i64::MAX).return_dtype(&dtype).unwrap(),
            DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_eq!(
            lit(true).return_dtype(&dtype).unwrap(),
            DType::Bool(Nullability::NonNullable)
        );
        assert_eq!(
            lit(Scalar::null(DType::Bool(Nullability::Nullable)))
                .return_dtype(&dtype)
                .unwrap(),
            DType::Bool(Nullability::Nullable)
        );

        let sdtype = DType::Struct(
            StructFields::new(
                ["dog", "cat"].into(),
                vec![
                    DType::Primitive(PType::U32, Nullability::NonNullable),
                    DType::Utf8(Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        );
        assert_eq!(
            lit(Scalar::struct_(
                sdtype.clone(),
                vec![Scalar::from(32_u32), Scalar::from("rufus".to_string())]
            ))
            .return_dtype(&dtype)
            .unwrap(),
            sdtype
        );
    }

    #[rstest]
    #[case(Scalar::bool(false, Nullability::NonNullable))]
    #[case(Scalar::bool(true, Nullability::NonNullable))]
    #[case(Scalar::bool(false, Nullability::Nullable))]
    #[case(Scalar::bool(true, Nullability::Nullable))]
    #[case(Scalar::null(DType::Bool(Nullability::Nullable)))]
    fn bool_literals_are_shared(#[case] scalar: Scalar) {
        let (first, second) = (lit(scalar.clone()), lit(scalar.clone()));
        assert_eq!(first, Literal.new_expr(scalar.clone(), []));
        // Both handles point at the same erased scalar fn, so neither allocated one.
        assert!(std::ptr::eq(
            first.as_::<Literal>(),
            second.as_::<Literal>()
        ));

        let (first, second) = (bound::lit(scalar.clone()), bound::lit(scalar.clone()));
        assert_eq!(first.dtype(), scalar.dtype());
        assert!(std::ptr::eq(
            first.as_::<Literal>(),
            second.as_::<Literal>()
        ));
    }

    #[test]
    fn distinct_bool_literals_are_not_shared() {
        assert_ne!(lit(true), lit(false));
        assert!(!std::ptr::eq(
            lit(true).as_::<Literal>(),
            lit(false).as_::<Literal>()
        ));

        let nullable = lit(Scalar::bool(true, Nullability::Nullable));
        assert_eq!(
            nullable.as_::<Literal>().dtype(),
            &DType::Bool(Nullability::Nullable)
        );
        assert!(!std::ptr::eq(
            lit(true).as_::<Literal>(),
            nullable.as_::<Literal>()
        ));
    }
}
