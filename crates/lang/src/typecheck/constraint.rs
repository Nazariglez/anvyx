use crate::{
    ast::{ExprId, Ident, Type},
    span::Span,
};

use super::{error::TypeErr, types::TypeChecker, unify::unify_equal};

#[derive(Debug, Clone)]
pub(super) enum TypeRef {
    Expr(ExprId),
    Var(Ident),
    Concrete(Type),
}

impl TypeRef {
    pub(super) fn concrete(ty: &Type) -> Self {
        TypeRef::Concrete(ty.clone())
    }
}

#[derive(Debug, Clone)]
pub(super) struct Constraint {
    pub span: Span,
    pub left: TypeRef,
    pub right: TypeRef,
}

pub(super) fn resolve_constraints(type_checker: &mut TypeChecker, errors: &mut Vec<TypeErr>) {
    // keep going until we make no progress infering types
    loop {
        if !resolve_constraints_pass(type_checker, errors) {
            break;
        }
    }
}

fn resolve_constraints_pass(type_checker: &mut TypeChecker, errors: &mut Vec<TypeErr>) -> bool {
    let mut made_progress = false;

    let constraints = std::mem::take(&mut type_checker.constraints);
    for c in constraints {
        let unified = unify_equal(type_checker, c.span, &c.left, &c.right, errors);
        if !unified {
            type_checker.constraints.push(c);
        }

        // if unified just set made_progress to true otherwise keep it false
        made_progress |= unified;
    }

    made_progress
}
