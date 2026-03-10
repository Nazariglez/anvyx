use crate::ast::{Ident, Type};
use internment::Intern;

pub(super) fn range_type(elem_ty: Type) -> Type {
    Type::Struct {
        name: range_ident(),
        type_args: vec![elem_ty],
    }
}

pub(super) fn range_inclusive_type(elem_ty: Type) -> Type {
    Type::Struct {
        name: range_inclusive_ident(),
        type_args: vec![elem_ty],
    }
}

pub(super) fn range_ident() -> Ident {
    // TODO: define a const for Range and RangeInclusive to be reused across the code
    Ident(Intern::new("Range".to_string()))
}

pub(super) fn range_inclusive_ident() -> Ident {
    Ident(Intern::new("RangeInclusive".to_string()))
}
