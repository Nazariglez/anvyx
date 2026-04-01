use internment::Intern;

use crate::ast::{Ident, Type};

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

pub(super) fn range_from_type(elem_ty: Type) -> Type {
    Type::Struct {
        name: Ident(Intern::new("RangeFrom".to_string())),
        type_args: vec![elem_ty],
    }
}

pub(super) fn range_to_type(elem_ty: Type) -> Type {
    Type::Struct {
        name: Ident(Intern::new("RangeTo".to_string())),
        type_args: vec![elem_ty],
    }
}

pub(super) fn range_to_inclusive_type(elem_ty: Type) -> Type {
    Type::Struct {
        name: Ident(Intern::new("RangeToInclusive".to_string())),
        type_args: vec![elem_ty],
    }
}

pub(super) fn range_element_type(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Struct { name, type_args } if type_args.len() == 1 => {
            let s = name.0.as_ref();
            let is_range = s == "Range"
                || s == "RangeInclusive"
                || s == "RangeFrom"
                || s == "RangeTo"
                || s == "RangeToInclusive";
            if is_range { Some(&type_args[0]) } else { None }
        }
        _ => None,
    }
}

pub(super) fn range_ident() -> Ident {
    // TODO: define a const for Range and RangeInclusive to be reused across the code
    Ident(Intern::new("Range".to_string()))
}

pub(super) fn range_inclusive_ident() -> Ident {
    Ident(Intern::new("RangeInclusive".to_string()))
}
