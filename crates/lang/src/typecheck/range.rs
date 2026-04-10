use internment::Intern;

use crate::ast::{Ident, Type};

fn make_range_struct(name: &str, elem_ty: Type) -> Type {
    Type::Struct {
        name: Ident(Intern::new(name.to_string())),
        type_args: vec![elem_ty],
        origin: None,
    }
}

pub(super) fn range_type(elem_ty: Type) -> Type {
    make_range_struct("Range", elem_ty)
}

pub(super) fn range_inclusive_type(elem_ty: Type) -> Type {
    make_range_struct("RangeInclusive", elem_ty)
}

pub(super) fn range_from_type(elem_ty: Type) -> Type {
    make_range_struct("RangeFrom", elem_ty)
}

pub(super) fn range_to_type(elem_ty: Type) -> Type {
    make_range_struct("RangeTo", elem_ty)
}

pub(super) fn range_to_inclusive_type(elem_ty: Type) -> Type {
    make_range_struct("RangeToInclusive", elem_ty)
}

pub(super) fn range_element_type(ty: &Type) -> Option<&Type> {
    match ty {
        Type::Struct {
            name, type_args, ..
        } if type_args.len() == 1 => {
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
