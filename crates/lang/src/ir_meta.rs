use crate::{ast::Type, hir::FuncId};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AggregateKind {
    Struct,
    DataRef,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldMeta {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AggregateMeta {
    pub name: String,
    pub qualified_name: String,
    pub kind: AggregateKind,
    pub fields: Vec<FieldMeta>,
    pub display_func: Option<FuncId>,
    pub cycle_capable: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumMeta {
    pub name: String,
    pub qualified_name: String,
    pub variants: Vec<VariantMeta>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariantMeta {
    pub name: String,
    pub shape: VariantShape,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VariantShape {
    Unit,
    Tuple(Vec<Type>),
    Struct(Vec<FieldMeta>),
}
