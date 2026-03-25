#[derive(Debug, Clone, PartialEq)]
pub struct StructMeta {
    pub name: String,
    pub field_names: Vec<String>,
    pub to_string_fn: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumMeta {
    pub name: String,
    pub variants: Vec<VariantMeta>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariantMeta {
    pub name: String,
    pub kind: VariantMetaKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VariantMetaKind {
    Unit,
    Tuple(usize),
    Struct(Vec<String>),
}
