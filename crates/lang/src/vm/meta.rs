use anvyx_runtime::CycleVtable;

#[derive(Debug, Clone)]
pub struct StructMeta {
    pub name: String,
    pub field_names: Vec<String>,
    pub to_string_fn: Option<usize>,
    pub is_dataref: bool,
    pub cycle_capable: bool,
    pub vtable: Option<&'static CycleVtable>,
}

impl PartialEq for StructMeta {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.field_names == other.field_names
            && self.to_string_fn == other.to_string_fn
            && self.is_dataref == other.is_dataref
            && self.cycle_capable == other.cycle_capable
    }
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
