use crate::{
    ast::{
        ArrayLen, BinaryOp, BlockNode, CallNode, EnumDecl, ExprId, ExtendMethodNode,
        FieldAccessNode, FuncNode, Ident, IndexNode, Method, MethodReceiver, Mutability, Param,
        StmtNode, StructDecl, StructField, Type, TypeParam, TypeVarId, UnaryOp, VariantKind,
        Visibility,
    },
    span::Span,
};

use super::const_eval::{ConstDef, ConstValue};
use std::collections::{HashMap, HashSet};

use super::{
    constraint::{Constraint, TypeRef},
    error::{TypeErr, TypeErrKind},
    infer::{build_subst, subst_type},
    unify::{contains_infer, is_assignable, unify_equal},
};

#[derive(Debug, Clone)]
pub(super) struct EnumVariantDef {
    pub name: Ident,
    pub kind: VariantKind,
}

#[derive(Debug, Clone)]
pub(super) struct EnumDef {
    pub type_params: Vec<TypeParam>,
    pub variants: Vec<EnumVariantDef>,
}

impl EnumDef {
    pub(super) fn from_ast(decl: &EnumDecl) -> Self {
        let variants = decl
            .variants
            .iter()
            .map(|v| EnumVariantDef {
                name: v.name,
                kind: v.kind.clone(),
            })
            .collect();
        Self {
            type_params: decl.type_params.clone(),
            variants,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct MethodDef {
    pub type_params: Vec<TypeParam>,
    pub receiver: Option<MethodReceiver>,
    pub params: Vec<Param>,
    pub param_defaults: Vec<Option<ConstValue>>,
    pub ret: Type,
    pub body: BlockNode,
}

impl MethodDef {
    pub(super) fn from_ast(method: &Method) -> (Ident, Self) {
        (
            method.name,
            Self {
                type_params: method.type_params.clone(),
                receiver: method.receiver,
                params: method.params.clone(),
                param_defaults: vec![],
                ret: method.ret.clone(),
                body: method.body.clone(),
            },
        )
    }
}

#[derive(Debug, Clone)]
pub enum FieldDefault {
    Const(ConstValue),
    EmptyArray,
    EmptyMap,
}

pub(super) fn type_references_generic(ty: &Type, type_params: &[TypeParam]) -> bool {
    match ty {
        Type::Var(id) => type_params.iter().any(|p| p.id == *id),
        Type::List { elem } | Type::Array { elem, .. } | Type::ArrayView { elem } => {
            type_references_generic(elem, type_params)
        }
        Type::Map { key, value } => {
            type_references_generic(key, type_params) || type_references_generic(value, type_params)
        }
        Type::Tuple(elems) => elems
            .iter()
            .any(|e| type_references_generic(e, type_params)),
        Type::NamedTuple(fields) => fields
            .iter()
            .any(|(_, t)| type_references_generic(t, type_params)),
        Type::Struct { type_args, .. } | Type::Enum { type_args, .. } => type_args
            .iter()
            .any(|a| type_references_generic(a, type_params)),
        Type::Func { params, ret } => {
            params
                .iter()
                .any(|t| type_references_generic(t, type_params))
                || type_references_generic(ret, type_params)
        }
        _ => false,
    }
}

#[derive(Debug, Clone)]
pub(super) struct StructDef {
    pub type_params: Vec<TypeParam>,
    pub fields: Vec<StructField>,
    pub methods: HashMap<Ident, MethodDef>,
    pub field_defaults: HashMap<Ident, FieldDefault>,
}

impl StructDef {
    pub(super) fn from_ast(decl: &StructDecl) -> Self {
        let methods = decl.methods.iter().map(MethodDef::from_ast).collect();
        Self {
            type_params: decl.type_params.clone(),
            fields: decl.fields.clone(),
            methods,
            field_defaults: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExternFieldDef {
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct ExternMethodDef {
    pub receiver: Option<MethodReceiver>,
    pub params: Vec<Param>,
    pub ret: Type,
}

#[derive(Debug, Clone)]
pub struct ExternOpDef {
    pub op: BinaryOp,
    pub other_ty: Type,
    pub ret: Type,
    pub self_on_right: bool,
}

#[derive(Debug, Clone)]
pub struct ExternUnaryOpDef {
    pub op: UnaryOp,
    pub ret: Type,
}

#[derive(Debug, Clone)]
pub struct ExternTypeDef {
    pub has_init: bool,
    pub field_order: Vec<Ident>,
    pub fields: HashMap<Ident, ExternFieldDef>,
    pub methods: HashMap<Ident, ExternMethodDef>,
    pub statics: HashMap<Ident, ExternMethodDef>,
    pub operators: Vec<ExternOpDef>,
    pub unary_operators: Vec<ExternUnaryOpDef>,
}

#[derive(Debug, Clone)]
pub(super) struct ExtendMethodDef {
    pub params: Vec<Param>, // full param list including self
    pub ret: Type,
    pub internal_name: Ident,
}

#[derive(Debug, Clone)]
pub(super) struct ExtendEntry {
    pub source_module: Vec<String>,
    pub binding: Ident,
    pub def: ExtendMethodDef,
}

#[derive(Debug, Clone)]
pub struct ModuleExtendEntry {
    pub ty: Type,
    pub name: Ident,
    pub def: ExtendMethodDef,
}

#[derive(Debug, Clone)]
pub struct GenericExtendTemplate {
    pub type_params: Vec<TypeParam>,
    pub method: ExtendMethodNode,
    pub visibility: Visibility,
    pub source_module: Vec<String>,
    pub binding: Ident,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtendSpecKey {
    pub base_name: Ident,
    pub method_name: Ident,
    pub type_args: Vec<Type>,
}

impl ExtendSpecKey {
    pub fn mangle(&self, source_module: &[String]) -> Ident {
        use internment::Intern;
        let module_part = if source_module.is_empty() {
            String::new()
        } else {
            source_module.join("::")
        };
        let args_str = self
            .type_args
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let type_str = format!("{}<{}>", self.base_name, args_str);
        let name = format!(
            "__extend::{}::{}::{}",
            module_part, type_str, self.method_name
        );
        Ident(Intern::new(name))
    }
}

#[derive(Debug, Clone)]
pub struct ModuleGenericExtendEntry {
    pub base_name: Ident,
    pub type_params: Vec<TypeParam>,
    pub method_name: Ident,
    pub method: ExtendMethodNode,
}

pub(super) type InferenceSlots = HashMap<TypeVarId, Ident>;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct RetType {
    pub ty: Type,
    pub has_explicit: bool,
    pub span: Option<Span>,
}

#[derive(Debug, Clone)]
pub(super) struct MethodContext {
    pub struct_name: Ident,
    pub receiver: Option<MethodReceiver>,
}

/// Key for caching scpecialized generic functions (instantiated with concrete types)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpecializationKey {
    pub func_name: Ident,
    pub type_args: Vec<Type>,
}

/// Key for caching specialized generic method bodies
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct MethodSpecKey {
    pub struct_name: Ident,
    pub method_name: Ident,
    /// Struct type args concatenated with method type args
    pub type_args: Vec<Type>,
}

#[derive(Debug, Clone)]
pub struct SpecializationResult {
    pub ret_ty: Type,
    pub err: Option<(Span, TypeErrKind)>,
    pub body_types: HashMap<ExprId, (Span, Type)>,
}

#[derive(Debug, Clone)]
pub(super) struct VarInfo {
    pub ty: Type,
    pub mutable: bool,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ModuleDef {
    pub funcs: HashMap<Ident, Type>,
    pub func_param_info: HashMap<Ident, Vec<(Ident, Mutability)>>,
    pub struct_defs: HashMap<Ident, StructDef>,
    pub enum_defs: HashMap<Ident, EnumDef>,
    pub extern_types: HashMap<Ident, ExternTypeDef>,
    pub func_type_params: HashMap<Ident, Vec<TypeParam>>,
    pub generic_func_templates: HashMap<Ident, FuncNode>,

    //  all top-level declaration names (public and private) for private vs missing checks
    pub all_names: HashSet<Ident>,

    /// sub modules re-exported via `pub import X;` or `pub import X as alias;`
    pub re_exported_modules: HashMap<Ident, ModuleDef>,

    /// public const definitions exported by this module
    pub const_defs: HashMap<Ident, ConstDef>,

    /// extend methods exported by this module
    pub extend_methods: Vec<ModuleExtendEntry>,

    /// generic extend method templates exported by this module
    pub generic_extend_methods: Vec<ModuleGenericExtendEntry>,

    /// evaluated parameter defaults for exported functions, parallel to params (None = required)
    pub func_param_defaults: HashMap<Ident, Vec<Option<ConstValue>>>,
}

impl ModuleDef {
    /// iterator over all publicly visible symbols (funcs + structs + enums)
    pub fn all_public_names(&self) -> impl Iterator<Item = Ident> + '_ {
        self.funcs
            .keys()
            .chain(self.struct_defs.keys())
            .chain(self.enum_defs.keys())
            .chain(self.extern_types.keys())
            .chain(self.const_defs.keys())
            .copied()
    }
}

#[derive(Debug, Default)]
pub struct TypeChecker {
    /// Resolved type for each expression
    pub(super) types: HashMap<ExprId, (Span, Type)>,

    /// Stack of scopes for variable lookup
    pub(super) scopes: Vec<HashMap<Ident, VarInfo>>,

    /// Stack of return types for function calls
    pub(super) return_types: Vec<RetType>,

    /// Stack tracking the current method context (if any)
    pub(super) method_contexts: Vec<MethodContext>,

    /// Type constraints to be resolved by inference pass
    pub(super) constraints: Vec<Constraint>,

    /// Generic type params declared for function
    pub(super) func_type_params: HashMap<Ident, Vec<TypeParam>>,

    /// Identify inference slots uniquely across multiple generic calls
    pub(super) next_infer_call_id: usize,

    /// Stores the generic function templates for later instantiation at call sites
    /// the bodies are checked when instantiated with concrete type arguments in a later pass
    pub(super) generic_func_templates: HashMap<Ident, FuncNode>,

    /// Stores specialized functions avoiding re-checking for same type arguments
    pub(super) specialization_cache: HashMap<SpecializationKey, SpecializationResult>,

    /// Stores specialized generic method bodies avoiding re-checking for same type arguments
    pub(super) method_spec_cache: HashMap<MethodSpecKey, SpecializationResult>,

    /// Stores struct definitions (name -> fields)
    pub(super) struct_defs: HashMap<Ident, StructDef>,

    /// Stores enum definitions (name -> variants)
    pub(super) enum_defs: HashMap<Ident, EnumDef>,

    /// Stores extern type definitions declared with 'extern type'
    pub(super) extern_type_defs: HashMap<Ident, ExternTypeDef>,

    /// Stores param info for free functions
    pub(super) func_param_info: HashMap<Ident, Vec<(Ident, Mutability)>>,

    /// Stores evaluated parameter defaults for free functions, parallel to params (None = required)
    pub(super) func_param_defaults: HashMap<Ident, Vec<Option<ConstValue>>>,

    /// Tracks depth of nested loops to validate break/continue usage
    pub(super) loop_depth: usize,

    /// Stmts from resolved imported modules, keyed by import path segments
    pub(super) resolved_module_stmts: HashMap<Vec<String>, Vec<StmtNode>>,

    /// Pre-built ModuleDefs for each resolved module, keyed by import path segments
    pub(super) resolved_module_defs: HashMap<Vec<String>, ModuleDef>,

    /// Module bindings for qualified access (binding_name -> module declarations)
    pub(super) module_defs: HashMap<Ident, ModuleDef>,

    /// Active snapshot for capturing expression types during generic body specialization
    pub(super) spec_type_snapshot: Option<HashMap<ExprId, (Span, Type)>>,

    /// Resolved type args per call site, keyed by callee ExprId
    pub resolved_call_type_args: HashMap<ExprId, (Ident, Vec<Type>)>,

    /// Tracks which module a generic function was imported from
    pub(super) generic_func_source_module: HashMap<Ident, Vec<String>>,

    /// Stores evaluated const definitions keyed by name
    pub(super) const_defs: HashMap<Ident, ConstDef>,

    /// Maps expression IDs to their const values for inlining by the lowering pass
    pub const_values: HashMap<ExprId, ConstValue>,

    /// Maps pattern spans (start, end) to resolved const values for match-pattern lowering
    pub const_pattern_values: HashMap<(usize, usize), ConstValue>,

    /// Tracks const names introduced per block scope inside function bodies
    pub(super) const_scope_stack: Vec<HashSet<Ident>>,

    /// Registered extend methods keyed by (extended type, method name)
    pub(super) extend_defs: HashMap<(Type, Ident), Vec<ExtendEntry>>,

    /// Generic extend templates keyed by (base type name, method name)
    pub(super) generic_extend_templates: HashMap<(Ident, Ident), Vec<GenericExtendTemplate>>,

    /// Specialization cache for generic extend methods
    pub(super) extend_spec_cache: HashMap<ExtendSpecKey, SpecializationResult>,

    /// Maps call expression ExprId to resolved extend method internal_name (for lowering)
    pub(super) extend_call_targets: HashMap<ExprId, Ident>,
}

impl TypeChecker {
    pub(super) fn next_call_id(&mut self) -> usize {
        let id = self.next_infer_call_id;
        self.next_infer_call_id += 1;
        id
    }

    pub(super) fn push_method_context(&mut self, ctx: MethodContext) {
        self.method_contexts.push(ctx);
    }

    pub(super) fn pop_method_context(&mut self) {
        self.method_contexts.pop();
    }

    pub(super) fn current_method(&self) -> Option<&MethodContext> {
        self.method_contexts.last()
    }

    pub fn func_param_defaults(&self, name: Ident) -> &[Option<ConstValue>] {
        self.func_param_defaults
            .get(&name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn method_param_defaults(
        &self,
        struct_name: Ident,
        method_name: Ident,
    ) -> &[Option<ConstValue>] {
        self.struct_defs
            .get(&struct_name)
            .and_then(|sd| sd.methods.get(&method_name))
            .map(|m| m.param_defaults.as_slice())
            .unwrap_or(&[])
    }

    pub fn module_func_param_defaults(
        &self,
        module_name: Ident,
        func_name: Ident,
    ) -> &[Option<ConstValue>] {
        self.module_defs
            .get(&module_name)
            .and_then(|m| m.func_param_defaults.get(&func_name))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub(super) fn get_struct(&self, name: Ident) -> Option<&StructDef> {
        self.struct_defs.get(&name)
    }

    pub(super) fn get_enum(&self, name: Ident) -> Option<&EnumDef> {
        self.enum_defs.get(&name)
    }

    pub(super) fn get_const(&self, name: Ident) -> Option<&ConstDef> {
        self.const_defs.get(&name)
    }

    pub fn get_extern_type(&self, name: Ident) -> Option<&ExternTypeDef> {
        self.extern_type_defs.get(&name)
    }

    pub fn extern_type_field_order(&self, name: Ident) -> Option<&[Ident]> {
        self.extern_type_defs
            .get(&name)
            .map(|def| def.field_order.as_slice())
    }

    pub fn struct_names(&self) -> impl Iterator<Item = Ident> + '_ {
        self.struct_defs.keys().copied()
    }

    pub fn struct_field_names(&self, name: Ident) -> Option<Vec<Ident>> {
        self.struct_defs
            .get(&name)
            .map(|def| def.fields.iter().map(|f| f.name).collect())
    }

    pub fn struct_field_index(&self, struct_name: Ident, field_name: Ident) -> Option<usize> {
        self.struct_defs
            .get(&struct_name)?
            .fields
            .iter()
            .position(|f| f.name == field_name)
    }

    pub fn struct_field_type(&self, struct_name: Ident, field_name: Ident) -> Option<Type> {
        self.struct_defs
            .get(&struct_name)?
            .fields
            .iter()
            .find(|f| f.name == field_name)
            .map(|f| f.ty.clone())
    }

    pub fn struct_field_default(
        &self,
        struct_name: Ident,
        field_name: Ident,
    ) -> Option<&FieldDefault> {
        self.struct_defs
            .get(&struct_name)?
            .field_defaults
            .get(&field_name)
    }

    pub fn struct_to_string_body(&self, name: Ident) -> Option<(&BlockNode, &Type)> {
        use internment::Intern;
        let def = self.struct_defs.get(&name)?;
        if !def.type_params.is_empty() {
            return None;
        }
        let to_string = Ident(Intern::new("to_string".to_string()));
        let method = def.methods.get(&to_string)?;
        if method.receiver != Some(MethodReceiver::Value) {
            return None;
        }
        if method.ret != Type::String {
            return None;
        }
        if !method.params.is_empty() {
            return None;
        }
        if !method.type_params.is_empty() {
            return None;
        }
        Some((&method.body, &method.ret))
    }

    pub fn enum_names(&self) -> impl Iterator<Item = Ident> + '_ {
        self.enum_defs.keys().copied()
    }

    pub fn enum_variant_index(&self, enum_name: Ident, variant_name: Ident) -> Option<u16> {
        let def = self.enum_defs.get(&enum_name)?;
        let idx = def.variants.iter().position(|v| v.name == variant_name)?;
        Some(idx as u16)
    }

    pub fn enum_variant_field_names(
        &self,
        enum_name: Ident,
        variant_name: Ident,
    ) -> Option<Vec<Ident>> {
        let def = self.enum_defs.get(&enum_name)?;
        let variant = def.variants.iter().find(|v| v.name == variant_name)?;
        match &variant.kind {
            VariantKind::Struct(fields) => Some(fields.iter().map(|f| f.name).collect()),
            _ => None,
        }
    }

    pub fn enum_variant_field_types(
        &self,
        enum_name: Ident,
        variant_name: Ident,
        type_args: &[Type],
    ) -> Option<Vec<Type>> {
        let def = self.enum_defs.get(&enum_name)?;
        let variant = def.variants.iter().find(|v| v.name == variant_name)?;
        let subst = build_subst(&def.type_params, type_args);
        match &variant.kind {
            VariantKind::Tuple(types) => {
                Some(types.iter().map(|t| subst_type(t, &subst)).collect())
            }
            VariantKind::Struct(fields) => {
                Some(fields.iter().map(|f| subst_type(&f.ty, &subst)).collect())
            }
            VariantKind::Unit => Some(vec![]),
        }
    }

    pub fn enum_variant_kinds(&self, name: Ident) -> Option<Vec<(Ident, &VariantKind)>> {
        self.enum_defs
            .get(&name)
            .map(|def| def.variants.iter().map(|v| (v.name, &v.kind)).collect())
    }

    pub(super) fn get_extend_methods(&self, ty: &Type, name: Ident) -> &[ExtendEntry] {
        self.extend_defs
            .get(&(ty.clone(), name))
            .map_or(&[], Vec::as_slice)
    }

    pub(super) fn get_module(&self, name: Ident) -> Option<&ModuleDef> {
        self.module_defs.get(&name)
    }

    pub fn is_module_name(&self, name: Ident) -> bool {
        self.module_defs.contains_key(&name)
    }

    pub(super) fn resolve_type(&self, ty: &Type) -> Type {
        match ty {
            Type::UnresolvedName(name) if self.extern_type_defs.contains_key(name) => {
                Type::Extern { name: *name }
            }
            Type::UnresolvedName(name) if self.struct_defs.contains_key(name) => Type::Struct {
                name: *name,
                type_args: vec![],
            },
            Type::UnresolvedName(name) if self.enum_defs.contains_key(name) => Type::Enum {
                name: *name,
                type_args: vec![],
            },
            // the parser creates a struce for any named type with type args, this can be an enum or a struct
            Type::Struct { name, type_args } if self.enum_defs.contains_key(name) => Type::Enum {
                name: *name,
                type_args: type_args.iter().map(|t| self.resolve_type(t)).collect(),
            },
            Type::Struct { name, type_args } => Type::Struct {
                name: *name,
                type_args: type_args.iter().map(|t| self.resolve_type(t)).collect(),
            },
            Type::Enum { name, type_args } => Type::Enum {
                name: *name,
                type_args: type_args.iter().map(|t| self.resolve_type(t)).collect(),
            },
            Type::Tuple(elems) => Type::Tuple(elems.iter().map(|t| self.resolve_type(t)).collect()),
            Type::NamedTuple(fields) => Type::NamedTuple(
                fields
                    .iter()
                    .map(|(n, t)| (*n, self.resolve_type(t)))
                    .collect(),
            ),
            Type::Func { params, ret } => Type::Func {
                params: params.iter().map(|t| self.resolve_type(t)).collect(),
                ret: Box::new(self.resolve_type(ret)),
            },
            Type::Array { elem, len } => {
                let resolved_len = match len {
                    ArrayLen::Named(ident) => match self.const_defs.get(ident) {
                        Some(def) => match &def.value {
                            ConstValue::Int(n) if *n >= 0 => ArrayLen::Fixed(*n as usize),
                            _ => ArrayLen::Named(*ident),
                        },
                        None => ArrayLen::Named(*ident),
                    },
                    other => *other,
                };
                Type::Array {
                    elem: self.resolve_type(elem).boxed(),
                    len: resolved_len,
                }
            }
            Type::ArrayView { elem } => Type::ArrayView {
                elem: self.resolve_type(elem).boxed(),
            },
            Type::List { elem } => Type::List {
                elem: self.resolve_type(elem).boxed(),
            },
            Type::Map { key, value } => Type::Map {
                key: self.resolve_type(key).boxed(),
                value: self.resolve_type(value).boxed(),
            },
            _ => ty.clone(),
        }
    }

    pub(super) fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub(super) fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn set_type(&mut self, id: ExprId, ty: Type, span: Span) {
        if let Some(snapshot) = &mut self.spec_type_snapshot {
            snapshot.insert(id, (span, ty));
            return;
        }
        self.types.insert(id, (span, ty));
    }

    pub fn get_type(&self, id: ExprId) -> Option<&(Span, Type)> {
        if let Some(snapshot) = &self.spec_type_snapshot
            && let Some(entry) = snapshot.get(&id)
        {
            return Some(entry);
        }
        self.types.get(&id)
    }

    pub fn types(&self) -> impl Iterator<Item = (&ExprId, &(Span, Type))> {
        self.types.iter()
    }

    pub fn specializations(&self) -> &HashMap<SpecializationKey, SpecializationResult> {
        &self.specialization_cache
    }

    pub fn generic_template(&self, name: Ident) -> Option<&FuncNode> {
        self.generic_func_templates.get(&name)
    }

    pub fn call_type_args(&self, callee_expr_id: ExprId) -> Option<&(Ident, Vec<Type>)> {
        self.resolved_call_type_args.get(&callee_expr_id)
    }

    pub fn extend_call_target(&self, id: ExprId) -> Option<Ident> {
        self.extend_call_targets.get(&id).copied()
    }

    pub fn extend_specializations(&self) -> &HashMap<ExtendSpecKey, SpecializationResult> {
        &self.extend_spec_cache
    }

    pub fn get_generic_extend_template(
        &self,
        base_name: Ident,
        method_name: Ident,
    ) -> Option<&GenericExtendTemplate> {
        self.generic_extend_templates
            .get(&(base_name, method_name))
            .and_then(|v| v.first())
    }

    pub(super) fn set_var(&mut self, name: Ident, ty: Type, mutable: bool) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, VarInfo { ty, mutable });
        }
    }

    pub(super) fn enter_loop(&mut self) {
        self.loop_depth += 1;
    }

    pub(super) fn exit_loop(&mut self) {
        self.loop_depth = self.loop_depth.saturating_sub(1);
    }

    pub(super) fn in_loop(&self) -> bool {
        self.loop_depth > 0
    }

    pub(super) fn get_var(&self, name: Ident) -> Option<&VarInfo> {
        for scope in self.scopes.iter().rev() {
            if let Some(info) = scope.get(&name) {
                return Some(info);
            }
        }
        None
    }

    pub(super) fn push_return_type(&mut self, ty: Type, span: Option<Span>) {
        self.return_types.push(RetType {
            ty,
            has_explicit: false,
            span,
        });
    }

    pub(super) fn pop_return_type(&mut self) {
        self.return_types.pop();
    }

    pub(super) fn current_return_type(&self) -> Option<&Type> {
        self.return_types.last().map(|r| &r.ty)
    }

    pub(super) fn mark_explicit_return(&mut self) {
        if let Some(ret_ty) = self.return_types.last_mut() {
            ret_ty.has_explicit = true;
        }
    }

    pub(super) fn has_explicit_return(&self) -> bool {
        self.return_types.last().is_some_and(|r| r.has_explicit)
    }

    pub(super) fn add_constraint(&mut self, span: Span, left: TypeRef, right: TypeRef) {
        self.constraints.push(Constraint { span, left, right });
    }

    pub(super) fn get_type_ref(&self, r: &TypeRef) -> Option<Type> {
        match r {
            TypeRef::Expr(id) => self.get_type(*id).map(|(_, ty)| ty.clone()),
            TypeRef::Var(ident) => self.get_var(*ident).map(|info| info.ty.clone()),
            TypeRef::Concrete(t) => Some(t.clone()),
        }
    }

    pub(super) fn set_type_ref(&mut self, r: &TypeRef, ty: Type, span: Span) {
        match r {
            TypeRef::Expr(id) => self.set_type(*id, ty, span),
            TypeRef::Var(ident) => self.set_var(*ident, ty, true),
            TypeRef::Concrete(_) => {} // Cannot write to concrete types
        }
    }

    /// Constrains two types that must be the same
    /// ie: let x:int = 10; x += 10; (int to int)
    pub(super) fn constrain_equal(
        &mut self,
        span: Span,
        left: TypeRef,
        right: TypeRef,
        errors: &mut Vec<TypeErr>,
    ) {
        // try to unify the types immediately
        let unified = unify_equal(self, span, &left, &right, errors);

        // otherwise add a constraint to be resolved later
        if !unified {
            self.add_constraint(span, left, right);
        }
    }

    /// Constraints 'value' to be assignable to 'target'
    /// ie: let x:int? = 10; fn() -> int? { return 10; } (T to T?)
    pub(super) fn constrain_assignable(
        &mut self,
        span: Span,
        from: TypeRef,
        to: TypeRef,
        errors: &mut Vec<TypeErr>,
    ) {
        let from_ty = self.get_type_ref(&from).unwrap_or(Type::Infer);
        let to_ty = self.get_type_ref(&to).unwrap_or(Type::Infer);

        // if both types are resolved, check assignability and set the types
        let both_resolved = !(contains_infer(&from_ty) || contains_infer(&to_ty));
        if both_resolved {
            if !is_assignable(&from_ty, &to_ty) {
                errors.push(TypeErr::new(
                    span,
                    TypeErrKind::MismatchedTypes {
                        expected: to_ty.clone(),
                        found: from_ty.clone(),
                    },
                ));
            }
            self.set_type_ref(&to, to_ty, span);
            self.set_type_ref(&from, from_ty, span);
            return;
        }

        // at this point at least one type is unresolved
        // if both are options, constrain the inner types
        if from_ty.is_option() && to_ty.is_option() {
            let inner_from = from_ty.option_inner().cloned().unwrap_or(Type::Infer);
            let inner_to = to_ty.option_inner().cloned().unwrap_or(Type::Infer);
            let inner_from_ref = TypeRef::concrete(&inner_from);
            let inner_to_ref = TypeRef::concrete(&inner_to);
            self.constrain_equal(span, inner_from_ref, inner_to_ref, errors);
            self.set_type_ref(&from, Type::option_of(inner_to), span);
            return;
        }

        // if to is an option and from has inference, constrain from to the inner type of to
        if to_ty.is_option() && contains_infer(&from_ty) {
            let inner_to = to_ty.option_inner().cloned().unwrap_or(Type::Infer);
            self.constrain_equal(span, from, TypeRef::concrete(&inner_to), errors);
            return;
        }

        // optional values cannot be assigned to non-optional targets once the target is resolved
        let from_is_optional = from_ty.is_optional();
        let to_is_optional = to_ty.is_optional();
        if from_is_optional && !to_is_optional && !contains_infer(&to_ty) {
            errors.push(TypeErr::new(
                span,
                TypeErrKind::MismatchedTypes {
                    expected: to_ty.clone(),
                    found: from_ty.clone(),
                },
            ));
            return;
        }

        // single-element containers constrain inner element types
        let inner_pair = match (&from_ty, &to_ty) {
            (Type::Array { elem: f, .. }, Type::Array { elem: t, .. }) => Some((f, t)),
            (Type::ArrayView { elem: f }, Type::ArrayView { elem: t }) => Some((f, t)),
            (Type::Array { elem: f, .. }, Type::ArrayView { elem: t }) => Some((f, t)),
            (Type::List { elem: f }, Type::ArrayView { elem: t }) => Some((f, t)),
            // skip when either side has Infer, constrain_equal handles unification
            (Type::List { elem: f }, Type::List { elem: t })
                if !contains_infer(f) && !contains_infer(t) =>
            {
                Some((f, t))
            }
            _ => None,
        };

        if let Some((from_elem, to_elem)) = inner_pair {
            let from_ref = TypeRef::concrete(from_elem);
            let to_ref = TypeRef::concrete(to_elem);
            self.constrain_assignable(span, from_ref, to_ref, errors);
            return;
        }

        // if both are maps, constrain key and value types
        // skip when any inner type has Infer, constrain_equal handles unification
        if let (Type::Map { key: kf, value: vf }, Type::Map { key: kt, value: vt }) =
            (&from_ty, &to_ty)
        {
            let has_infer = contains_infer(kf)
                || contains_infer(kt)
                || contains_infer(vf)
                || contains_infer(vt);
            if !has_infer {
                let kf_ref = TypeRef::concrete(kf);
                let kt_ref = TypeRef::concrete(kt);
                self.constrain_assignable(span, kf_ref, kt_ref, errors);
                let vf_ref = TypeRef::concrete(vf);
                let vt_ref = TypeRef::concrete(vt);
                self.constrain_assignable(span, vf_ref, vt_ref, errors);
                return;
            }
        }

        let type_arg_pairs = match (&from_ty, &to_ty) {
            (
                Type::Struct {
                    name: nf,
                    type_args: af,
                },
                Type::Struct {
                    name: nt,
                    type_args: at,
                },
            )
            | (
                Type::Enum {
                    name: nf,
                    type_args: af,
                },
                Type::Enum {
                    name: nt,
                    type_args: at,
                },
            ) if nf == nt
                && af.len() == at.len()
                && !af.iter().any(contains_infer)
                && !at.iter().any(contains_infer) =>
            {
                Some(af.iter().zip(at.iter()))
            }
            _ => None,
        };

        if let Some(pairs) = type_arg_pairs {
            for (arg_from, arg_to) in pairs {
                let from_ref = TypeRef::concrete(arg_from);
                let to_ref = TypeRef::concrete(arg_to);
                self.constrain_assignable(span, from_ref, to_ref, errors);
            }
            return;
        }

        // otherwise just constrain them to be the same as fallback
        self.constrain_equal(span, from, to, errors);
    }
}

#[derive(Copy, Clone)]
pub(super) enum PostfixNodeRef<'a> {
    Field {
        expr_id: ExprId,
        node: &'a FieldAccessNode,
    },
    Index {
        expr_id: ExprId,
        node: &'a IndexNode,
    },
    Call {
        expr_id: ExprId,
        node: &'a CallNode,
    },
}

impl<'a> PostfixNodeRef<'a> {
    pub fn safe(&self) -> bool {
        match self {
            PostfixNodeRef::Field { node, .. } => node.node.safe,
            PostfixNodeRef::Index { node, .. } => node.node.safe,
            PostfixNodeRef::Call { node, .. } => node.node.safe,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            PostfixNodeRef::Field { node, .. } => node.span,
            PostfixNodeRef::Index { node, .. } => node.span,
            PostfixNodeRef::Call { node, .. } => node.span,
        }
    }

    pub fn expr_id(&self) -> ExprId {
        match self {
            PostfixNodeRef::Field { expr_id, .. } => *expr_id,
            PostfixNodeRef::Index { expr_id, .. } => *expr_id,
            PostfixNodeRef::Call { expr_id, .. } => *expr_id,
        }
    }
}

pub(super) fn type_field_on_base(
    base_ty: &Type,
    field: Ident,
    span: Span,
    type_checker: &TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    match base_ty {
        Type::NamedTuple(fields) => {
            for (label, ty) in fields {
                if *label == field {
                    return ty.clone();
                }
            }
            errors.push(TypeErr::new(
                span,
                TypeErrKind::NoSuchFieldOnTuple {
                    field,
                    tuple_type: base_ty.clone(),
                },
            ));
            Type::Infer
        }
        Type::Struct {
            name: struct_name,
            type_args,
        } => {
            let Some(struct_def) = type_checker.get_struct(*struct_name) else {
                errors.push(TypeErr::new(
                    span,
                    TypeErrKind::UnknownStruct { name: *struct_name },
                ));
                return Type::Infer;
            };

            let subst = build_subst(&struct_def.type_params, type_args);

            for struct_field in &struct_def.fields {
                if struct_field.name == field {
                    let field_ty = subst_type(&struct_field.ty, &subst);
                    return type_checker.resolve_type(&field_ty);
                }
            }

            errors.push(TypeErr::new(
                span,
                TypeErrKind::StructUnknownField {
                    struct_name: *struct_name,
                    field,
                },
            ));
            Type::Infer
        }
        Type::Extern { name } => {
            let Some(extern_def) = type_checker.get_extern_type(*name) else {
                errors.push(TypeErr::new(
                    span,
                    TypeErrKind::FieldAccessOnNonNamedTuple {
                        field,
                        found: base_ty.clone(),
                    },
                ));
                return Type::Infer;
            };

            if let Some(field_def) = extern_def.fields.get(&field) {
                return field_def.ty.clone();
            }

            errors.push(TypeErr::new(
                span,
                TypeErrKind::ExternUnknownField {
                    type_name: *name,
                    field,
                },
            ));
            Type::Infer
        }
        Type::Infer => Type::Infer,
        _ => {
            errors.push(TypeErr::new(
                span,
                TypeErrKind::FieldAccessOnNonNamedTuple {
                    field,
                    found: base_ty.clone(),
                },
            ));
            Type::Infer
        }
    }
}

pub(super) fn type_index_on_base(
    base_ty: &Type,
    index_ty: &Type,
    index_expr_id: ExprId,
    span: Span,
    index_span: Span,
    type_checker: &mut TypeChecker,
    errors: &mut Vec<TypeErr>,
) -> Type {
    if let Type::Map { key, value } = base_ty {
        let key_ref = TypeRef::Expr(index_expr_id);
        let expected_ref = TypeRef::concrete(key);
        type_checker.constrain_equal(index_span, key_ref, expected_ref, errors);
        return (**value).clone();
    }

    let maybe_int = matches!(index_ty, Type::Int | Type::Infer);
    if !maybe_int {
        errors.push(TypeErr::new(
            index_span,
            TypeErrKind::IndexNotInt {
                found: index_ty.clone(),
            },
        ));
        return Type::Infer;
    }

    match indexable_element_type(base_ty) {
        Some(elem_ty) => elem_ty,
        None => {
            errors.push(TypeErr::new(
                span,
                TypeErrKind::IndexOnNonArray {
                    found: base_ty.clone(),
                },
            ));
            Type::Infer
        }
    }
}

pub(super) fn indexable_element_type(ty: &Type) -> Option<Type> {
    match ty {
        Type::Array { elem, .. } => Some((**elem).clone()),
        Type::List { elem } => Some((**elem).clone()),
        Type::ArrayView { elem } => Some((**elem).clone()),
        _ => None,
    }
}

pub(super) fn unwrap_opt_typ(ty: &Type) -> &Type {
    ty.option_inner().unwrap_or(ty)
}

fn check_enum_property(
    name: Ident,
    type_args: &[Type],
    tc: &TypeChecker,
    check: impl Fn(&Type, &TypeChecker) -> Result<(), String>,
) -> Result<(), String> {
    let Some(enum_def) = tc.enum_defs.get(&name) else {
        return Err(format!("enum '{name}' is not known"));
    };
    let subst = build_subst(&enum_def.type_params, type_args);
    for v in &enum_def.variants {
        let resolved_types: Vec<Type> = match &v.kind {
            VariantKind::Unit => continue,
            VariantKind::Tuple(types) => types.iter().map(|t| subst_type(t, &subst)).collect(),
            VariantKind::Struct(fields) => {
                fields.iter().map(|f| subst_type(&f.ty, &subst)).collect()
            }
        };
        for resolved in &resolved_types {
            check(resolved, tc).map_err(|msg| {
                format!("variant '{}' has payload type '{resolved}': {msg}", v.name)
            })?;
        }
    }
    Ok(())
}

fn check_struct_property(
    name: Ident,
    type_args: &[Type],
    tc: &TypeChecker,
    check: impl Fn(&Type, &TypeChecker) -> Result<(), String>,
) -> Result<(), String> {
    let Some(struct_def) = tc.struct_defs.get(&name) else {
        return Err(format!("struct '{name}' is not known"));
    };
    let subst = build_subst(&struct_def.type_params, type_args);
    for f in &struct_def.fields {
        let resolved = subst_type(&f.ty, &subst);
        check(&resolved, tc)
            .map_err(|msg| format!("field '{}' has type '{resolved}': {msg}", f.name))?;
    }
    Ok(())
}

fn check_keyable(ty: &Type, tc: &TypeChecker) -> Result<(), String> {
    match ty {
        Type::Int | Type::Bool | Type::String => Ok(()),
        Type::Float | Type::Double => Err(
            "float is not keyable due to NaN and precision issues; use int or string instead"
                .to_string(),
        ),
        Type::Tuple(elems) => {
            for (i, t) in elems.iter().enumerate() {
                check_keyable(t, tc).map_err(|_| {
                    format!("tuple element {i} has type '{t}' which is not keyable")
                })?;
            }
            Ok(())
        }
        Type::NamedTuple(fields) => {
            for (label, t) in fields {
                check_keyable(t, tc)
                    .map_err(|_| format!("field '{label}' has type '{t}' which is not keyable"))?;
            }
            Ok(())
        }
        Type::Enum { name, type_args } => {
            if ty.is_option() {
                return Err("optional types cannot be used as map keys".to_string());
            }
            check_enum_property(*name, type_args, tc, check_keyable)
        }
        Type::Struct { name, type_args } => {
            check_struct_property(*name, type_args, tc, check_keyable)
        }
        other => Err(format!("type '{other}' is not keyable")),
    }
}

pub(super) fn is_keyable(ty: &Type, tc: &TypeChecker) -> bool {
    check_keyable(ty, tc).is_ok()
}

fn check_equatable(ty: &Type, tc: &TypeChecker) -> Result<(), String> {
    match ty {
        Type::Infer | Type::Int | Type::Float | Type::Double | Type::Bool | Type::String => Ok(()),
        Type::Tuple(elems) => {
            for (i, t) in elems.iter().enumerate() {
                check_equatable(t, tc).map_err(|_| {
                    format!("tuple element {i} has type '{t}' which is not equatable")
                })?;
            }
            Ok(())
        }
        Type::NamedTuple(fields) => {
            for (label, t) in fields {
                check_equatable(t, tc).map_err(|_| {
                    format!("field '{label}' has type '{t}' which is not equatable")
                })?;
            }
            Ok(())
        }
        Type::Enum { name, type_args } => {
            if ty.is_option() {
                let inner = ty.option_inner().cloned().unwrap_or(Type::Infer);
                return check_equatable(&inner, tc);
            }
            check_enum_property(*name, type_args, tc, check_equatable)
        }
        Type::Struct { name, type_args } => {
            check_struct_property(*name, type_args, tc, check_equatable)
        }
        Type::List { elem } | Type::Array { elem, .. } => {
            check_equatable(elem, tc).map_err(|_| format!("element type '{elem}' is not equatable"))
        }
        Type::Map { key, value } => {
            check_equatable(key, tc).map_err(|_| format!("key type '{key}' is not equatable"))?;
            check_equatable(value, tc).map_err(|_| format!("value type '{value}' is not equatable"))
        }
        other => Err(format!("type '{other}' is not equatable")),
    }
}

pub(super) fn is_equatable(ty: &Type, tc: &TypeChecker) -> bool {
    check_equatable(ty, tc).is_ok()
}

pub(super) fn equatable_reason(ty: &Type, tc: &TypeChecker) -> Option<String> {
    check_equatable(ty, tc).err()
}

pub(super) fn keyable_reason(ty: &Type, tc: &TypeChecker) -> Option<String> {
    check_keyable(ty, tc).err()
}

pub(super) fn validate_map_key_type(
    span: Span,
    key_ty: &Type,
    type_checker: &TypeChecker,
    errors: &mut Vec<TypeErr>,
) {
    if matches!(key_ty, Type::Infer) {
        return;
    }
    if is_keyable(key_ty, type_checker) {
        return;
    }
    if key_ty.is_option() {
        errors.push(TypeErr::new(
            span,
            TypeErrKind::MapOptionalKeyNotAllowed {
                found: key_ty.clone(),
            },
        ));
    } else if matches!(key_ty, Type::Float | Type::Double) {
        errors.push(TypeErr::new(span, TypeErrKind::MapKeyFloat));
    } else {
        let mut err = TypeErr::new(
            span,
            TypeErrKind::MapKeyNotKeyable {
                found: key_ty.clone(),
            },
        );
        if let Some(reason) = keyable_reason(key_ty, type_checker) {
            err.notes.push(reason);
        }
        errors.push(err);
    }
}

pub(super) fn build_param_info(params: &[Param]) -> Vec<(Ident, Mutability)> {
    params.iter().map(|p| (p.name, p.mutability)).collect()
}
