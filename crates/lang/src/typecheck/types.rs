use crate::{
    ast::{
        BlockNode, CallNode, ExprId, FieldAccessNode, FuncNode, Ident, IndexNode, MethodReceiver,
        Mutability, Param, StmtNode, StructField, Type, TypeParam, TypeVarId, VariantKind,
    },
    span::Span,
};
use std::collections::{HashMap, HashSet};

use super::{
    constraint::{Constraint, TypeRef},
    error::{TypeErr, TypeErrKind},
    infer::subst_type,
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

#[derive(Debug, Clone)]
pub(super) struct MethodDef {
    pub type_params: Vec<TypeParam>,
    pub receiver: Option<MethodReceiver>,
    pub params: Vec<Param>,
    pub ret: Type,
    pub body: BlockNode,
}

#[derive(Debug, Clone)]
pub(super) struct StructDef {
    pub type_params: Vec<TypeParam>,
    pub fields: Vec<StructField>,
    pub methods: HashMap<Ident, MethodDef>,
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
    pub func_type_params: HashMap<Ident, Vec<TypeParam>>,
    pub generic_func_templates: HashMap<Ident, FuncNode>,

    //  all top-level declaration names (public and private) for private vs missing checks
    pub all_names: HashSet<Ident>,

    /// sub modules re-exported via `pub import X;` or `pub import X as alias;`
    pub re_exported_modules: HashMap<Ident, ModuleDef>,
}

impl ModuleDef {
    /// iterator over all publicly visible symbols (funcs + structs + enums)
    pub fn all_public_names(&self) -> impl Iterator<Item = Ident> + '_ {
        self.funcs
            .keys()
            .chain(self.struct_defs.keys())
            .chain(self.enum_defs.keys())
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

    /// Stores extern type names declared with 'extern type'
    pub(super) extern_type_defs: HashSet<Ident>,

    /// Stores param info for free functions
    pub(super) func_param_info: HashMap<Ident, Vec<(Ident, Mutability)>>,

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

    pub(super) fn get_struct(&self, name: Ident) -> Option<&StructDef> {
        self.struct_defs.get(&name)
    }

    pub(super) fn get_enum(&self, name: Ident) -> Option<&EnumDef> {
        self.enum_defs.get(&name)
    }

    pub(super) fn get_module(&self, name: Ident) -> Option<&ModuleDef> {
        self.module_defs.get(&name)
    }

    pub fn is_module_name(&self, name: Ident) -> bool {
        self.module_defs.contains_key(&name)
    }

    pub(super) fn resolve_type(&self, ty: &Type) -> Type {
        match ty {
            Type::UnresolvedName(name) if self.extern_type_defs.contains(name) => {
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
            Type::Array { elem, len } => Type::Array {
                elem: self.resolve_type(elem).boxed(),
                len: *len,
            },
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
            snapshot.insert(id, (span, ty.clone()));
        }
        self.types.insert(id, (span, ty));
    }

    pub fn get_type(&self, id: ExprId) -> Option<&(Span, Type)> {
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
            let inner_from_ref = TypeRef::Concrete(inner_from);
            let inner_to_ref = TypeRef::Concrete(inner_to.clone());
            self.constrain_equal(span, inner_from_ref, inner_to_ref, errors);
            self.set_type_ref(&from, Type::option_of(inner_to), span);
            return;
        }

        // if to is an option and from has inference, constrain from to the inner type of to
        if to_ty.is_option() && contains_infer(&from_ty) {
            let inner_to = to_ty.option_inner().cloned().unwrap_or(Type::Infer);
            self.constrain_equal(span, from, TypeRef::Concrete(inner_to), errors);
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

        // if both are arrays, constrain element types
        if let (
            Type::Array {
                elem: elem_from, ..
            },
            Type::Array { elem: elem_to, .. },
        ) = (&from_ty, &to_ty)
        {
            let elem_from_ref = TypeRef::Concrete(*elem_from.clone());
            let elem_to_ref = TypeRef::Concrete(*elem_to.clone());
            self.constrain_assignable(span, elem_from_ref, elem_to_ref, errors);
            return;
        }

        // if both are views, constrain element types
        if let (Type::ArrayView { elem: elem_from }, Type::ArrayView { elem: elem_to }) =
            (&from_ty, &to_ty)
        {
            let elem_from_ref = TypeRef::Concrete(*elem_from.clone());
            let elem_to_ref = TypeRef::Concrete(*elem_to.clone());
            self.constrain_assignable(span, elem_from_ref, elem_to_ref, errors);
            return;
        }

        // if assigning array to view, constrain element types
        if let (
            Type::Array {
                elem: elem_from, ..
            },
            Type::ArrayView { elem: elem_to },
        ) = (&from_ty, &to_ty)
        {
            let elem_from_ref = TypeRef::Concrete(*elem_from.clone());
            let elem_to_ref = TypeRef::Concrete(*elem_to.clone());
            self.constrain_assignable(span, elem_from_ref, elem_to_ref, errors);
            return;
        }

        // if assigning list to view, constrain element types
        if let (Type::List { elem: elem_from }, Type::ArrayView { elem: elem_to }) =
            (&from_ty, &to_ty)
        {
            let elem_from_ref = TypeRef::Concrete(*elem_from.clone());
            let elem_to_ref = TypeRef::Concrete(*elem_to.clone());
            self.constrain_assignable(span, elem_from_ref, elem_to_ref, errors);
            return;
        }

        // if both are lists, constrain element types
        // skip when either side has Infer, constrain_equal handles unification
        if let (Type::List { elem: elem_from }, Type::List { elem: elem_to }) = (&from_ty, &to_ty) {
            let has_infer = contains_infer(elem_from) || contains_infer(elem_to);
            if !has_infer {
                let elem_from_ref = TypeRef::Concrete(*elem_from.clone());
                let elem_to_ref = TypeRef::Concrete(*elem_to.clone());
                self.constrain_assignable(span, elem_from_ref, elem_to_ref, errors);
                return;
            }
        }

        // if both are maps, constrain key and value types
        // skip when any inner type has Infer, constrain_equal handles unification
        if let (
            Type::Map {
                key: key_from,
                value: val_from,
            },
            Type::Map {
                key: key_to,
                value: val_to,
            },
        ) = (&from_ty, &to_ty)
        {
            let has_infer = contains_infer(key_from)
                || contains_infer(key_to)
                || contains_infer(val_from)
                || contains_infer(val_to);
            if !has_infer {
                let key_from_ref = TypeRef::Concrete(*key_from.clone());
                let key_to_ref = TypeRef::Concrete(*key_to.clone());
                self.constrain_assignable(span, key_from_ref, key_to_ref, errors);
                let val_from_ref = TypeRef::Concrete(*val_from.clone());
                let val_to_ref = TypeRef::Concrete(*val_to.clone());
                self.constrain_assignable(span, val_from_ref, val_to_ref, errors);
                return;
            }
        }

        // if both are structs with the same name, constrain type args pairwise
        // skip when any type arg has Infen, constrain_equal handles unification
        if let (
            Type::Struct {
                name: name_from,
                type_args: args_from,
            },
            Type::Struct {
                name: name_to,
                type_args: args_to,
            },
        ) = (&from_ty, &to_ty)
        {
            let has_infer =
                args_from.iter().any(contains_infer) || args_to.iter().any(contains_infer);
            if name_from == name_to && args_from.len() == args_to.len() && !has_infer {
                for (arg_from, arg_to) in args_from.iter().zip(args_to.iter()) {
                    let arg_from_ref = TypeRef::Concrete(arg_from.clone());
                    let arg_to_ref = TypeRef::Concrete(arg_to.clone());
                    self.constrain_assignable(span, arg_from_ref, arg_to_ref, errors);
                }
                return;
            }
        }

        // if both are enums with the same name, constrain type args pairwise
        // skip when any type arg has Infer, constrain_equal handles unification
        if let (
            Type::Enum {
                name: name_from,
                type_args: args_from,
            },
            Type::Enum {
                name: name_to,
                type_args: args_to,
            },
        ) = (&from_ty, &to_ty)
        {
            let has_infer =
                args_from.iter().any(contains_infer) || args_to.iter().any(contains_infer);
            if name_from == name_to && args_from.len() == args_to.len() && !has_infer {
                for (arg_from, arg_to) in args_from.iter().zip(args_to.iter()) {
                    let arg_from_ref = TypeRef::Concrete(arg_from.clone());
                    let arg_to_ref = TypeRef::Concrete(arg_to.clone());
                    self.constrain_assignable(span, arg_from_ref, arg_to_ref, errors);
                }
                return;
            }
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
            let Some(struct_def) = type_checker.get_struct(*struct_name).cloned() else {
                errors.push(TypeErr::new(
                    span,
                    TypeErrKind::UnknownStruct { name: *struct_name },
                ));
                return Type::Infer;
            };

            let subst = struct_def
                .type_params
                .iter()
                .zip(type_args.iter())
                .map(|(param, arg)| (param.id, arg.clone()))
                .collect::<HashMap<_, _>>();

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
        let expected_ref = TypeRef::Concrete((**key).clone());
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

pub(super) fn is_keyable(ty: &Type, tc: &TypeChecker) -> bool {
    match ty {
        Type::Int | Type::Bool | Type::String => true,
        Type::Tuple(elems) => elems.iter().all(|t| is_keyable(t, tc)),
        Type::NamedTuple(fields) => fields.iter().all(|(_, t)| is_keyable(t, tc)),
        Type::Enum { name, type_args } => {
            if ty.is_option() {
                return false;
            }
            let Some(enum_def) = tc.enum_defs.get(name).cloned() else {
                return false;
            };
            let subst: HashMap<_, _> = enum_def
                .type_params
                .iter()
                .zip(type_args.iter())
                .map(|(param, arg)| (param.id, arg.clone()))
                .collect();
            enum_def.variants.iter().all(|v| match &v.kind {
                VariantKind::Unit => true,
                VariantKind::Tuple(types) => {
                    types.iter().all(|t| is_keyable(&subst_type(t, &subst), tc))
                }
                VariantKind::Struct(fields) => fields
                    .iter()
                    .all(|f| is_keyable(&subst_type(&f.ty, &subst), tc)),
            })
        }
        Type::Struct { name, type_args } => {
            let Some(struct_def) = tc.struct_defs.get(name).cloned() else {
                return false;
            };
            let subst: HashMap<_, _> = struct_def
                .type_params
                .iter()
                .zip(type_args.iter())
                .map(|(param, arg)| (param.id, arg.clone()))
                .collect();
            struct_def
                .fields
                .iter()
                .all(|f| is_keyable(&subst_type(&f.ty, &subst), tc))
        }
        _ => false,
    }
}

pub(super) fn is_equatable(ty: &Type, tc: &TypeChecker) -> bool {
    match ty {
        Type::Infer => true,
        Type::Int | Type::Float | Type::Bool | Type::String => true,
        Type::Tuple(elems) => elems.iter().all(|t| is_equatable(t, tc)),
        Type::NamedTuple(fields) => fields.iter().all(|(_, t)| is_equatable(t, tc)),
        Type::Enum { name, type_args } => {
            if ty.is_option() {
                let inner = ty.option_inner().map(|t| t.clone()).unwrap_or(Type::Infer);
                return is_equatable(&inner, tc);
            }
            let Some(enum_def) = tc.enum_defs.get(name).cloned() else {
                return false;
            };
            let subst: HashMap<_, _> = enum_def
                .type_params
                .iter()
                .zip(type_args.iter())
                .map(|(param, arg)| (param.id, arg.clone()))
                .collect();
            enum_def.variants.iter().all(|v| match &v.kind {
                VariantKind::Unit => true,
                VariantKind::Tuple(types) => types
                    .iter()
                    .all(|t| is_equatable(&subst_type(t, &subst), tc)),
                VariantKind::Struct(fields) => fields
                    .iter()
                    .all(|f| is_equatable(&subst_type(&f.ty, &subst), tc)),
            })
        }
        Type::Struct { name, type_args } => {
            let Some(struct_def) = tc.struct_defs.get(name).cloned() else {
                return false;
            };
            let subst: HashMap<_, _> = struct_def
                .type_params
                .iter()
                .zip(type_args.iter())
                .map(|(param, arg)| (param.id, arg.clone()))
                .collect();
            struct_def
                .fields
                .iter()
                .all(|f| is_equatable(&subst_type(&f.ty, &subst), tc))
        }
        Type::List { elem } => is_equatable(elem, tc),
        Type::Array { elem, .. } => is_equatable(elem, tc),
        Type::Map { key, value } => is_equatable(key, tc) && is_equatable(value, tc),
        _ => false,
    }
}

pub(super) fn equatable_reason(ty: &Type, tc: &TypeChecker) -> Option<String> {
    match ty {
        Type::Infer | Type::Int | Type::Float | Type::Bool | Type::String => None,
        Type::Tuple(elems) => elems.iter().enumerate().find_map(|(i, t)| {
            if !is_equatable(t, tc) {
                Some(format!(
                    "tuple element {i} has type '{t}' which is not equatable"
                ))
            } else {
                None
            }
        }),
        Type::NamedTuple(fields) => fields.iter().find_map(|(label, t)| {
            if !is_equatable(t, tc) {
                Some(format!(
                    "field '{label}' has type '{t}' which is not equatable"
                ))
            } else {
                None
            }
        }),
        Type::Enum { name, type_args } => {
            if ty.is_option() {
                let inner = ty.option_inner().map(|t| t.clone()).unwrap_or(Type::Infer);
                return equatable_reason(&inner, tc);
            }
            let Some(enum_def) = tc.enum_defs.get(name).cloned() else {
                return Some(format!("enum '{name}' is not known"));
            };
            let subst: HashMap<_, _> = enum_def
                .type_params
                .iter()
                .zip(type_args.iter())
                .map(|(param, arg)| (param.id, arg.clone()))
                .collect();
            enum_def.variants.iter().find_map(|v| {
                let offending = match &v.kind {
                    VariantKind::Unit => None,
                    VariantKind::Tuple(types) => types.iter().find_map(|t| {
                        let resolved = subst_type(t, &subst);
                        if !is_equatable(&resolved, tc) {
                            Some(resolved)
                        } else {
                            None
                        }
                    }),
                    VariantKind::Struct(fields) => fields.iter().find_map(|f| {
                        let resolved = subst_type(&f.ty, &subst);
                        if !is_equatable(&resolved, tc) {
                            Some(resolved)
                        } else {
                            None
                        }
                    }),
                };
                offending.map(|bad_ty| {
                    format!(
                        "variant '{}' has payload type '{bad_ty}' which is not equatable",
                        v.name
                    )
                })
            })
        }
        Type::Struct { name, type_args } => {
            let Some(struct_def) = tc.struct_defs.get(name).cloned() else {
                return Some(format!("struct '{name}' is not known"));
            };
            let subst: HashMap<_, _> = struct_def
                .type_params
                .iter()
                .zip(type_args.iter())
                .map(|(param, arg)| (param.id, arg.clone()))
                .collect();
            struct_def.fields.iter().find_map(|f| {
                let resolved = subst_type(&f.ty, &subst);
                if !is_equatable(&resolved, tc) {
                    Some(format!(
                        "field '{}' has type '{resolved}' which is not equatable",
                        f.name
                    ))
                } else {
                    None
                }
            })
        }
        Type::List { elem } | Type::Array { elem, .. } => {
            if !is_equatable(elem, tc) {
                Some(format!("element type '{elem}' is not equatable"))
            } else {
                None
            }
        }
        Type::Map { key, value } => {
            if !is_equatable(key, tc) {
                Some(format!("key type '{key}' is not equatable"))
            } else if !is_equatable(value, tc) {
                Some(format!("value type '{value}' is not equatable"))
            } else {
                None
            }
        }
        other => Some(format!("type '{other}' is not equatable")),
    }
}

pub(super) fn keyable_reason(ty: &Type, tc: &TypeChecker) -> Option<String> {
    match ty {
        Type::Int | Type::Bool | Type::String => None,
        Type::Float => Some(
            "float is not keyable due to NaN and precision issues; use int or string instead"
                .to_string(),
        ),
        Type::Tuple(elems) => elems.iter().enumerate().find_map(|(i, t)| {
            if !is_keyable(t, tc) {
                Some(format!(
                    "tuple element {i} has type '{t}' which is not keyable"
                ))
            } else {
                None
            }
        }),
        Type::NamedTuple(fields) => fields.iter().find_map(|(label, t)| {
            if !is_keyable(t, tc) {
                Some(format!(
                    "field '{label}' has type '{t}' which is not keyable"
                ))
            } else {
                None
            }
        }),
        Type::Enum { name, type_args } => {
            if ty.is_option() {
                return Some("optional types cannot be used as map keys".to_string());
            }
            let Some(enum_def) = tc.enum_defs.get(name).cloned() else {
                return Some(format!("enum '{name}' is not known"));
            };
            let subst: HashMap<_, _> = enum_def
                .type_params
                .iter()
                .zip(type_args.iter())
                .map(|(param, arg)| (param.id, arg.clone()))
                .collect();
            enum_def.variants.iter().find_map(|v| {
                let offending = match &v.kind {
                    VariantKind::Unit => None,
                    VariantKind::Tuple(types) => types.iter().find_map(|t| {
                        let resolved = subst_type(t, &subst);
                        if !is_keyable(&resolved, tc) {
                            Some(resolved)
                        } else {
                            None
                        }
                    }),
                    VariantKind::Struct(fields) => fields.iter().find_map(|f| {
                        let resolved = subst_type(&f.ty, &subst);
                        if !is_keyable(&resolved, tc) {
                            Some(resolved)
                        } else {
                            None
                        }
                    }),
                };
                offending.map(|bad_ty| {
                    format!(
                        "variant '{}' has payload type '{bad_ty}' which is not keyable",
                        v.name
                    )
                })
            })
        }
        Type::Struct { name, type_args } => {
            let Some(struct_def) = tc.struct_defs.get(name).cloned() else {
                return Some(format!("struct '{name}' is not known"));
            };
            let subst: HashMap<_, _> = struct_def
                .type_params
                .iter()
                .zip(type_args.iter())
                .map(|(param, arg)| (param.id, arg.clone()))
                .collect();
            struct_def.fields.iter().find_map(|f| {
                let resolved = subst_type(&f.ty, &subst);
                if !is_keyable(&resolved, tc) {
                    Some(format!(
                        "field '{}' has type '{resolved}' which is not keyable",
                        f.name
                    ))
                } else {
                    None
                }
            })
        }
        other => Some(format!("type '{other}' is not keyable")),
    }
}
