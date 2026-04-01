#![allow(clippy::result_large_err)]

use std::collections::{HashMap, HashSet, VecDeque};

use super::{
    error::{Diagnostic, DiagnosticKind},
    types::ModuleDef,
};
use crate::{
    ast::{
        BinaryOp, ConstDeclNode, ExprKind, ExprNode, FloatSuffix, FormatKind, FormatSpec, Ident,
        ImportKind, Lit, Stmt, StmtNode, StringPart, Type, UnaryOp, Visibility,
    },
    span::Span,
};

#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    Int(i64),
    Float(f32),
    Double(f64),
    Bool(bool),
    String(String),
    Nil,
}

impl ConstValue {
    pub fn ty(&self) -> Type {
        match self {
            ConstValue::Int(_) => Type::Int,
            ConstValue::Float(_) => Type::Float,
            ConstValue::Double(_) => Type::Double,
            ConstValue::Bool(_) => Type::Bool,
            ConstValue::String(_) => Type::String,
            ConstValue::Nil => {
                unreachable!("Nil is polymorphic; do not call ty() on ConstValue::Nil")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConstDef {
    pub ty: Type,
    pub value: ConstValue,
    pub visibility: Visibility,
}

pub(super) fn collect_const_decls(stmts: &[StmtNode]) -> Vec<&ConstDeclNode> {
    stmts
        .iter()
        .filter_map(|s| {
            if let Stmt::Const(node) = &s.node {
                Some(node)
            } else {
                None
            }
        })
        .collect()
}

/// Returns indices into `decls` in topological evaluation order.
pub(super) fn build_const_dependency_graph(
    decls: &[&ConstDeclNode],
) -> Result<Vec<usize>, Diagnostic> {
    // Map const name -> index
    let mut name_to_idx: HashMap<Ident, usize> = HashMap::new();
    for (i, decl) in decls.iter().enumerate() {
        let name = decl.node.name;
        if name_to_idx.contains_key(&name) {
            return Err(Diagnostic::new(
                decl.span,
                DiagnosticKind::DuplicateConst { name },
            ));
        }
        name_to_idx.insert(name, i);
    }

    // Build adjacency list (edges: i -> j means i depends on j)
    let n = decls.len();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for (i, decl) in decls.iter().enumerate() {
        let mut deps: HashSet<usize> = HashSet::new();
        collect_expr_ident_refs(&decl.node.value, &name_to_idx, &mut deps);
        for j in deps {
            adj[j].push(i);
            in_degree[i] += 1;
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<usize> = in_degree
        .iter()
        .enumerate()
        .filter(|&(_, &d)| d == 0)
        .map(|(i, _)| i)
        .collect();

    let mut order = vec![];
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &dep in &adj[node] {
            in_degree[dep] -= 1;
            if in_degree[dep] == 0 {
                queue.push_back(dep);
            }
        }
    }

    if order.len() != n {
        // Find a node still in a cycle
        let cycle_node = in_degree
            .iter()
            .enumerate()
            .find(|&(_, &d)| d > 0)
            .map(|(i, _)| i)
            .expect("cycle must exist if order is incomplete");
        let decl = decls[cycle_node];
        return Err(Diagnostic::new(
            decl.span,
            DiagnosticKind::CircularConstDependency {
                name: decl.node.name,
            },
        ));
    }

    Ok(order)
}

fn collect_expr_ident_refs(
    expr: &ExprNode,
    const_names: &HashMap<Ident, usize>,
    deps: &mut HashSet<usize>,
) {
    match &expr.node.kind {
        ExprKind::Ident(name) => {
            if let Some(&idx) = const_names.get(name) {
                deps.insert(idx);
            }
        }
        ExprKind::Binary(bin) => {
            collect_expr_ident_refs(&bin.node.left, const_names, deps);
            collect_expr_ident_refs(&bin.node.right, const_names, deps);
        }
        ExprKind::Unary(un) => {
            collect_expr_ident_refs(&un.node.expr, const_names, deps);
        }
        ExprKind::Cast(cast_node) => {
            collect_expr_ident_refs(&cast_node.node.expr, const_names, deps);
        }
        ExprKind::StringInterp(parts) => {
            for part in parts {
                if let StringPart::Expr(e, _) = part {
                    collect_expr_ident_refs(e, const_names, deps);
                }
            }
        }
        _ => {}
    }
}

pub(super) fn validate_const_expr(
    expr: &ExprNode,
    known_consts: &HashSet<Ident>,
) -> Result<(), Diagnostic> {
    match &expr.node.kind {
        ExprKind::Lit(_) => Ok(()),
        ExprKind::Ident(name) => {
            if known_consts.contains(name) {
                Ok(())
            } else {
                Err(Diagnostic::new(
                    expr.span,
                    DiagnosticKind::NotConstantExpression,
                ))
            }
        }
        ExprKind::Binary(bin) => {
            validate_const_expr(&bin.node.left, known_consts)?;
            validate_const_expr(&bin.node.right, known_consts)
        }
        ExprKind::Unary(un) => validate_const_expr(&un.node.expr, known_consts),
        ExprKind::Cast(cast_node) => {
            let target = &cast_node.node.target;
            let is_numeric = matches!(target, Type::Int | Type::Float | Type::Double);
            if !is_numeric {
                return Err(Diagnostic::new(
                    expr.span,
                    DiagnosticKind::NotConstantExpression,
                ));
            }
            validate_const_expr(&cast_node.node.expr, known_consts)
        }
        ExprKind::StringInterp(parts) => {
            for part in parts {
                if let StringPart::Expr(e, _) = part {
                    validate_const_expr(e, known_consts)?;
                }
            }
            Ok(())
        }
        _ => Err(Diagnostic::new(
            expr.span,
            DiagnosticKind::NotConstantExpression,
        )),
    }
}

pub(super) fn eval_const_expr(
    expr: &ExprNode,
    const_defs: &HashMap<Ident, ConstDef>,
) -> Result<ConstValue, Diagnostic> {
    match &expr.node.kind {
        ExprKind::Lit(lit) => eval_lit(lit),
        ExprKind::Ident(name) => {
            if let Some(def) = const_defs.get(name) {
                Ok(def.value.clone())
            } else {
                Err(Diagnostic::new(
                    expr.span,
                    DiagnosticKind::NotConstantExpression,
                ))
            }
        }
        ExprKind::Binary(bin) => {
            let left = eval_const_expr(&bin.node.left, const_defs)?;
            let right = eval_const_expr(&bin.node.right, const_defs)?;
            eval_binary(left, bin.node.op, right, bin.span)
        }
        ExprKind::Unary(un) => {
            let val = eval_const_expr(&un.node.expr, const_defs)?;
            eval_unary(un.node.op, val, un.span)
        }
        ExprKind::Cast(cast_node) => {
            let val = eval_const_expr(&cast_node.node.expr, const_defs)?;
            eval_cast(val, &cast_node.node.target, cast_node.span)
        }
        ExprKind::StringInterp(parts) => {
            let mut result = String::new();
            for part in parts {
                match part {
                    StringPart::Text(s) => result.push_str(s),
                    StringPart::Expr(e, fmt) => {
                        let val = eval_const_expr(e, const_defs)?;
                        let s = match fmt {
                            Some(spanned_spec) => const_format_value(&val, &spanned_spec.node),
                            None => const_value_to_string(&val),
                        };
                        result.push_str(&s);
                    }
                }
            }
            Ok(ConstValue::String(result))
        }
        _ => Err(Diagnostic::new(
            expr.span,
            DiagnosticKind::NotConstantExpression,
        )),
    }
}

fn eval_lit(lit: &Lit) -> Result<ConstValue, Diagnostic> {
    match lit {
        Lit::Int(n) => Ok(ConstValue::Int(*n)),
        Lit::Float { value, suffix } => match suffix {
            Some(FloatSuffix::D) => Ok(ConstValue::Double(*value)),
            _ => Ok(ConstValue::Float(*value as f32)),
        },
        Lit::Bool(b) => Ok(ConstValue::Bool(*b)),
        Lit::String(s) => Ok(ConstValue::String(s.clone())),
        Lit::Nil => Ok(ConstValue::Nil),
    }
}

macro_rules! const_binary_arms {
    (
        ($left:expr, $op:expr, $right:expr);
        $( $variant:ident => $result:ident { $( $bin_op:ident => $rust_op:tt ),+ $(,)? } ),+
        $(,)?
    ) => {
        match ($left, $op, $right) {
            $($(
                (ConstValue::$variant(l), BinaryOp::$bin_op, ConstValue::$variant(r)) =>
                    Some(ConstValue::$result(l $rust_op r)),
            )+)+
            _ => None,
        }
    };
}

fn eval_binary(
    left: ConstValue,
    op: BinaryOp,
    right: ConstValue,
    span: Span,
) -> Result<ConstValue, Diagnostic> {
    use BinaryOp::{Add, Div, Mul, Rem, Shl, Shr, Sub};
    use ConstValue::{Int, String};

    match (left, op, right) {
        // Int division by zero (must precede Div/Rem arms)
        (Int(_), Div | Rem, Int(0)) => {
            Err(Diagnostic::new(span, DiagnosticKind::ConstDivisionByZero))
        }

        // Int checked arithmetic (overflow detection)
        (Int(l), Add, Int(r)) => l
            .checked_add(r)
            .map(Int)
            .ok_or_else(|| Diagnostic::new(span, DiagnosticKind::ConstIntegerOverflow)),
        (Int(l), Sub, Int(r)) => l
            .checked_sub(r)
            .map(Int)
            .ok_or_else(|| Diagnostic::new(span, DiagnosticKind::ConstIntegerOverflow)),
        (Int(l), Mul, Int(r)) => l
            .checked_mul(r)
            .map(Int)
            .ok_or_else(|| Diagnostic::new(span, DiagnosticKind::ConstIntegerOverflow)),
        (Int(l), Div, Int(r)) => l
            .checked_div(r)
            .map(Int)
            .ok_or_else(|| Diagnostic::new(span, DiagnosticKind::ConstIntegerOverflow)),
        (Int(l), Rem, Int(r)) => l
            .checked_rem(r)
            .map(Int)
            .ok_or_else(|| Diagnostic::new(span, DiagnosticKind::ConstIntegerOverflow)),

        // Int shifts (wrapping)
        (Int(l), Shl, Int(r)) => Ok(Int(l.wrapping_shl(r as u32))),
        (Int(l), Shr, Int(r)) => Ok(Int(l.wrapping_shr(r as u32))),

        // String concatenation
        (String(l), Add, String(r)) => Ok(String(l + &r)),

        // All remaining simple operator cases
        (left, op, right) => const_binary_arms!((left, op, right);
            Int => Int { Xor => ^, BitAnd => &, BitOr => | },
            Float => Float { Add => +, Sub => -, Mul => *, Div => /, Rem => % },
            Double => Double { Add => +, Sub => -, Mul => *, Div => /, Rem => % },
            Int => Bool { Eq => ==, NotEq => !=, LessThan => <, GreaterThan => >, LessThanEq => <=, GreaterThanEq => >= },
            Float => Bool { Eq => ==, NotEq => !=, LessThan => <, GreaterThan => >, LessThanEq => <=, GreaterThanEq => >= },
            Double => Bool { Eq => ==, NotEq => !=, LessThan => <, GreaterThan => >, LessThanEq => <=, GreaterThanEq => >= },
            Bool => Bool { Eq => ==, NotEq => !=, And => &&, Or => ||, Xor => ^ },
            String => Bool { Eq => ==, NotEq => != },
        )
        .ok_or_else(|| Diagnostic::new(span, DiagnosticKind::NotConstantExpression)),
    }
}

fn eval_unary(op: UnaryOp, val: ConstValue, span: Span) -> Result<ConstValue, Diagnostic> {
    use ConstValue::{Bool, Double, Float, Int};
    match (op, val) {
        (UnaryOp::Neg, Int(n)) => n
            .checked_neg()
            .map(Int)
            .ok_or_else(|| Diagnostic::new(span, DiagnosticKind::ConstIntegerOverflow)),
        (UnaryOp::Neg, Float(f)) => Ok(Float(-f)),
        (UnaryOp::Neg, Double(d)) => Ok(Double(-d)),
        (UnaryOp::Not, Bool(b)) => Ok(Bool(!b)),
        (UnaryOp::BitNot, Int(n)) => Ok(Int(!n)),
        _ => Err(Diagnostic::new(span, DiagnosticKind::NotConstantExpression)),
    }
}

fn const_value_to_string(val: &ConstValue) -> String {
    match val {
        ConstValue::Int(n) => n.to_string(),
        ConstValue::Float(f) => {
            if f.fract() == 0.0 && f.is_finite() {
                format!("{f:.1}")
            } else {
                format!("{f}")
            }
        }
        ConstValue::Double(d) => {
            if d.fract() == 0.0 && d.is_finite() {
                format!("{d:.1}")
            } else {
                format!("{d}")
            }
        }
        ConstValue::Bool(b) => b.to_string(),
        ConstValue::String(s) => s.clone(),
        ConstValue::Nil => "None".to_string(),
    }
}

fn const_format_value(val: &ConstValue, spec: &FormatSpec) -> String {
    let raw = const_format_raw(val, spec);
    let is_numeric = matches!(
        val,
        ConstValue::Int(_) | ConstValue::Float(_) | ConstValue::Double(_)
    );
    let signed = spec.apply_sign(&raw, is_numeric);
    let is_string = matches!(val, ConstValue::String(_));
    spec.apply_padding(&signed, is_string)
}

fn const_format_raw(val: &ConstValue, spec: &FormatSpec) -> String {
    match spec.kind {
        FormatKind::Default => match (val, spec.precision) {
            (ConstValue::Float(v), Some(prec)) => format!("{:.prec$}", v, prec = prec as usize),
            (ConstValue::Double(v), Some(prec)) => format!("{:.prec$}", v, prec = prec as usize),
            (ConstValue::String(s), Some(prec)) => s.chars().take(prec as usize).collect(),
            _ => const_value_to_string(val),
        },
        FormatKind::Hex => {
            let ConstValue::Int(n) = val else {
                unreachable!("typechecker validated")
            };
            format!("{n:x}")
        }
        FormatKind::HexUpper => {
            let ConstValue::Int(n) = val else {
                unreachable!("typechecker validated")
            };
            format!("{n:X}")
        }
        FormatKind::Binary => {
            let ConstValue::Int(n) = val else {
                unreachable!("typechecker validated")
            };
            format!("{n:b}")
        }
        FormatKind::Exp => match (val, spec.precision) {
            (ConstValue::Float(v), Some(prec)) => format!("{:.prec$e}", v, prec = prec as usize),
            (ConstValue::Float(v), None) => format!("{v:e}"),
            (ConstValue::Double(v), Some(prec)) => format!("{:.prec$e}", v, prec = prec as usize),
            (ConstValue::Double(v), None) => format!("{v:e}"),
            _ => unreachable!("typechecker validated"),
        },
        FormatKind::ExpUpper => match (val, spec.precision) {
            (ConstValue::Float(v), Some(prec)) => format!("{:.prec$E}", v, prec = prec as usize),
            (ConstValue::Float(v), None) => format!("{v:E}"),
            (ConstValue::Double(v), Some(prec)) => format!("{:.prec$E}", v, prec = prec as usize),
            (ConstValue::Double(v), None) => format!("{v:E}"),
            _ => unreachable!("typechecker validated"),
        },
    }
}

fn eval_cast(val: ConstValue, target: &Type, span: Span) -> Result<ConstValue, Diagnostic> {
    use ConstValue::{Double, Float, Int};
    match (&val, target) {
        (Int(_), Type::Int) | (Float(_), Type::Float) | (Double(_), Type::Double) => Ok(val),
        (Int(n), Type::Float) => Ok(Float(*n as f32)),
        (Int(n), Type::Double) => Ok(Double(*n as f64)),
        (Float(f), Type::Int) => Ok(Int(*f as i64)),
        (Float(f), Type::Double) => Ok(Double(f64::from(*f))),
        (Double(d), Type::Int) => Ok(Int(*d as i64)),
        (Double(d), Type::Float) => Ok(Float(*d as f32)),
        _ => Err(Diagnostic::new(span, DiagnosticKind::NotConstantExpression)),
    }
}

/// Evaluates all const declarations in `stmts`, injecting imported consts from
/// already-resolved module defs (which must be in DFS post-order). Returns a
/// map of all evaluated consts (public and private) and any errors encountered.
pub(super) fn evaluate_and_export_consts(
    stmts: &[StmtNode],
    resolved_module_defs: &HashMap<Vec<String>, ModuleDef>,
) -> (HashMap<Ident, ConstDef>, Vec<Diagnostic>) {
    let mut local_consts: HashMap<Ident, ConstDef> = HashMap::new();
    let mut errors = vec![];

    // Inject imported consts from dependency modules.
    for stmt in stmts {
        let Stmt::Import(node) = &stmt.node else {
            continue;
        };
        let import = &node.node;
        let path_key: Vec<String> = import.path.iter().map(ToString::to_string).collect();
        let Some(module_def) = resolved_module_defs.get(&path_key) else {
            continue;
        };

        match &import.kind {
            ImportKind::Selective(items) => {
                for item in items {
                    let bind_as = item.alias.unwrap_or(item.name);
                    if let Some(def) = module_def.const_defs.get(&item.name) {
                        local_consts.insert(bind_as, def.clone());
                    }
                }
            }
            ImportKind::Wildcard => {
                for (name, def) in &module_def.const_defs {
                    local_consts.insert(*name, def.clone());
                }
            }
            // Module and ModuleAs bindings use qualified access (mod.NAME),
            // which is not a plain Ident and thus not reachable via validate_const_expr.
            ImportKind::Module | ImportKind::ModuleAs(_) => {}
        }
    }

    let decls = collect_const_decls(stmts);
    if decls.is_empty() {
        return (local_consts, errors);
    }

    let order = match build_const_dependency_graph(&decls) {
        Ok(order) => order,
        Err(err) => {
            errors.push(err);
            return (local_consts, errors);
        }
    };

    for idx in order {
        let decl = decls[idx];
        let name = decl.node.name;

        let known_consts: HashSet<Ident> = local_consts.keys().copied().collect();
        if let Err(err) = validate_const_expr(&decl.node.value, &known_consts) {
            errors.push(err);
            continue;
        }

        let const_value = match eval_const_expr(&decl.node.value, &local_consts) {
            Ok(val) => val,
            Err(err) => {
                errors.push(err);
                continue;
            }
        };

        let value_ty = const_value.ty();
        if let Some(ann_ty) = &decl.node.ty
            && *ann_ty != value_ty
        {
            errors.push(Diagnostic::new(
                decl.span,
                DiagnosticKind::ConstTypeMismatch {
                    expected: ann_ty.clone(),
                    got: value_ty,
                },
            ));
            continue;
        }

        let final_ty = decl.node.ty.clone().unwrap_or_else(|| const_value.ty());
        local_consts.insert(
            name,
            ConstDef {
                ty: final_ty,
                value: const_value,
                visibility: decl.node.visibility,
            },
        );
    }

    (local_consts, errors)
}
