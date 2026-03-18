use crate::ast::{BinaryOp, Ident, Type, UnaryOp};
use crate::builtin::Builtin;
use crate::span::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub funcs: Vec<Func>,
    pub externs: Vec<ExternDecl>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ExternId(pub u32);

#[derive(Debug, Clone, PartialEq)]
pub struct ExternDecl {
    pub id: ExternId,
    pub name: Ident,
    pub params: Vec<Type>,
    pub ret: Type,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct LocalId(pub u32);

#[derive(Debug, Clone, PartialEq)]
pub struct Func {
    pub id: FuncId,
    pub name: Ident,

    pub locals: Vec<Local>,
    pub params_len: u32,

    pub ret: Type,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Local {
    pub name: Option<Ident>,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Stmt {
    pub span: Span,
    pub kind: StmtKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StmtKind {
    Let {
        local: LocalId,
        init: Expr,
    },
    Assign {
        local: LocalId,
        value: Expr,
    },
    Expr(Expr),
    Return(Option<Expr>),

    If {
        cond: Expr,
        then_block: Block,
        else_block: Option<Block>,
    },

    While {
        cond: Expr,
        body: Block,
    },

    Break,
    Continue,

    SetField {
        object: LocalId,
        field_index: u16,
        value: Expr,
    },

    Match {
        scrutinee_init: Box<Expr>, // evaluated once and stored to scrutinee local
        scrutinee: LocalId,
        arms: Vec<MatchArm>,
        else_body: Option<MatchElse>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub variant: u16,
    pub bindings: Vec<MatchBinding>,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchBinding {
    pub field_index: u16,
    pub local: LocalId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchElse {
    pub binding: Option<LocalId>, // some for pattern::Ident, none for wildcard
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub ty: Type,
    pub span: Span,
    pub kind: ExprKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Local(LocalId),

    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Nil,

    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },

    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    Call {
        func: FuncId,
        args: Vec<Expr>,
    },

    CallBuiltin {
        builtin: Builtin,
        args: Vec<Expr>,
    },

    CallExtern {
        extern_id: ExternId,
        args: Vec<Expr>,
    },

    StructLiteral {
        type_id: u32,
        fields: Vec<Expr>, // in declaration order
    },

    TupleLiteral {
        elements: Vec<Expr>,
    },

    FieldGet {
        object: Box<Expr>,
        index: u16,
    },

    TupleIndex {
        tuple: Box<Expr>,
        index: u16,
    },

    EnumLiteral {
        type_id: u32,
        variant: u16,
        fields: Vec<Expr>, // in declaration order
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{
        dummy_ident, dummy_span, hir_bool_expr as bool_expr, hir_int_expr as int_expr,
    };

    #[test]
    fn func_id_is_copy_and_eq() {
        let a = FuncId(0);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn local_id_is_copy_and_eq() {
        let a = LocalId(3);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn different_func_ids_are_not_equal() {
        assert_ne!(FuncId(0), FuncId(1));
    }

    #[test]
    fn different_local_ids_are_not_equal() {
        assert_ne!(LocalId(0), LocalId(1));
    }

    #[test]
    fn empty_program() {
        let prog = Program {
            funcs: vec![],
            externs: vec![],
        };
        assert!(prog.funcs.is_empty());
    }

    #[test]
    fn program_with_empty_func() {
        let func = Func {
            id: FuncId(0),
            name: dummy_ident("main"),
            locals: vec![],
            params_len: 0,
            ret: Type::Void,
            body: Block { stmts: vec![] },
            span: dummy_span(),
        };
        let prog = Program {
            funcs: vec![func],
            externs: vec![],
        };
        assert_eq!(prog.funcs.len(), 1);
        assert_eq!(prog.funcs[0].name.to_string(), "main");
    }

    #[test]
    fn func_with_params_and_locals() {
        let func = Func {
            id: FuncId(0),
            name: dummy_ident("add"),
            locals: vec![
                Local {
                    name: Some(dummy_ident("a")),
                    ty: Type::Int,
                },
                Local {
                    name: Some(dummy_ident("b")),
                    ty: Type::Int,
                },
                Local {
                    name: Some(dummy_ident("result")),
                    ty: Type::Int,
                },
            ],
            params_len: 2,
            ret: Type::Int,
            body: Block { stmts: vec![] },
            span: dummy_span(),
        };
        assert_eq!(func.params_len, 2);
        assert_eq!(func.locals.len(), 3);
    }

    #[test]
    fn stmt_let() {
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::Let {
                local: LocalId(0),
                init: int_expr(42),
            },
        };
        assert!(matches!(stmt.kind, StmtKind::Let { .. }));
    }

    #[test]
    fn stmt_assign() {
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::Assign {
                local: LocalId(0),
                value: int_expr(10),
            },
        };
        assert!(matches!(stmt.kind, StmtKind::Assign { .. }));
    }

    #[test]
    fn stmt_expr() {
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::Expr(int_expr(1)),
        };
        assert!(matches!(stmt.kind, StmtKind::Expr(_)));
    }

    #[test]
    fn stmt_return_with_value() {
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::Return(Some(int_expr(0))),
        };
        assert!(matches!(stmt.kind, StmtKind::Return(Some(_))));
    }

    #[test]
    fn stmt_return_void() {
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::Return(None),
        };
        assert!(matches!(stmt.kind, StmtKind::Return(None)));
    }

    #[test]
    fn stmt_if_without_else() {
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::If {
                cond: bool_expr(true),
                then_block: Block { stmts: vec![] },
                else_block: None,
            },
        };
        assert!(matches!(
            stmt.kind,
            StmtKind::If {
                else_block: None,
                ..
            }
        ));
    }

    #[test]
    fn stmt_if_with_else() {
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::If {
                cond: bool_expr(true),
                then_block: Block { stmts: vec![] },
                else_block: Some(Block { stmts: vec![] }),
            },
        };
        assert!(matches!(
            stmt.kind,
            StmtKind::If {
                else_block: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn stmt_while() {
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::While {
                cond: bool_expr(true),
                body: Block { stmts: vec![] },
            },
        };
        assert!(matches!(stmt.kind, StmtKind::While { .. }));
    }

    #[test]
    fn stmt_break() {
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::Break,
        };
        assert!(matches!(stmt.kind, StmtKind::Break));
    }

    #[test]
    fn stmt_continue() {
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::Continue,
        };
        assert!(matches!(stmt.kind, StmtKind::Continue));
    }

    #[test]
    fn expr_local() {
        let expr = Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::Local(LocalId(0)),
        };
        assert!(matches!(expr.kind, ExprKind::Local(LocalId(0))));
    }

    #[test]
    fn expr_int() {
        let expr = int_expr(42);
        assert!(matches!(expr.kind, ExprKind::Int(42)));
    }

    #[test]
    fn expr_float() {
        let expr = Expr {
            ty: Type::Float,
            span: dummy_span(),
            kind: ExprKind::Float(3.14),
        };
        assert!(matches!(expr.kind, ExprKind::Float(v) if (v - 3.14).abs() < f64::EPSILON));
    }

    #[test]
    fn expr_bool() {
        let expr = bool_expr(true);
        assert!(matches!(expr.kind, ExprKind::Bool(true)));
    }

    #[test]
    fn expr_string() {
        let expr = Expr {
            ty: Type::String,
            span: dummy_span(),
            kind: ExprKind::String("hello".into()),
        };
        assert!(matches!(expr.kind, ExprKind::String(ref s) if s == "hello"));
    }

    #[test]
    fn expr_nil() {
        let expr = Expr {
            ty: Type::Void,
            span: dummy_span(),
            kind: ExprKind::Nil,
        };
        assert!(matches!(expr.kind, ExprKind::Nil));
    }

    #[test]
    fn expr_unary() {
        let expr = Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::Unary {
                op: UnaryOp::Neg,
                expr: Box::new(int_expr(1)),
            },
        };
        assert!(matches!(
            expr.kind,
            ExprKind::Unary {
                op: UnaryOp::Neg,
                ..
            }
        ));
    }

    #[test]
    fn expr_binary() {
        let expr = Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::Binary {
                op: BinaryOp::Add,
                lhs: Box::new(int_expr(1)),
                rhs: Box::new(int_expr(2)),
            },
        };
        assert!(matches!(
            expr.kind,
            ExprKind::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    #[test]
    fn expr_call() {
        let expr = Expr {
            ty: Type::Void,
            span: dummy_span(),
            kind: ExprKind::Call {
                func: FuncId(0),
                args: vec![int_expr(1)],
            },
        };
        assert!(matches!(expr.kind, ExprKind::Call { .. }));
    }

    #[test]
    fn expr_call_builtin() {
        let expr = Expr {
            ty: Type::Void,
            span: dummy_span(),
            kind: ExprKind::CallBuiltin {
                builtin: Builtin::Println,
                args: vec![Expr {
                    ty: Type::String,
                    span: dummy_span(),
                    kind: ExprKind::String("hi".into()),
                }],
            },
        };
        assert!(matches!(
            expr.kind,
            ExprKind::CallBuiltin {
                builtin: Builtin::Println,
                ..
            }
        ));
    }

    #[test]
    fn expr_call_builtin_assert_msg() {
        let expr = Expr {
            ty: Type::Void,
            span: dummy_span(),
            kind: ExprKind::CallBuiltin {
                builtin: Builtin::AssertMsg,
                args: vec![
                    bool_expr(true),
                    Expr {
                        ty: Type::String,
                        span: dummy_span(),
                        kind: ExprKind::String("ok".into()),
                    },
                ],
            },
        };
        assert!(matches!(
            expr.kind,
            ExprKind::CallBuiltin {
                builtin: Builtin::AssertMsg,
                ..
            }
        ));
    }

    #[test]
    fn expr_struct_literal() {
        let expr = Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::StructLiteral {
                type_id: 7,
                fields: vec![int_expr(10), int_expr(20)],
            },
        };
        assert!(matches!(
            expr.kind,
            ExprKind::StructLiteral { type_id: 7, .. }
        ));
    }

    #[test]
    fn expr_tuple_literal() {
        let expr = Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::TupleLiteral {
                elements: vec![int_expr(1), bool_expr(false)],
            },
        };
        assert!(matches!(expr.kind, ExprKind::TupleLiteral { .. }));
    }

    #[test]
    fn expr_field_get() {
        let obj = int_expr(0);
        let expr = Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::FieldGet {
                object: Box::new(obj),
                index: 2,
            },
        };
        assert!(matches!(expr.kind, ExprKind::FieldGet { index: 2, .. }));
    }

    #[test]
    fn expr_tuple_index() {
        let tup = int_expr(0);
        let expr = Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::TupleIndex {
                tuple: Box::new(tup),
                index: 1,
            },
        };
        assert!(matches!(expr.kind, ExprKind::TupleIndex { index: 1, .. }));
    }

    #[test]
    fn stmt_set_field() {
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::SetField {
                object: LocalId(0),
                field_index: 3,
                value: int_expr(99),
            },
        };
        assert!(matches!(
            stmt.kind,
            StmtKind::SetField { field_index: 3, .. }
        ));
    }

    #[test]
    fn expr_enum_literal_unit() {
        let expr = Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::EnumLiteral {
                type_id: 5,
                variant: 0,
                fields: vec![],
            },
        };
        assert!(matches!(
            expr.kind,
            ExprKind::EnumLiteral {
                type_id: 5,
                variant: 0,
                ..
            }
        ));
    }

    #[test]
    fn expr_enum_literal_with_fields() {
        let expr = Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::EnumLiteral {
                type_id: 2,
                variant: 1,
                fields: vec![int_expr(42)],
            },
        };
        assert!(matches!(
            expr.kind,
            ExprKind::EnumLiteral {
                type_id: 2,
                variant: 1,
                ..
            }
        ));
        if let ExprKind::EnumLiteral { fields, .. } = &expr.kind {
            assert_eq!(fields.len(), 1);
        }
    }

    #[test]
    fn stmt_match_with_arms() {
        let arm = MatchArm {
            variant: 0,
            bindings: vec![],
            body: Block { stmts: vec![] },
        };
        let stmt = Stmt {
            span: dummy_span(),
            kind: StmtKind::Match {
                scrutinee_init: Box::new(int_expr(0)),
                scrutinee: LocalId(0),
                arms: vec![arm],
                else_body: None,
            },
        };
        assert!(matches!(stmt.kind, StmtKind::Match { .. }));
        if let StmtKind::Match {
            arms, else_body, ..
        } = &stmt.kind
        {
            assert_eq!(arms.len(), 1);
            assert!(else_body.is_none());
        }
    }

    #[test]
    fn match_arm_with_bindings() {
        let binding = MatchBinding {
            field_index: 0,
            local: LocalId(1),
        };
        let arm = MatchArm {
            variant: 2,
            bindings: vec![binding],
            body: Block { stmts: vec![] },
        };
        assert_eq!(arm.variant, 2);
        assert_eq!(arm.bindings.len(), 1);
        assert_eq!(arm.bindings[0].field_index, 0);
    }

    #[test]
    fn match_else_with_binding() {
        let else_b = MatchElse {
            binding: Some(LocalId(3)),
            body: Block { stmts: vec![] },
        };
        assert!(else_b.binding.is_some());
    }

    #[test]
    fn match_else_wildcard() {
        let else_b = MatchElse {
            binding: None,
            body: Block { stmts: vec![] },
        };
        assert!(else_b.binding.is_none());
    }

    #[test]
    fn func_with_let_and_return() {
        let body = Block {
            stmts: vec![
                Stmt {
                    span: dummy_span(),
                    kind: StmtKind::Let {
                        local: LocalId(0),
                        init: int_expr(42),
                    },
                },
                Stmt {
                    span: dummy_span(),
                    kind: StmtKind::Return(Some(Expr {
                        ty: Type::Int,
                        span: dummy_span(),
                        kind: ExprKind::Local(LocalId(0)),
                    })),
                },
            ],
        };
        let func = Func {
            id: FuncId(0),
            name: dummy_ident("answer"),
            locals: vec![Local {
                name: Some(dummy_ident("x")),
                ty: Type::Int,
            }],
            params_len: 0,
            ret: Type::Int,
            body,
            span: dummy_span(),
        };
        assert_eq!(func.body.stmts.len(), 2);
        assert!(matches!(func.body.stmts[0].kind, StmtKind::Let { .. }));
        assert!(matches!(func.body.stmts[1].kind, StmtKind::Return(Some(_))));
    }
}
