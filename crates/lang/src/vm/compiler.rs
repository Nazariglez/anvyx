use std::fmt;
use std::rc::Rc;

use crate::ast::{BinaryOp, UnaryOp};
use crate::builtin::Builtin;
use crate::hir;

use super::bytecode::{Chunk, Op};
use super::value::Value;

#[derive(Debug)]
pub enum CompileError {
    NoMainFunction,
    TooManyConstants { func_name: String },
    TooManyLocals { func_name: String },
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoMainFunction => write!(f, "no main function found"),
            Self::TooManyConstants { func_name } => {
                write!(f, "too many constants in function '{func_name}'")
            }
            Self::TooManyLocals { func_name } => {
                write!(f, "too many locals in function '{func_name}'")
            }
        }
    }
}

pub struct CompiledProgram {
    pub chunks: Vec<Chunk>,
    pub main_idx: usize,
}

struct LoopState {
    start: usize,
    break_patches: Vec<usize>,
}

struct FuncCompiler {
    chunk: Chunk,
    loop_stack: Vec<LoopState>,
}

impl FuncCompiler {
    fn new(name: impl Into<String>, local_count: u16, params_count: u8) -> Self {
        Self {
            chunk: Chunk::new(name, local_count, params_count),
            loop_stack: vec![],
        }
    }

    fn emit(&mut self, op: Op) {
        self.chunk.emit(op);
    }

    fn emit_jump(&mut self, op: Op) -> usize {
        self.chunk.emit_jump(op)
    }

    fn patch_jump(&mut self, pos: usize) {
        self.chunk.patch_jump(pos);
    }

    fn add_constant(&mut self, value: Value) -> Result<u16, CompileError> {
        let idx = self.chunk.constants.len();
        if idx > u16::MAX as usize {
            return Err(CompileError::TooManyConstants {
                func_name: self.chunk.name.clone(),
            });
        }
        Ok(self.chunk.add_constant(value))
    }
}

pub fn compile(hir: &hir::Program) -> Result<CompiledProgram, CompileError> {
    let mut chunks = vec![];

    for func in &hir.funcs {
        chunks.push(compile_func(func)?);
    }

    let main_idx = hir
        .funcs
        .iter()
        .position(|f| f.name.to_string() == "main")
        .ok_or(CompileError::NoMainFunction)?;

    Ok(CompiledProgram { chunks, main_idx })
}

fn compile_func(func: &hir::Func) -> Result<Chunk, CompileError> {
    let local_count = func.locals.len();
    if local_count > u16::MAX as usize {
        return Err(CompileError::TooManyLocals {
            func_name: func.name.to_string(),
        });
    }

    let mut fc = FuncCompiler::new(
        func.name.to_string(),
        local_count as u16,
        func.params_len as u8,
    );

    compile_block(&mut fc, &func.body)?;

    // implicit fallthrough, void return, unreachable if all paths already returned
    fc.emit(Op::Nil);
    fc.emit(Op::Return);

    Ok(fc.chunk)
}

fn compile_block(fc: &mut FuncCompiler, block: &hir::Block) -> Result<(), CompileError> {
    for stmt in &block.stmts {
        compile_stmt(fc, stmt)?;
    }
    Ok(())
}

fn compile_stmt(fc: &mut FuncCompiler, stmt: &hir::Stmt) -> Result<(), CompileError> {
    match &stmt.kind {
        hir::StmtKind::Let { local, init } => {
            compile_expr(fc, init)?;
            fc.emit(Op::SetLocal(local.0 as u16));
        }

        hir::StmtKind::Assign { local, value } => {
            compile_expr(fc, value)?;
            fc.emit(Op::SetLocal(local.0 as u16));
        }

        hir::StmtKind::Expr(expr) => {
            compile_expr(fc, expr)?;
            fc.emit(Op::Pop);
        }

        hir::StmtKind::Return(None) => {
            fc.emit(Op::Nil);
            fc.emit(Op::Return);
        }

        hir::StmtKind::Return(Some(expr)) => {
            compile_expr(fc, expr)?;
            fc.emit(Op::Return);
        }

        hir::StmtKind::If {
            cond,
            then_block,
            else_block,
        } => {
            compile_expr(fc, cond)?;
            let jump_pos = fc.emit_jump(Op::JumpIfFalse(0));

            compile_block(fc, then_block)?;

            match else_block {
                Some(else_b) => {
                    let skip_else_pos = fc.emit_jump(Op::Jump(0));
                    fc.patch_jump(jump_pos);
                    compile_block(fc, else_b)?;
                    fc.patch_jump(skip_else_pos);
                }
                None => {
                    fc.patch_jump(jump_pos);
                }
            }
        }

        hir::StmtKind::While { cond, body } => {
            let loop_start = fc.chunk.code.len();
            fc.loop_stack.push(LoopState {
                start: loop_start,
                break_patches: vec![],
            });

            compile_expr(fc, cond)?;
            let exit_jump_pos = fc.emit_jump(Op::JumpIfFalse(0));

            compile_block(fc, body)?;

            // back-jump to loop condition, offset is negative
            let back_offset = loop_start as isize - fc.chunk.code.len() as isize - 1;
            fc.emit(Op::Jump(back_offset as i16));

            // patch exit jump to land after the back-jump
            fc.patch_jump(exit_jump_pos);

            // patch all break jumps to the same landing spot
            let loop_state = fc.loop_stack.pop().expect("loop stack underflow");
            let after_loop = fc.chunk.code.len();
            for patch_pos in loop_state.break_patches {
                let offset = after_loop as isize - patch_pos as isize - 1;
                match &mut fc.chunk.code[patch_pos] {
                    Op::Jump(o) => *o = offset as i16,
                    _ => panic!("expected Jump at break patch position {patch_pos}"),
                }
            }
        }

        hir::StmtKind::Break => {
            let pos = fc.chunk.code.len();
            fc.emit(Op::Jump(0));
            fc.loop_stack
                .last_mut()
                .expect("break outside loop (should be caught earlier)")
                .break_patches
                .push(pos);
        }

        hir::StmtKind::Continue => {
            let loop_start = fc
                .loop_stack
                .last()
                .expect("continue outside loop (should be caught earlier)")
                .start;
            let back_offset = loop_start as isize - fc.chunk.code.len() as isize - 1;
            fc.emit(Op::Jump(back_offset as i16));
        }
    }

    Ok(())
}

fn compile_expr(fc: &mut FuncCompiler, expr: &hir::Expr) -> Result<(), CompileError> {
    match &expr.kind {
        hir::ExprKind::Local(id) => {
            fc.emit(Op::GetLocal(id.0 as u16));
        }

        hir::ExprKind::Int(v) => {
            let idx = fc.add_constant(Value::Int(*v))?;
            fc.emit(Op::Constant(idx));
        }

        hir::ExprKind::Float(v) => {
            let idx = fc.add_constant(Value::Float(*v))?;
            fc.emit(Op::Constant(idx));
        }

        hir::ExprKind::Bool(v) => {
            fc.emit(if *v { Op::True } else { Op::False });
        }

        hir::ExprKind::String(s) => {
            let idx = fc.add_constant(Value::String(Rc::from(s.as_str())))?;
            fc.emit(Op::Constant(idx));
        }

        hir::ExprKind::Nil => {
            fc.emit(Op::Nil);
        }

        hir::ExprKind::Unary { op, expr } => {
            compile_expr(fc, expr)?;
            match op {
                UnaryOp::Neg => fc.emit(Op::Negate),
                UnaryOp::Not => fc.emit(Op::Not),
            }
        }

        hir::ExprKind::Binary { op, lhs, rhs } => {
            compile_expr(fc, lhs)?;
            compile_expr(fc, rhs)?;
            let opcode = match op {
                BinaryOp::Add => Op::Add,
                BinaryOp::Sub => Op::Sub,
                BinaryOp::Mul => Op::Mul,
                BinaryOp::Div => Op::Div,
                BinaryOp::Rem => Op::Rem,
                BinaryOp::Eq => Op::Eq,
                BinaryOp::NotEq => Op::NotEq,
                BinaryOp::LessThan => Op::LessThan,
                BinaryOp::GreaterThan => Op::GreaterThan,
                BinaryOp::LessThanEq => Op::LessThanEq,
                BinaryOp::GreaterThanEq => Op::GreaterThanEq,
                BinaryOp::And => Op::And,
                BinaryOp::Or => Op::Or,
                BinaryOp::Xor => Op::Xor,
                BinaryOp::Coalesce => unreachable!("coalesce rejected during lowering"),
            };
            fc.emit(opcode);
        }

        hir::ExprKind::Call { func, args } => {
            for arg in args {
                compile_expr(fc, arg)?;
            }
            fc.emit(Op::Call(func.0 as u16, args.len() as u8));
        }

        hir::ExprKind::CallBuiltin { builtin, args } => {
            for arg in args {
                compile_expr(fc, arg)?;
            }
            let builtin_idx = Builtin::all()
                .iter()
                .position(|b| b == builtin)
                .expect("unknown builtin") as u8;
            fc.emit(Op::CallBuiltin(builtin_idx, args.len() as u8));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Ident;
    use crate::ast::Type;
    use crate::hir::{
        Block, Expr, ExprKind, Func, FuncId, Local, LocalId, Program, Stmt, StmtKind,
    };
    use crate::span::Span;
    use internment::Intern;

    fn dummy_span() -> Span {
        Span::new(0, 0)
    }

    fn dummy_ident(name: &str) -> Ident {
        Ident(Intern::new(name.to_string()))
    }

    fn int_expr(v: i64) -> Expr {
        Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::Int(v),
        }
    }

    fn bool_expr(v: bool) -> Expr {
        Expr {
            ty: Type::Bool,
            span: dummy_span(),
            kind: ExprKind::Bool(v),
        }
    }

    fn local_expr(id: u32) -> Expr {
        Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::Local(LocalId(id)),
        }
    }

    fn binary_expr(op: BinaryOp, lhs: Expr, rhs: Expr) -> Expr {
        Expr {
            ty: Type::Int,
            span: dummy_span(),
            kind: ExprKind::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
        }
    }

    fn stmt(kind: StmtKind) -> Stmt {
        Stmt {
            span: dummy_span(),
            kind,
        }
    }

    fn simple_func(
        name: &str,
        stmts: Vec<Stmt>,
        locals: Vec<Local>,
        params_len: u32,
        ret: Type,
    ) -> Func {
        Func {
            id: FuncId(0),
            name: dummy_ident(name),
            locals,
            params_len,
            ret,
            body: Block { stmts },
            span: dummy_span(),
        }
    }

    fn main_func(stmts: Vec<Stmt>) -> Func {
        simple_func("main", stmts, vec![], 0, Type::Void)
    }

    fn prog(func: Func) -> Program {
        Program { funcs: vec![func] }
    }

    #[test]
    fn empty_main_emits_nil_return() {
        let compiled = compile(&prog(main_func(vec![]))).unwrap();
        assert_eq!(compiled.main_idx, 0);
        let chunk = &compiled.chunks[0];
        assert_eq!(chunk.code, vec![Op::Nil, Op::Return]);
        assert_eq!(chunk.constants.len(), 0);
        assert_eq!(chunk.local_count, 0);
    }

    #[test]
    fn let_local_emits_set_local() {
        let func = Func {
            id: FuncId(0),
            name: dummy_ident("main"),
            locals: vec![Local {
                name: Some(dummy_ident("x")),
                ty: Type::Int,
            }],
            params_len: 0,
            ret: Type::Void,
            body: Block {
                stmts: vec![stmt(StmtKind::Let {
                    local: LocalId(0),
                    init: int_expr(42),
                })],
            },
            span: dummy_span(),
        };
        let compiled = compile(&prog(func)).unwrap();
        let chunk = &compiled.chunks[0];
        assert_eq!(
            chunk.code,
            vec![Op::Constant(0), Op::SetLocal(0), Op::Nil, Op::Return]
        );
        assert_eq!(chunk.constants, vec![Value::Int(42)]);
        assert_eq!(chunk.local_count, 1);
    }

    #[test]
    fn return_binary_add() {
        let func = Func {
            id: FuncId(0),
            name: dummy_ident("main"),
            locals: vec![],
            params_len: 0,
            ret: Type::Int,
            body: Block {
                stmts: vec![stmt(StmtKind::Return(Some(binary_expr(
                    BinaryOp::Add,
                    int_expr(1),
                    int_expr(2),
                ))))],
            },
            span: dummy_span(),
        };
        let compiled = compile(&prog(func)).unwrap();
        let chunk = &compiled.chunks[0];
        // return comes first from the explicit stmt, nil+return follows (unreachable)
        assert!(matches!(
            &chunk.code[..4],
            [Op::Constant(0), Op::Constant(1), Op::Add, Op::Return]
        ));
        assert_eq!(chunk.constants, vec![Value::Int(1), Value::Int(2)]);
    }

    #[test]
    fn bool_literals_use_true_false_ops() {
        let func = main_func(vec![
            stmt(StmtKind::Expr(bool_expr(true))),
            stmt(StmtKind::Expr(bool_expr(false))),
        ]);
        let compiled = compile(&prog(func)).unwrap();
        let chunk = &compiled.chunks[0];
        assert_eq!(chunk.code[0], Op::True);
        assert_eq!(chunk.code[2], Op::False);
    }

    #[test]
    fn if_without_else_emits_correct_jumps() {
        // if true { } (then_block is empty)
        let func = main_func(vec![stmt(StmtKind::If {
            cond: bool_expr(true),
            then_block: Block { stmts: vec![] },
            else_block: None,
        })]);
        let compiled = compile(&prog(func)).unwrap();
        let chunk = &compiled.chunks[0];
        // 0: True, 1: JumpIfFalse(?), 2: Nil, 3: Return
        assert_eq!(chunk.code[0], Op::True);
        assert!(matches!(chunk.code[1], Op::JumpIfFalse(_)));
        // JumpIfFalse should skip 0 instructions and land right after (at Nil)
        // offset = 2 - 1 - 1 = 0
        assert_eq!(chunk.code[1], Op::JumpIfFalse(0));
    }

    #[test]
    fn if_else_emits_skip_jump() {
        // if true { } else { }
        let func = main_func(vec![stmt(StmtKind::If {
            cond: bool_expr(true),
            then_block: Block { stmts: vec![] },
            else_block: Some(Block { stmts: vec![] }),
        })]);
        let compiled = compile(&prog(func)).unwrap();
        let chunk = &compiled.chunks[0];
        // 0: True, 1: JumpIfFalse(1), 2: Jump(0), 3: Nil, 4: Return
        assert_eq!(chunk.code[0], Op::True);
        assert_eq!(chunk.code[1], Op::JumpIfFalse(1));
        assert_eq!(chunk.code[2], Op::Jump(0));
    }

    #[test]
    fn while_false_emits_back_jump() {
        // while false {}
        let func = main_func(vec![stmt(StmtKind::While {
            cond: bool_expr(false),
            body: Block { stmts: vec![] },
        })]);
        let compiled = compile(&prog(func)).unwrap();
        let chunk = &compiled.chunks[0];
        // 0: False, 1: JumpIfFalse(1), 2: Jump(-3), 3: Nil, 4: Return
        assert_eq!(chunk.code[0], Op::False);
        assert!(matches!(chunk.code[1], Op::JumpIfFalse(_)));
        assert!(matches!(chunk.code[2], Op::Jump(n) if n < 0));
        // verify exit jump lands at position 3 (Nil)
        // offset for JumpIfFalse at pos 1: target = 3, offset = 3 - 1 - 1 = 1
        assert_eq!(chunk.code[1], Op::JumpIfFalse(1));
    }

    #[test]
    fn while_with_break_patches_correctly() {
        // while true { break; }
        let func = main_func(vec![stmt(StmtKind::While {
            cond: bool_expr(true),
            body: Block {
                stmts: vec![stmt(StmtKind::Break)],
            },
        })]);
        let compiled = compile(&prog(func)).unwrap();
        let chunk = &compiled.chunks[0];
        // 0: True, 1: JumpIfFalse(?), 2: Jump(?) [break], 3: Jump(-4) [back], 4: Nil, 5: Return
        assert_eq!(chunk.code[0], Op::True);
        assert!(matches!(chunk.code[1], Op::JumpIfFalse(_)));
        assert!(matches!(chunk.code[2], Op::Jump(_)));
        assert!(matches!(chunk.code[3], Op::Jump(n) if n < 0));
        // break at pos 2 jumps to after_loop = 4, offset = 4 - 2 - 1 = 1
        assert_eq!(chunk.code[2], Op::Jump(1));
        // exit jump at pos 1 also lands at 4, offset = 4 - 1 - 1 = 2
        assert_eq!(chunk.code[1], Op::JumpIfFalse(2));
    }

    #[test]
    fn get_local_roundtrip() {
        let func = Func {
            id: FuncId(0),
            name: dummy_ident("main"),
            locals: vec![Local {
                name: Some(dummy_ident("x")),
                ty: Type::Int,
            }],
            params_len: 0,
            ret: Type::Void,
            body: Block {
                stmts: vec![
                    stmt(StmtKind::Let {
                        local: LocalId(0),
                        init: int_expr(7),
                    }),
                    stmt(StmtKind::Expr(local_expr(0))),
                ],
            },
            span: dummy_span(),
        };
        let compiled = compile(&prog(func)).unwrap();
        let chunk = &compiled.chunks[0];
        // Constant(0), SetLocal(0), GetLocal(0), Pop, Nil, Return
        assert_eq!(chunk.code[0], Op::Constant(0));
        assert_eq!(chunk.code[1], Op::SetLocal(0));
        assert_eq!(chunk.code[2], Op::GetLocal(0));
        assert_eq!(chunk.code[3], Op::Pop);
    }

    #[test]
    fn two_function_call_emits_call_op() {
        // fn helper() {} fn main() { helper(); }
        let helper = Func {
            id: FuncId(0),
            name: dummy_ident("helper"),
            locals: vec![],
            params_len: 0,
            ret: Type::Void,
            body: Block { stmts: vec![] },
            span: dummy_span(),
        };
        let main = Func {
            id: FuncId(1),
            name: dummy_ident("main"),
            locals: vec![],
            params_len: 0,
            ret: Type::Void,
            body: Block {
                stmts: vec![stmt(StmtKind::Expr(Expr {
                    ty: Type::Void,
                    span: dummy_span(),
                    kind: ExprKind::Call {
                        func: FuncId(0),
                        args: vec![],
                    },
                }))],
            },
            span: dummy_span(),
        };
        let program = Program {
            funcs: vec![helper, main],
        };
        let compiled = compile(&program).unwrap();
        assert_eq!(compiled.main_idx, 1);
        let main_chunk = &compiled.chunks[1];
        assert_eq!(main_chunk.code[0], Op::Call(0, 0));
        assert_eq!(main_chunk.code[1], Op::Pop);
    }

    #[test]
    fn no_main_returns_error() {
        let program = Program {
            funcs: vec![simple_func("notmain", vec![], vec![], 0, Type::Void)],
        };
        assert!(matches!(
            compile(&program),
            Err(CompileError::NoMainFunction)
        ));
    }

    #[test]
    fn call_builtin_println_emits_call_builtin_op() {
        use crate::hir::ExprKind as EK;
        let func = main_func(vec![stmt(StmtKind::Expr(Expr {
            ty: Type::Void,
            span: dummy_span(),
            kind: EK::CallBuiltin {
                builtin: Builtin::Println,
                args: vec![Expr {
                    ty: Type::String,
                    span: dummy_span(),
                    kind: EK::String("hi".into()),
                }],
            },
        }))]);
        let compiled = compile(&prog(func)).unwrap();
        let chunk = &compiled.chunks[0];
        // Constant(0) [String "hi"], CallBuiltin(0, 1), Pop, Nil, Return
        assert_eq!(chunk.code[0], Op::Constant(0));
        assert_eq!(chunk.code[1], Op::CallBuiltin(0, 1));
        assert_eq!(chunk.code[2], Op::Pop);
    }
}
