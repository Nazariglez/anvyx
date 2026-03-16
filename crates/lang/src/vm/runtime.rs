use crate::builtin::Builtin;

use super::builtins;
use super::bytecode::Op;
use super::compiler::CompiledProgram;
use super::value::{
    RuntimeError, Value, value_add, value_and, value_div, value_eq, value_gt, value_gte, value_lt,
    value_lte, value_mul, value_negate, value_neq, value_not, value_or, value_rem, value_sub,
    value_xor,
};

struct CallFrame {
    chunk_idx: usize,
    ip: usize,
    stack_base: usize,
}

pub struct VM<'a> {
    program: &'a CompiledProgram,
    stack: Vec<Value>,
    frames: Vec<CallFrame>,
    pub stdout: String,
}

impl<'a> VM<'a> {
    pub fn new(program: &'a CompiledProgram) -> Self {
        Self {
            program,
            stack: vec![],
            frames: vec![],
            stdout: String::new(),
        }
    }

    fn push(&mut self, value: Value) {
        self.stack.push(value);
    }

    fn pop(&mut self) -> Value {
        self.stack.pop().expect("stack underflow")
    }

    pub fn run(&mut self) -> Result<(), RuntimeError> {
        // bootstrap the main frame
        let main_idx = self.program.main_idx;
        let main_local_count = self.program.chunks[main_idx].local_count as usize;

        self.frames.push(CallFrame {
            chunk_idx: main_idx,
            ip: 0,
            stack_base: 0,
        });
        for _ in 0..main_local_count {
            self.stack.push(Value::Nil);
        }

        loop {
            // fetch current instruction and advance IP atomically
            let (chunk_idx, ip, stack_base) = {
                let frame = self.frames.last_mut().expect("no active frame");
                let result = (frame.chunk_idx, frame.ip, frame.stack_base);
                frame.ip += 1;
                result
            };

            let op = self.program.chunks[chunk_idx].code[ip].clone();

            match op {
                Op::Constant(idx) => {
                    let val = self.program.chunks[chunk_idx].constants[idx as usize].clone();
                    self.push(val);
                }

                Op::True => self.push(Value::Bool(true)),
                Op::False => self.push(Value::Bool(false)),
                Op::Nil => self.push(Value::Nil),

                Op::Pop => {
                    self.pop();
                }

                Op::GetLocal(idx) => {
                    let val = self.stack[stack_base + idx as usize].clone();
                    self.push(val);
                }

                Op::SetLocal(idx) => {
                    let val = self.pop();
                    self.stack[stack_base + idx as usize] = val;
                }

                Op::Negate => {
                    let v = self.pop();
                    self.push(value_negate(v)?);
                }

                Op::Not => {
                    let v = self.pop();
                    self.push(value_not(v)?);
                }

                Op::Add => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_add(lhs, rhs)?);
                }

                Op::Sub => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_sub(lhs, rhs)?);
                }

                Op::Mul => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_mul(lhs, rhs)?);
                }

                Op::Div => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_div(lhs, rhs)?);
                }

                Op::Rem => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_rem(lhs, rhs)?);
                }

                Op::Eq => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_eq(&lhs, &rhs));
                }

                Op::NotEq => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_neq(&lhs, &rhs));
                }

                Op::LessThan => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_lt(lhs, rhs)?);
                }

                Op::GreaterThan => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_gt(lhs, rhs)?);
                }

                Op::LessThanEq => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_lte(lhs, rhs)?);
                }

                Op::GreaterThanEq => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_gte(lhs, rhs)?);
                }

                Op::And => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_and(lhs, rhs)?);
                }

                Op::Or => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_or(lhs, rhs)?);
                }

                Op::Xor => {
                    let rhs = self.pop();
                    let lhs = self.pop();
                    self.push(value_xor(lhs, rhs)?);
                }

                Op::Jump(offset) => {
                    let frame = self.frames.last_mut().expect("no active frame");
                    frame.ip = (frame.ip as isize + offset as isize) as usize;
                }

                Op::JumpIfFalse(offset) => {
                    let cond = self.pop();
                    let is_falsy = matches!(cond, Value::Bool(false) | Value::Nil);
                    if is_falsy {
                        let frame = self.frames.last_mut().expect("no active frame");
                        frame.ip = (frame.ip as isize + offset as isize) as usize;
                    }
                }

                Op::Call(callee_chunk_idx, arg_count) => {
                    let callee_idx = callee_chunk_idx as usize;
                    let (local_count, params_count) = {
                        let callee = &self.program.chunks[callee_idx];
                        (callee.local_count as usize, callee.params_count as usize)
                    };

                    let new_stack_base = self.stack.len() - arg_count as usize;
                    self.frames.push(CallFrame {
                        chunk_idx: callee_idx,
                        ip: 0,
                        stack_base: new_stack_base,
                    });

                    let extra_locals = local_count - params_count;
                    for _ in 0..extra_locals {
                        self.stack.push(Value::Nil);
                    }
                }

                Op::CallBuiltin(builtin_idx, arg_count) => {
                    let n = arg_count as usize;
                    let args: Vec<Value> = self.stack.drain(self.stack.len() - n..).collect();
                    let builtin = Builtin::all()[builtin_idx as usize];
                    let result = builtins::call_builtin(builtin, args, &mut self.stdout)?;
                    self.push(result);
                }

                Op::Return => {
                    let return_val = self.pop();
                    let frame = self.frames.pop().expect("no frame to return from");

                    // discard all callee locals and temporaries
                    self.stack.truncate(frame.stack_base);

                    if self.frames.is_empty() {
                        return Ok(());
                    }

                    // push return value for the caller
                    self.push(return_val);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::bytecode::Chunk;
    use std::rc::Rc;

    fn make_program(chunks: Vec<Chunk>, main_idx: usize) -> CompiledProgram {
        CompiledProgram { chunks, main_idx }
    }

    fn simple_chunk(name: &str, ops: Vec<Op>, constants: Vec<Value>) -> Chunk {
        let mut chunk = Chunk::new(name, 0, 0);
        for op in ops {
            chunk.emit(op);
        }
        for val in constants {
            chunk.add_constant(val);
        }
        chunk
    }

    fn str_val(s: &str) -> Value {
        Value::String(Rc::from(s))
    }

    #[test]
    fn empty_main_runs_ok() {
        let chunk = simple_chunk("main", vec![Op::Nil, Op::Return], vec![]);
        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
        assert!(vm.stdout.is_empty());
    }

    #[test]
    fn return_constant() {
        let chunk = simple_chunk(
            "main",
            vec![Op::Constant(0), Op::Return],
            vec![Value::Int(42)],
        );
        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn set_and_get_local() {
        // local_count = 1; store 10, load, return
        let mut chunk = Chunk::new("main", 1, 0);
        let idx = chunk.add_constant(Value::Int(10));
        chunk.emit(Op::Constant(idx));
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn arithmetic_add() {
        let mut chunk = Chunk::new("main", 0, 0);
        let i1 = chunk.add_constant(Value::Int(3));
        let i2 = chunk.add_constant(Value::Int(4));
        chunk.emit(Op::Constant(i1));
        chunk.emit(Op::Constant(i2));
        chunk.emit(Op::Add);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn jump_if_false_skips_when_false() {
        // push False; JumpIfFalse(1); Nil; Return(skipped via jump); Nil; Return
        let mut chunk = Chunk::new("main", 0, 0);
        chunk.emit(Op::False);
        chunk.emit(Op::JumpIfFalse(1)); // skip one op
        chunk.emit(Op::True); // skipped
        chunk.emit(Op::Nil);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn jump_if_false_does_not_skip_when_true() {
        // push True; JumpIfFalse(1); True(not skipped); Return
        let mut chunk = Chunk::new("main", 0, 0);
        chunk.emit(Op::True);
        chunk.emit(Op::JumpIfFalse(1)); // would skip next op if false
        chunk.emit(Op::Nil); // NOT skipped — falls through
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn call_and_return() {
        // callee (chunk 0): returns Int(99)
        // main (chunk 1): calls callee, Pop, Nil, Return
        let mut callee = Chunk::new("callee", 0, 0);
        let ci = callee.add_constant(Value::Int(99));
        callee.emit(Op::Constant(ci));
        callee.emit(Op::Return);

        let mut main = Chunk::new("main", 0, 0);
        main.emit(Op::Call(0, 0)); // call chunk 0 with 0 args
        main.emit(Op::Pop);
        main.emit(Op::Nil);
        main.emit(Op::Return);

        let program = make_program(vec![callee, main], 1);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn call_with_args() {
        // fn add(a, b) -> a+b  (chunk 0: local_count=2, params_count=2)
        // main calls add(3, 4) and discards result
        let mut add_chunk = Chunk::new("add", 2, 2);
        add_chunk.emit(Op::GetLocal(0));
        add_chunk.emit(Op::GetLocal(1));
        add_chunk.emit(Op::Add);
        add_chunk.emit(Op::Return);

        let mut main_chunk = Chunk::new("main", 0, 0);
        let i3 = main_chunk.add_constant(Value::Int(3));
        let i4 = main_chunk.add_constant(Value::Int(4));
        main_chunk.emit(Op::Constant(i3));
        main_chunk.emit(Op::Constant(i4));
        main_chunk.emit(Op::Call(0, 2));
        main_chunk.emit(Op::Pop);
        main_chunk.emit(Op::Nil);
        main_chunk.emit(Op::Return);

        let program = make_program(vec![add_chunk, main_chunk], 1);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn call_builtin_println_writes_stdout() {
        let mut chunk = Chunk::new("main", 0, 0);
        let si = chunk.add_constant(str_val("hello world"));
        chunk.emit(Op::Constant(si));
        chunk.emit(Op::CallBuiltin(0, 1)); // Println is index 0
        chunk.emit(Op::Pop);
        chunk.emit(Op::Nil);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        vm.run().unwrap();
        assert_eq!(vm.stdout, "hello world\n");
    }

    #[test]
    fn runtime_error_div_by_zero() {
        let mut chunk = Chunk::new("main", 0, 0);
        let i1 = chunk.add_constant(Value::Int(1));
        let i0 = chunk.add_constant(Value::Int(0));
        chunk.emit(Op::Constant(i1));
        chunk.emit(Op::Constant(i0));
        chunk.emit(Op::Div);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        let result = vm.run();
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("zero"));
    }
}
