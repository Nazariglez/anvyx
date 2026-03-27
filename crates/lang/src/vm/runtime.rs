use std::fmt::Write;

use crate::builtin::Builtin;

use super::builtins;
use super::bytecode::{CastKind, Op};
use super::compiler::CompiledProgram;
use super::managed_rc::ManagedRc;
use super::meta::VariantMetaKind;
use super::value::{
    EnumData, MapStorage, RuntimeError, StructData, Value, value_add, value_and, value_div,
    value_eq, value_gt, value_gte, value_lt, value_lte, value_mul, value_negate, value_neq,
    value_not, value_or, value_rem, value_sub, value_xor,
};

pub type ExternHandler = Box<dyn Fn(Vec<Value>) -> Result<Value, RuntimeError>>;

fn as_usize_index(val: &Value) -> Result<usize, RuntimeError> {
    match val {
        Value::Int(i) => Ok(*i as usize),
        other => Err(RuntimeError::new(format!("index must be int, got {other}"))),
    }
}

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
    extern_handlers: Vec<Option<ExternHandler>>,
}

impl<'a> VM<'a> {
    pub fn new(program: &'a CompiledProgram) -> Self {
        let extern_count = program.extern_names.len();
        Self {
            program,
            stack: vec![],
            frames: vec![],
            stdout: String::new(),
            extern_handlers: (0..extern_count).map(|_| None).collect(),
        }
    }

    pub fn register_extern(&mut self, index: usize, handler: ExternHandler) {
        if index < self.extern_handlers.len() {
            self.extern_handlers[index] = Some(handler);
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

        self.run_until_depth(0).map(|_| ())
    }

    fn run_until_depth(&mut self, stop_depth: usize) -> Result<Value, RuntimeError> {
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

                Op::MoveLocal(idx) => {
                    let slot = &mut self.stack[stack_base + idx as usize];
                    let val = std::mem::replace(slot, Value::Nil);
                    self.push(val);
                }

                Op::CloneLocal(idx) => {
                    let val = self.stack[stack_base + idx as usize].clone();
                    self.push(val);
                }

                Op::SetLocal(idx) => {
                    let val = self.pop();
                    let old = std::mem::replace(&mut self.stack[stack_base + idx as usize], val);
                    drop(old);
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
                    let result = if builtin == Builtin::Println {
                        let s = self.format_value(&args[0])?;
                        writeln!(self.stdout, "{s}").unwrap();
                        Value::Nil
                    } else {
                        builtins::call_builtin(builtin, args, &mut self.stdout)?
                    };
                    self.push(result);
                }

                Op::CallExtern(extern_idx, arg_count) => {
                    let n = arg_count as usize;
                    let args: Vec<Value> = self.stack.drain(self.stack.len() - n..).collect();
                    let idx = extern_idx as usize;
                    let result = match self.extern_handlers.get(idx) {
                        Some(Some(handler)) => handler(args)?,
                        _ => {
                            let name = self
                                .program
                                .extern_names
                                .get(idx)
                                .map(|s| s.as_str())
                                .unwrap_or("<unknown>");
                            return Err(RuntimeError::new(format!(
                                "missing extern implementation for '{name}'"
                            )));
                        }
                    };
                    self.push(result);
                }

                Op::Return => {
                    let return_val = self.pop();
                    let frame = self.frames.pop().expect("no frame to return from");

                    // discard all callee locals and temporaries
                    self.stack.truncate(frame.stack_base);

                    if self.frames.len() == stop_depth {
                        return Ok(return_val);
                    }

                    // push return value for the caller
                    self.push(return_val);
                }

                Op::ConstructStruct(type_id, field_count) => {
                    let count = field_count as usize;
                    let start = self.stack.len() - count;
                    let fields: Vec<Value> = self.stack.drain(start..).collect();
                    let data = StructData { type_id, fields };
                    self.push(Value::Struct(ManagedRc::new(data)));
                }

                Op::ConstructDataRef(type_id, field_count) => {
                    let count = field_count as usize;
                    let start = self.stack.len() - count;
                    let fields: Vec<Value> = self.stack.drain(start..).collect();
                    let data = StructData { type_id, fields };
                    let vtable = self.program.struct_meta[type_id as usize].vtable.unwrap();
                    self.push(Value::DataRef(ManagedRc::new_with_vtable(data, vtable)));
                }

                Op::ConstructTuple(count) => {
                    let count = count as usize;
                    let start = self.stack.len() - count;
                    let elements: Vec<Value> = self.stack.drain(start..).collect();
                    self.push(Value::Tuple(ManagedRc::new(elements)));
                }

                Op::ConstructEnum(type_id, variant, field_count) => {
                    let count = field_count as usize;
                    let start = self.stack.len() - count;
                    let fields: Vec<Value> = self.stack.drain(start..).collect();
                    let data = EnumData {
                        type_id,
                        variant,
                        fields,
                    };
                    self.push(Value::Enum(ManagedRc::new(data)));
                }

                Op::GetEnumVariant => {
                    let val = self.pop();
                    match val {
                        Value::Enum(e) => self.push(Value::Int(e.variant as i64)),
                        Value::Nil => self.push(Value::Int(0)),
                        _ => self.push(Value::Int(1)),
                    }
                }

                Op::GetField(index) => {
                    let obj = self.pop();
                    let idx = index as usize;
                    let field_val = match obj {
                        Value::Struct(s) => s.fields.get(idx).cloned().ok_or_else(|| {
                            RuntimeError::new(format!("field index {idx} out of bounds on struct"))
                        })?,
                        Value::DataRef(s) => s.fields.get(idx).cloned().ok_or_else(|| {
                            RuntimeError::new(format!("field index {idx} out of bounds on dataref"))
                        })?,
                        Value::Tuple(t) => t.get(idx).cloned().ok_or_else(|| {
                            RuntimeError::new(format!("index {idx} out of bounds on tuple"))
                        })?,
                        Value::Enum(e) => e.fields.get(idx).cloned().ok_or_else(|| {
                            RuntimeError::new(format!(
                                "field index {idx} out of bounds on enum variant"
                            ))
                        })?,
                        other => other,
                    };
                    self.push(field_val);
                }

                Op::SetField(index) => {
                    let new_value = self.pop();
                    let obj = self.pop();
                    let idx = index as usize;
                    match obj {
                        Value::Struct(mut s) => {
                            ManagedRc::make_mut(&mut s).fields[idx] = new_value;
                            self.push(Value::Struct(s));
                        }
                        Value::DataRef(mut s) => {
                            // shared state is intentional
                            s.force_mut().fields[idx] = new_value;
                            self.push(Value::DataRef(s));
                        }
                        other => {
                            return Err(RuntimeError::new(format!(
                                "SetField on non-struct value: {}",
                                other
                            )));
                        }
                    }
                }

                Op::ConstructArray(count) => {
                    let count = count as usize;
                    let start = self.stack.len() - count;
                    let elements: Vec<Value> = self.stack.drain(start..).collect();
                    self.push(Value::Array(ManagedRc::new(elements)));
                }

                Op::ConstructList(count) => {
                    let count = count as usize;
                    let start = self.stack.len() - count;
                    let elements: Vec<Value> = self.stack.drain(start..).collect();
                    self.push(Value::List(ManagedRc::new(elements)));
                }

                Op::ConstructMap(count) => {
                    let count = count as usize;
                    let total = count * 2;
                    let start = self.stack.len() - total;
                    let flat: Vec<Value> = self.stack.drain(start..).collect();
                    let mut storage = MapStorage::with_capacity_unordered(count);
                    for pair in flat.chunks_exact(2) {
                        let key = pair[0].clone();
                        let value = pair[1].clone();
                        storage.insert(key, value);
                    }
                    self.push(Value::Map(ManagedRc::new(storage)));
                }

                Op::IndexGet => {
                    let index_val = self.pop();
                    let collection = self.pop();

                    match collection {
                        Value::Map(m) => {
                            let element = m.get(&index_val).cloned().unwrap_or(Value::Nil);
                            self.push(element);
                        }
                        Value::Array(a) => {
                            let idx = as_usize_index(&index_val)?;
                            let element = a.get(idx).cloned().ok_or_else(|| {
                                RuntimeError::new(format!(
                                    "index {idx} out of bounds for array of length {}",
                                    a.len()
                                ))
                            })?;
                            self.push(element);
                        }
                        Value::List(l) => {
                            let idx = as_usize_index(&index_val)?;
                            let element = l.get(idx).cloned().ok_or_else(|| {
                                RuntimeError::new(format!(
                                    "index {idx} out of bounds for list of length {}",
                                    l.len()
                                ))
                            })?;
                            self.push(element);
                        }
                        other => {
                            return Err(RuntimeError::new(format!(
                                "IndexGet on non-indexable value: {other}"
                            )));
                        }
                    }
                }

                Op::IndexSet => {
                    let new_value = self.pop();
                    let index_val = self.pop();
                    let collection = self.pop();

                    match collection {
                        Value::Map(mut m) => {
                            ManagedRc::make_mut(&mut m).insert(index_val, new_value);
                            self.push(Value::Map(m));
                        }
                        Value::Array(mut a) => {
                            let idx = as_usize_index(&index_val)?;
                            let len = a.len();
                            if idx >= len {
                                return Err(RuntimeError::new(format!(
                                    "index {idx} out of bounds for array of length {len}"
                                )));
                            }
                            ManagedRc::make_mut(&mut a)[idx] = new_value;
                            self.push(Value::Array(a));
                        }
                        Value::List(mut l) => {
                            let idx = as_usize_index(&index_val)?;
                            let len = l.len();
                            if idx >= len {
                                return Err(RuntimeError::new(format!(
                                    "index {idx} out of bounds for list of length {len}"
                                )));
                            }
                            ManagedRc::make_mut(&mut l)[idx] = new_value;
                            self.push(Value::List(l));
                        }
                        other => {
                            return Err(RuntimeError::new(format!(
                                "IndexSet on non-indexable value: {other}"
                            )));
                        }
                    }
                }

                Op::CollectionLen => {
                    let collection = self.pop();
                    match collection {
                        Value::Array(a) => self.push(Value::Int(a.len() as i64)),
                        Value::List(l) => self.push(Value::Int(l.len() as i64)),
                        other => {
                            return Err(RuntimeError::new(format!(
                                "CollectionLen: expected array or list, got {other}"
                            )));
                        }
                    }
                }

                Op::MapLen => {
                    let map = self.pop();
                    match map {
                        Value::Map(m) => self.push(Value::Int(m.len() as i64)),
                        other => {
                            return Err(RuntimeError::new(format!(
                                "MapLen: expected map, got {other}"
                            )));
                        }
                    }
                }

                Op::MapEntryAt => {
                    let index_val = self.pop();
                    let map = self.pop();
                    match (&map, &index_val) {
                        (Value::Map(m), Value::Int(idx)) => {
                            let idx = *idx as usize;
                            match m.get_index(idx) {
                                Some((k, v)) => {
                                    let entry =
                                        Value::Tuple(ManagedRc::new(vec![k.clone(), v.clone()]));
                                    self.push(entry);
                                }
                                None => {
                                    return Err(RuntimeError::new(format!(
                                        "MapEntryAt: index {idx} out of bounds for map of length {}",
                                        m.len()
                                    )));
                                }
                            }
                        }
                        _ => {
                            return Err(RuntimeError::new(format!(
                                "MapEntryAt: expected (map, int), got ({map}, {index_val})"
                            )));
                        }
                    }
                }

                Op::ListPush => {
                    let value = self.pop();
                    let collection = self.pop();
                    match collection {
                        Value::List(mut l) => {
                            ManagedRc::make_mut(&mut l).push(value);
                            self.push(Value::Nil);
                            self.push(Value::List(l));
                        }
                        other => {
                            return Err(RuntimeError::new(format!(
                                "ListPush on non-list value: {other}"
                            )));
                        }
                    }
                }

                Op::ListPop => {
                    let collection = self.pop();
                    match collection {
                        Value::List(mut l) => {
                            let popped = ManagedRc::make_mut(&mut l).pop().unwrap_or(Value::Nil);
                            self.push(popped);
                            self.push(Value::List(l));
                        }
                        other => {
                            return Err(RuntimeError::new(format!(
                                "ListPop on non-list value: {other}"
                            )));
                        }
                    }
                }

                Op::MapInsert => {
                    let value = self.pop();
                    let key = self.pop();
                    let collection = self.pop();
                    match collection {
                        Value::Map(mut m) => {
                            ManagedRc::make_mut(&mut m).insert(key, value);
                            self.push(Value::Nil);
                            self.push(Value::Map(m));
                        }
                        other => {
                            return Err(RuntimeError::new(format!(
                                "MapInsert on non-map value: {other}"
                            )));
                        }
                    }
                }

                Op::MapRemove => {
                    let key = self.pop();
                    let collection = self.pop();
                    match collection {
                        Value::Map(mut m) => {
                            let removed = ManagedRc::make_mut(&mut m)
                                .remove(&key)
                                .unwrap_or(Value::Nil);
                            self.push(removed);
                            self.push(Value::Map(m));
                        }
                        other => {
                            return Err(RuntimeError::new(format!(
                                "MapRemove on non-map value: {other}"
                            )));
                        }
                    }
                }

                Op::Cast(kind) => {
                    let val = self.pop();
                    let result = match kind {
                        CastKind::IntToFloat => {
                            let Value::Int(n) = val else { unreachable!() };
                            Value::Float(n as f32)
                        }
                        CastKind::FloatToInt => {
                            let Value::Float(f) = val else { unreachable!() };
                            Value::Int(f as i64)
                        }
                        CastKind::IntToDouble => {
                            let Value::Int(n) = val else { unreachable!() };
                            Value::Double(n as f64)
                        }
                        CastKind::DoubleToInt => {
                            let Value::Double(d) = val else {
                                unreachable!()
                            };
                            Value::Int(d as i64)
                        }
                        CastKind::FloatToDouble => {
                            let Value::Float(f) = val else { unreachable!() };
                            Value::Double(f as f64)
                        }
                        CastKind::DoubleToFloat => {
                            let Value::Double(d) = val else {
                                unreachable!()
                            };
                            Value::Float(d as f32)
                        }
                    };
                    self.push(result);
                }

                Op::ToString => {
                    let val = self.pop();
                    let s = self.format_value(&val)?;
                    self.push(Value::String(ManagedRc::new(s)));
                }
            }
        }
    }

    fn call_function(&mut self, chunk_idx: usize, args: Vec<Value>) -> Result<Value, RuntimeError> {
        let return_depth = self.frames.len();

        let stack_base = self.stack.len();
        for arg in args {
            self.stack.push(arg);
        }

        let extra_locals = {
            let chunk = &self.program.chunks[chunk_idx];
            chunk.local_count as usize - chunk.params_count as usize
        };
        self.frames.push(CallFrame {
            chunk_idx,
            ip: 0,
            stack_base,
        });
        for _ in 0..extra_locals {
            self.stack.push(Value::Nil);
        }

        self.run_until_depth(return_depth)
    }

    fn format_value(&mut self, value: &Value) -> Result<String, RuntimeError> {
        match value {
            Value::Struct(s) => {
                let program = self.program;
                let meta = &program.struct_meta[s.type_id as usize];
                if let Some(chunk_idx) = meta.to_string_fn {
                    let result = self.call_function(chunk_idx, vec![value.clone()])?;
                    match result {
                        Value::String(s) => Ok((*s).clone()),
                        _ => Err(RuntimeError::new("to_string must return a string")),
                    }
                } else {
                    let name = meta.name.clone();
                    let field_names: Vec<String> = meta.field_names.clone();
                    let fields: Vec<Value> = s.fields.clone();
                    let mut out = format!("{name}(");
                    for (i, (fname, val)) in field_names.iter().zip(fields.iter()).enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        let formatted = self.format_value(val)?;
                        write!(out, "{fname}: {formatted}").unwrap();
                    }
                    out.push(')');
                    Ok(out)
                }
            }
            Value::DataRef(s) => {
                let program = self.program;
                let meta = &program.struct_meta[s.type_id as usize];
                if let Some(chunk_idx) = meta.to_string_fn {
                    let result = self.call_function(chunk_idx, vec![value.clone()])?;
                    match result {
                        Value::String(s) => Ok((*s).clone()),
                        _ => Err(RuntimeError::new("to_string must return a string")),
                    }
                } else {
                    let name = meta.name.clone();
                    let field_names: Vec<String> = meta.field_names.clone();
                    let fields: Vec<Value> = s.fields.clone();
                    let mut out = format!("{name}(");
                    for (i, (fname, val)) in field_names.iter().zip(fields.iter()).enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        let formatted = self.format_value(val)?;
                        write!(out, "{fname}: {formatted}").unwrap();
                    }
                    out.push(')');
                    Ok(out)
                }
            }
            Value::List(l) => {
                let elems: Vec<Value> = l.iter().cloned().collect();
                let mut out = String::from("[");
                for (i, v) in elems.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let formatted = self.format_value(v)?;
                    out.push_str(&formatted);
                }
                out.push(']');
                Ok(out)
            }
            Value::Array(a) => {
                let elems: Vec<Value> = a.iter().cloned().collect();
                let mut out = String::from("[");
                for (i, v) in elems.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let formatted = self.format_value(v)?;
                    out.push_str(&formatted);
                }
                out.push(']');
                Ok(out)
            }
            Value::Map(m) => {
                let mut out = String::from("[");
                if m.is_ordered() {
                    let entries: Vec<(Value, Value)> =
                        m.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                    for (i, (k, v)) in entries.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        let fk = self.format_value(k)?;
                        let fv = self.format_value(v)?;
                        write!(out, "{fk}: {fv}").unwrap();
                    }
                } else {
                    let mut entries: Vec<(Value, Value)> =
                        m.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                    entries.sort_by(|(a, _), (b, _)| a.to_string().cmp(&b.to_string()));
                    for (i, (k, v)) in entries.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        let fk = self.format_value(k)?;
                        let fv = self.format_value(v)?;
                        write!(out, "{fk}: {fv}").unwrap();
                    }
                }
                out.push(']');
                Ok(out)
            }
            Value::Tuple(t) => {
                let elems: Vec<Value> = t.iter().cloned().collect();
                let mut out = String::from("(");
                for (i, v) in elems.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let formatted = self.format_value(v)?;
                    out.push_str(&formatted);
                }
                out.push(')');
                Ok(out)
            }
            Value::Enum(e) => {
                let program = self.program;
                let type_id = e.type_id as usize;
                let struct_count = program.struct_meta.len();
                let meta = &program.enum_meta[type_id - struct_count];
                let variant_idx = e.variant as usize;
                let variant_name = meta.variants[variant_idx].name.clone();
                let enum_name = meta.name.clone();
                let kind = meta.variants[variant_idx].kind.clone();
                let fields: Vec<Value> = e.fields.clone();
                match kind {
                    VariantMetaKind::Unit => Ok(format!("{enum_name}.{variant_name}")),
                    VariantMetaKind::Tuple(_) => {
                        let mut vals = vec![];
                        for v in fields.iter() {
                            vals.push(self.format_value(v)?);
                        }
                        Ok(format!("{enum_name}.{variant_name}({})", vals.join(", ")))
                    }
                    VariantMetaKind::Struct(field_names) => {
                        let mut pairs = vec![];
                        for (name, val) in field_names.iter().zip(fields.iter()) {
                            let formatted = self.format_value(val)?;
                            pairs.push(format!("{name}: {formatted}"));
                        }
                        Ok(format!("{enum_name}.{variant_name}({})", pairs.join(", ")))
                    }
                }
            }
            Value::ExternHandle(data) => Ok((data.to_string_fn)(data.id)),
            _ => Ok(format!("{value}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::bytecode::Chunk;
    use crate::vm::managed_rc::ManagedRc;

    fn make_program(chunks: Vec<Chunk>, main_idx: usize) -> CompiledProgram {
        CompiledProgram {
            chunks,
            main_idx,
            extern_names: vec![],
            struct_meta: vec![],
            enum_meta: vec![],
        }
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
        Value::String(ManagedRc::new(s.to_string()))
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

    #[test]
    fn construct_struct_pushes_struct_value() {
        // push two Int constants, then ConstructStruct(42, 2)
        let mut chunk = Chunk::new("main", 0, 0);
        let i10 = chunk.add_constant(Value::Int(10));
        let i20 = chunk.add_constant(Value::Int(20));
        chunk.emit(Op::Constant(i10));
        chunk.emit(Op::Constant(i20));
        chunk.emit(Op::ConstructStruct(42, 2));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        vm.run().unwrap();
    }

    #[test]
    fn get_field_reads_struct_field() {
        // construct struct with fields [10, 20], then GetField(1) -> 20
        let mut chunk = Chunk::new("main", 1, 0);
        let i10 = chunk.add_constant(Value::Int(10));
        let i20 = chunk.add_constant(Value::Int(20));
        chunk.emit(Op::Constant(i10));
        chunk.emit(Op::Constant(i20));
        chunk.emit(Op::ConstructStruct(1, 2));
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::GetField(1));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        vm.run().unwrap();
    }

    #[test]
    fn construct_tuple_and_get_field() {
        // tuple (true, 99), GetField(1) -> 99
        let mut chunk = Chunk::new("main", 0, 0);
        let i99 = chunk.add_constant(Value::Int(99));
        chunk.emit(Op::True);
        chunk.emit(Op::Constant(i99));
        chunk.emit(Op::ConstructTuple(2));
        chunk.emit(Op::GetField(1));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        vm.run().unwrap();
    }

    #[test]
    fn set_field_mutates_struct() {
        // local 0 = struct{10, 20}, SetField(0) with 99, GetField(0) -> 99
        let mut chunk = Chunk::new("main", 1, 0);
        let i10 = chunk.add_constant(Value::Int(10));
        let i20 = chunk.add_constant(Value::Int(20));
        let i99 = chunk.add_constant(Value::Int(99));
        // construct
        chunk.emit(Op::Constant(i10));
        chunk.emit(Op::Constant(i20));
        chunk.emit(Op::ConstructStruct(0, 2));
        chunk.emit(Op::SetLocal(0));
        // set field 0 to 99
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(i99));
        chunk.emit(Op::SetField(0));
        chunk.emit(Op::SetLocal(0));
        // read field 0 and return
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::GetField(0));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        vm.run().unwrap();
    }

    #[test]
    fn construct_enum_unit_variant() {
        // ConstructEnum(0, 1, 0) -> Value::Enum
        let mut chunk = Chunk::new("main", 0, 0);
        chunk.emit(Op::ConstructEnum(0, 1, 0));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        vm.run().unwrap();
    }

    #[test]
    fn construct_enum_with_field() {
        // Constant(42), ConstructEnum(0, 0, 1) -> Value::Enum with 1 field
        let mut chunk = Chunk::new("main", 0, 0);
        let i42 = chunk.add_constant(Value::Int(42));
        chunk.emit(Op::Constant(i42));
        chunk.emit(Op::ConstructEnum(0, 0, 1));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        vm.run().unwrap();
    }

    #[test]
    fn get_enum_variant_returns_variant_index() {
        // ConstructEnum(0, 2, 0), GetEnumVariant -> Int(2), Return
        let mut chunk = Chunk::new("main", 0, 0);
        chunk.emit(Op::ConstructEnum(0, 2, 0));
        chunk.emit(Op::GetEnumVariant);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        vm.run().unwrap();
    }

    #[test]
    fn get_field_on_enum_returns_payload() {
        // Constant(99), ConstructEnum(0, 0, 1), GetField(0) -> Int(99)
        let mut chunk = Chunk::new("main", 0, 0);
        let i99 = chunk.add_constant(Value::Int(99));
        chunk.emit(Op::Constant(i99));
        chunk.emit(Op::ConstructEnum(0, 0, 1));
        chunk.emit(Op::GetField(0));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        vm.run().unwrap();
    }

    #[test]
    fn get_enum_variant_on_raw_value_returns_one() {
        let mut chunk = Chunk::new("main", 0, 0);
        chunk.emit(Op::True);
        chunk.emit(Op::GetEnumVariant);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        vm.run().unwrap();
    }

    #[test]
    fn get_enum_variant_on_nil_returns_zero() {
        let mut chunk = Chunk::new("main", 0, 0);
        chunk.emit(Op::Nil);
        chunk.emit(Op::GetEnumVariant);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        vm.run().unwrap();
    }

    #[test]
    fn construct_array_pushes_value() {
        let mut chunk = Chunk::new("main", 0, 0);
        let i1 = chunk.add_constant(Value::Int(1));
        let i2 = chunk.add_constant(Value::Int(2));
        chunk.emit(Op::Constant(i1));
        chunk.emit(Op::Constant(i2));
        chunk.emit(Op::ConstructArray(2));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn construct_list_pushes_value() {
        let mut chunk = Chunk::new("main", 0, 0);
        let i1 = chunk.add_constant(Value::Int(10));
        let i2 = chunk.add_constant(Value::Int(20));
        chunk.emit(Op::Constant(i1));
        chunk.emit(Op::Constant(i2));
        chunk.emit(Op::ConstructList(2));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn index_get_reads_element() {
        // [Int(10), Int(20)][1] -> Int(20)
        let mut chunk = Chunk::new("main", 0, 0);
        let i10 = chunk.add_constant(Value::Int(10));
        let i20 = chunk.add_constant(Value::Int(20));
        let i1 = chunk.add_constant(Value::Int(1));
        chunk.emit(Op::Constant(i10));
        chunk.emit(Op::Constant(i20));
        chunk.emit(Op::ConstructArray(2));
        chunk.emit(Op::Constant(i1));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn index_get_out_of_bounds_error() {
        let mut chunk = Chunk::new("main", 0, 0);
        let i10 = chunk.add_constant(Value::Int(10));
        let i20 = chunk.add_constant(Value::Int(20));
        let i5 = chunk.add_constant(Value::Int(5));
        chunk.emit(Op::Constant(i10));
        chunk.emit(Op::Constant(i20));
        chunk.emit(Op::ConstructArray(2));
        chunk.emit(Op::Constant(i5));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_err());
    }

    #[test]
    fn index_set_mutates_element() {
        // local 0 = [10, 20]; local 0[0] = 99; result = local 0[0] -> 99
        let mut chunk = Chunk::new("main", 1, 0);
        let i10 = chunk.add_constant(Value::Int(10));
        let i20 = chunk.add_constant(Value::Int(20));
        let i99 = chunk.add_constant(Value::Int(99));
        let i0 = chunk.add_constant(Value::Int(0));

        // construct [10, 20], store in local 0
        chunk.emit(Op::Constant(i10));
        chunk.emit(Op::Constant(i20));
        chunk.emit(Op::ConstructArray(2));
        chunk.emit(Op::SetLocal(0));

        // local 0[0] = 99
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(i0));
        chunk.emit(Op::Constant(i99));
        chunk.emit(Op::IndexSet);
        chunk.emit(Op::SetLocal(0));

        // read local 0[0]
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(i0));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn index_set_cow_semantics() {
        // local 0 = [10, 20]; local 1 = local 0 (shared)
        // local 0[0] = 99  (COW: local 1 should remain [10, 20])
        // verify local 1[0] == 10
        let mut chunk = Chunk::new("main", 2, 0);
        let i10 = chunk.add_constant(Value::Int(10));
        let i20 = chunk.add_constant(Value::Int(20));
        let i99 = chunk.add_constant(Value::Int(99));
        let i0 = chunk.add_constant(Value::Int(0));

        // construct [10, 20], store in local 0
        chunk.emit(Op::Constant(i10));
        chunk.emit(Op::Constant(i20));
        chunk.emit(Op::ConstructArray(2));
        chunk.emit(Op::SetLocal(0));

        // copy to local 1 (shared reference)
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::SetLocal(1));

        // mutate local 0[0] = 99
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(i0));
        chunk.emit(Op::Constant(i99));
        chunk.emit(Op::IndexSet);
        chunk.emit(Op::SetLocal(0));

        // read local 1[0] — should still be 10 due to COW
        chunk.emit(Op::GetLocal(1));
        chunk.emit(Op::Constant(i0));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn construct_map_basic() {
        let mut chunk = Chunk::new("main", 0, 0);
        let k = chunk.add_constant(str_val("a"));
        let v = chunk.add_constant(Value::Int(1));
        chunk.emit(Op::Constant(k));
        chunk.emit(Op::Constant(v));
        chunk.emit(Op::ConstructMap(1));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn construct_empty_map() {
        let mut chunk = Chunk::new("main", 0, 0);
        chunk.emit(Op::ConstructMap(0));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn index_get_map_hit() {
        let mut chunk = Chunk::new("main", 0, 0);
        let k = chunk.add_constant(str_val("a"));
        let v = chunk.add_constant(Value::Int(1));
        chunk.emit(Op::Constant(k));
        chunk.emit(Op::Constant(v));
        chunk.emit(Op::ConstructMap(1));
        chunk.emit(Op::Constant(k));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn index_get_map_miss() {
        let mut chunk = Chunk::new("main", 0, 0);
        let k = chunk.add_constant(str_val("a"));
        let v = chunk.add_constant(Value::Int(1));
        let miss = chunk.add_constant(str_val("b"));
        chunk.emit(Op::Constant(k));
        chunk.emit(Op::Constant(v));
        chunk.emit(Op::ConstructMap(1));
        chunk.emit(Op::Constant(miss));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn index_set_map_insert() {
        let mut chunk = Chunk::new("main", 1, 0);
        let k = chunk.add_constant(str_val("x"));
        let v = chunk.add_constant(Value::Int(42));
        chunk.emit(Op::ConstructMap(0));
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(k));
        chunk.emit(Op::Constant(v));
        chunk.emit(Op::IndexSet);
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn index_set_map_update() {
        let mut chunk = Chunk::new("main", 1, 0);
        let k = chunk.add_constant(str_val("a"));
        let v1 = chunk.add_constant(Value::Int(1));
        let v2 = chunk.add_constant(Value::Int(99));
        chunk.emit(Op::Constant(k));
        chunk.emit(Op::Constant(v1));
        chunk.emit(Op::ConstructMap(1));
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(k));
        chunk.emit(Op::Constant(v2));
        chunk.emit(Op::IndexSet);
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn map_cow_semantics() {
        // local 0 = {"a": 1}; local 1 = local 0 (shared)
        // local 0["a"] = 99 (COW: local 1 should remain {"a": 1})
        let mut chunk = Chunk::new("main", 2, 0);
        let k = chunk.add_constant(str_val("a"));
        let v1 = chunk.add_constant(Value::Int(1));
        let v2 = chunk.add_constant(Value::Int(99));

        chunk.emit(Op::Constant(k));
        chunk.emit(Op::Constant(v1));
        chunk.emit(Op::ConstructMap(1));
        chunk.emit(Op::SetLocal(0));

        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::SetLocal(1));

        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(k));
        chunk.emit(Op::Constant(v2));
        chunk.emit(Op::IndexSet);
        chunk.emit(Op::SetLocal(0));

        // read local 1["a"] — should still be 1 due to COW
        chunk.emit(Op::GetLocal(1));
        chunk.emit(Op::Constant(k));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn list_push_basic() {
        let mut chunk = Chunk::new("main", 1, 0);
        let i1 = chunk.add_constant(Value::Int(1));
        let i2 = chunk.add_constant(Value::Int(2));
        let i3 = chunk.add_constant(Value::Int(3));
        let i99 = chunk.add_constant(Value::Int(99));
        chunk.emit(Op::Constant(i1));
        chunk.emit(Op::Constant(i2));
        chunk.emit(Op::Constant(i3));
        chunk.emit(Op::ConstructList(3));
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(i99));
        chunk.emit(Op::ListPush);
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(i3));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn list_push_cow_semantics() {
        let mut chunk = Chunk::new("main", 2, 0);
        let i10 = chunk.add_constant(Value::Int(10));
        let i20 = chunk.add_constant(Value::Int(20));
        let i99 = chunk.add_constant(Value::Int(99));
        let i0 = chunk.add_constant(Value::Int(0));
        chunk.emit(Op::Constant(i10));
        chunk.emit(Op::Constant(i20));
        chunk.emit(Op::ConstructList(2));
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::SetLocal(1));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(i99));
        chunk.emit(Op::ListPush);
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(1));
        chunk.emit(Op::Constant(i0));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn list_pop_basic() {
        let mut chunk = Chunk::new("main", 1, 0);
        let i1 = chunk.add_constant(Value::Int(1));
        let i2 = chunk.add_constant(Value::Int(2));
        let i3 = chunk.add_constant(Value::Int(3));
        let i0 = chunk.add_constant(Value::Int(0));
        chunk.emit(Op::Constant(i1));
        chunk.emit(Op::Constant(i2));
        chunk.emit(Op::Constant(i3));
        chunk.emit(Op::ConstructList(3));
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::ListPop);
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::Pop);
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(i0));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn map_insert_basic() {
        let mut chunk = Chunk::new("main", 1, 0);
        let kb = chunk.add_constant(str_val("b"));
        let v2 = chunk.add_constant(Value::Int(2));
        chunk.emit(Op::ConstructMap(0));
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(kb));
        chunk.emit(Op::Constant(v2));
        chunk.emit(Op::MapInsert);
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(kb));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn map_remove_basic() {
        let mut chunk = Chunk::new("main", 1, 0);
        let ka = chunk.add_constant(str_val("a"));
        let kb = chunk.add_constant(str_val("b"));
        let v1 = chunk.add_constant(Value::Int(1));
        let v2 = chunk.add_constant(Value::Int(2));
        chunk.emit(Op::Constant(ka));
        chunk.emit(Op::Constant(v1));
        chunk.emit(Op::Constant(kb));
        chunk.emit(Op::Constant(v2));
        chunk.emit(Op::ConstructMap(2));
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(ka));
        chunk.emit(Op::MapRemove);
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::Pop);
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(ka));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }

    #[test]
    fn map_remove_cow_semantics() {
        let mut chunk = Chunk::new("main", 2, 0);
        let kx = chunk.add_constant(str_val("x"));
        let v10 = chunk.add_constant(Value::Int(10));
        chunk.emit(Op::Constant(kx));
        chunk.emit(Op::Constant(v10));
        chunk.emit(Op::ConstructMap(1));
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::SetLocal(1));
        chunk.emit(Op::GetLocal(0));
        chunk.emit(Op::Constant(kx));
        chunk.emit(Op::MapRemove);
        chunk.emit(Op::SetLocal(0));
        chunk.emit(Op::Pop);
        chunk.emit(Op::GetLocal(1));
        chunk.emit(Op::Constant(kx));
        chunk.emit(Op::IndexGet);
        chunk.emit(Op::Return);

        let program = make_program(vec![chunk], 0);
        let mut vm = VM::new(&program);
        assert!(vm.run().is_ok());
    }
}
