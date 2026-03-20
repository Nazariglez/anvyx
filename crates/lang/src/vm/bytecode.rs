use super::value::Value;

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    // constants / literals
    Constant(u16),
    True,
    False,
    Nil,

    // stack
    Pop,

    // locals
    GetLocal(u16),
    SetLocal(u16),

    // unary
    Negate,
    Not,

    // binary arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,

    // comparison
    Eq,
    NotEq,
    LessThan,
    GreaterThan,
    LessThanEq,
    GreaterThanEq,

    // logical
    And,
    Or,
    Xor,

    // control flow
    Jump(i16),
    JumpIfFalse(i16),

    // functions
    Call(u16, u8),
    CallBuiltin(u8, u8),
    CallExtern(u16, u8),
    Return,

    // composite types
    ConstructStruct(u32, u16),    // (type_id, field_count)
    ConstructTuple(u16),          // element_count
    ConstructEnum(u32, u16, u16), // (type_id, variant_index, field_count)
    GetField(u16),                // field_index
    SetField(u16),                // field_index
    GetEnumVariant,               // pops enum, pushes variant index as Int

    // arrays and lists
    ConstructArray(u16), // pops N values, pushes Value::Array
    ConstructList(u16),  // pops N values, pushes Value::List
    IndexGet,            // pops index, pops collection (Array|List|Map), pushes element
    IndexSet,      // pops value, pops index, pops collection (Array|List|Map), pushes mutated collection
    CollectionLen, // pops Array|List, pushes Int(len)

    // maps
    ConstructMap(u16), // pops 2*N values (key, value pairs), pushes Value::Map
    MapLen,            // pops Map, pushes Int(len)
    MapEntryAt,        // pops index, pops Map, pushes Tuple(key, value)

    // mutating collection methods
    ListPush,  // pops value, list -> pushes Nil, modified list
    ListPop,   // pops list -> pushes popped or Nil, modified list
    MapInsert, // pops value, key, map -> pushes Nil, modified map
    MapRemove, // pops key, map -> pushes removed or Nil, modified map
}

pub struct Chunk {
    pub code: Vec<Op>,
    pub constants: Vec<Value>,
    pub local_count: u16,
    pub params_count: u8,
    pub name: String,
}

impl Chunk {
    pub fn new(name: impl Into<String>, local_count: u16, params_count: u8) -> Self {
        Self {
            code: vec![],
            constants: vec![],
            local_count,
            params_count,
            name: name.into(),
        }
    }

    pub fn emit(&mut self, op: Op) {
        self.code.push(op);
    }

    pub fn add_constant(&mut self, value: Value) -> u16 {
        let idx = self.constants.len() as u16;
        self.constants.push(value);
        idx
    }

    pub fn emit_jump(&mut self, op: Op) -> usize {
        let pos = self.code.len();
        self.code.push(op);
        pos
    }

    pub fn patch_jump(&mut self, pos: usize) {
        let target = self.code.len();
        let offset = (target as isize - pos as isize - 1) as i16;
        match &mut self.code[pos] {
            Op::Jump(o) | Op::JumpIfFalse(o) => *o = offset,
            _ => panic!("patch_jump called on non-jump op at position {pos}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_val(n: i64) -> Value {
        Value::Int(n)
    }

    #[test]
    fn emit_and_read_back() {
        let mut chunk = Chunk::new("test", 0, 0);
        chunk.emit(Op::True);
        chunk.emit(Op::Pop);
        assert_eq!(chunk.code, vec![Op::True, Op::Pop]);
    }

    #[test]
    fn add_constant_returns_index() {
        let mut chunk = Chunk::new("test", 0, 0);
        let i0 = chunk.add_constant(int_val(1));
        let i1 = chunk.add_constant(int_val(2));
        let i2 = chunk.add_constant(int_val(3));
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(i2, 2);
        assert_eq!(chunk.constants.len(), 3);
    }

    #[test]
    fn constant_roundtrip() {
        let mut chunk = Chunk::new("test", 0, 0);
        let idx = chunk.add_constant(int_val(42));
        chunk.emit(Op::Constant(idx));
        assert_eq!(chunk.code[0], Op::Constant(0));
        assert_eq!(chunk.constants[0], int_val(42));
    }

    #[test]
    fn emit_jump_returns_placeholder_position() {
        let mut chunk = Chunk::new("test", 0, 0);
        chunk.emit(Op::Nil);
        let pos = chunk.emit_jump(Op::JumpIfFalse(0));
        assert_eq!(pos, 1);
        assert_eq!(chunk.code[1], Op::JumpIfFalse(0));
    }

    #[test]
    fn patch_jump_forward() {
        let mut chunk = Chunk::new("test", 0, 0);
        // pos 0: JumpIfFalse(placeholder)
        let pos = chunk.emit_jump(Op::JumpIfFalse(0));
        // pos 1, 2: some code
        chunk.emit(Op::Nil);
        chunk.emit(Op::Pop);
        // pos 3: target — patch jump to here
        chunk.patch_jump(pos);
        // after reading JumpIfFalse at pos 0, ip = 1. new_ip = 1 + offset.
        // offset = 3 - 0 - 1 = 2  =>  new_ip = 1 + 2 = 3
        assert_eq!(chunk.code[0], Op::JumpIfFalse(2));
    }

    #[test]
    fn chunk_metadata() {
        let chunk = Chunk::new("main", 3, 2);
        assert_eq!(chunk.name, "main");
        assert_eq!(chunk.local_count, 3);
        assert_eq!(chunk.params_count, 2);
    }
}
