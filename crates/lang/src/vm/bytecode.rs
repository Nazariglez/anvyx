use super::value::Value;
use crate::ast::FormatSpec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CastKind {
    IntToFloat,
    FloatToInt,
    IntToDouble,
    DoubleToInt,
    FloatToDouble,
    DoubleToFloat,
}

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
    MoveLocal(u16),
    CloneLocal(u16),
    PushRef(u16),                   // push StackRef(stack_base + idx) for local at idx
    PushPathRef(u16, u8, [u16; 4]), // push PathRef(stack_base + idx, depth, segments)
    DerefRead(u16),                 // read value through StackRef/PathRef at local[idx], push copy
    DerefWrite(u16),                // pop value, write through StackRef at local[idx]
    SetFieldRef(u16, u16),          // (local_idx, field_index): pop value, set field through ref
    SetIndexRef(u16),               // local_idx: pop value, pop index, set element through ref

    // unary
    Negate,
    Not,
    BitNot,

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
    BitAnd,
    BitOr,
    Shl,
    Shr,

    // control flow
    Jump(i16),
    JumpIfFalse(i16),

    // functions
    Call(u16, u8),
    CallBuiltin(u8, u8),
    CallExtern(u16, u8),
    CreateClosure(u16, u8), // (fn_chunk_idx, capture_count)
    CallClosure(u8),        // (arg_count)
    Return,

    // composite types
    ConstructStruct(u32, u16),    // (type_id, field_count)
    ConstructDataRef(u32, u16),   // (type_id, field_count)
    ConstructTuple(u16),          // element_count
    ConstructEnum(u32, u16, u16), // (type_id, variant_index, field_count)
    GetField(u16),                // field_index
    SetField(u16),                // field_index
    GetEnumVariant,               // pops enum, pushes variant index as Int
    UnwrapOptional, // pops optional value, pushes inner: Enum(Some(v)) -> v, flat -> self, Nil -> Nil

    // arrays and lists
    ConstructArray(u16), // pops N values, pushes Value::Array
    ConstructList(u16),  // pops N values, pushes Value::List
    IndexGet,            // pops index, pops collection (Array|List|Map), pushes element
    IndexSet, // pops value, pops index, pops collection (Array|List|Map), pushes mutated collection
    CollectionLen, // pops Array|List, pushes Int(len)
    Slice(bool), // pops end, start, collection; bool = inclusive; pushes sliced result

    // maps
    ConstructMap(u16), // pops 2*N values (key, value pairs), pushes Value::Map
    MapLen,            // pops Map, pushes Int(len)
    MapEntryAt,        // pops index, pops Map, pushes Tuple(key, value)

    // mutating collection methods
    ListPush,        // pops value, list -> pushes Nil, modified list
    ListPop,         // pops list -> pushes popped or Nil, modified list
    ListSortBy(u16), // pops list, sorts in place using comparator chunk, pushes sorted list
    MapInsert,       // pops value, key, map -> pushes Nil, modified map
    MapRemove,       // pops key, map -> pushes removed or Nil, modified map

    ToString,
    OptionalToString(u16), // pops optional value, pushes ".Some(inner)" or ".None<T>"; u16 = constant index for inner type name
    Format(FormatSpec),
    Cast(CastKind),
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
        assert!(
            u16::try_from(self.constants.len()).is_ok(),
            "constant pool overflow in chunk '{}'",
            self.name
        );
        let idx = self.constants.len() as u16;
        self.constants.push(value);
        idx
    }

    pub fn emit_jump(&mut self, op: Op) -> usize {
        let pos = self.code.len();
        self.code.push(op);
        pos
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
    fn chunk_metadata() {
        let chunk = Chunk::new("main", 3, 2);
        assert_eq!(chunk.name, "main");
        assert_eq!(chunk.local_count, 3);
        assert_eq!(chunk.params_count, 2);
    }
}
