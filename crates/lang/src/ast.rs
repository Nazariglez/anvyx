use crate::span::Spanned;
use internment::Intern;
use std::fmt::Display;

pub type ExprNode = Spanned<Expr>;
pub type StmtNode = Spanned<Stmt>;
pub type FuncNode = Spanned<Func>;
pub type BlockNode = Spanned<Block>;
pub type BindingNode = Spanned<Binding>;
pub type BinaryNode = Spanned<Binary>;
pub type UnaryNode = Spanned<Unary>;
pub type CallNode = Spanned<Call>;
pub type AssignNode = Spanned<Assign>;
pub type ReturnNode = Spanned<Return>;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub stmts: Vec<StmtNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Func(FuncNode),
    Expr(ExprNode),
    Binding(BindingNode),
    Return(ReturnNode),
}

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq, Default)]
pub struct ExprId(pub u64);

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub id: ExprId,
    pub kind: ExprKind,
}

impl Expr {
    pub fn new(kind: ExprKind, id: ExprId) -> Self {
        Self { id, kind }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Ident(Ident),
    Block(BlockNode),
    Lit(Lit),
    Call(CallNode),
    Binary(BinaryNode),
    Unary(UnaryNode),
    Assign(AssignNode),
}

#[derive(Debug, Copy, Clone, PartialEq, Hash, Eq)]
pub struct Ident(pub Intern<String>);

impl Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Infer,
    Int,
    Float,
    Bool,
    String,
    Void,
    Optional(Box<Type>),
    Func { params: Vec<Type>, ret: Box<Type> },
}

impl Type {
    pub fn is_num(&self) -> bool {
        matches!(self, Type::Int | Type::Float)
    }

    pub fn is_bool(&self) -> bool {
        matches!(self, Type::Bool)
    }

    pub fn is_str(&self) -> bool {
        matches!(self, Type::String)
    }

    pub fn is_void(&self) -> bool {
        matches!(self, Type::Void)
    }

    pub fn is_optional(&self) -> bool {
        matches!(self, Type::Optional(_))
    }

    pub fn is_func(&self) -> bool {
        matches!(self, Type::Func { .. })
    }

    pub fn is_unresolved(&self) -> bool {
        matches!(self, Type::Infer)
    }
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::Bool => write!(f, "bool"),
            Type::String => write!(f, "string"),
            Type::Void => write!(f, "void"),
            Type::Optional(ty) => write!(f, "{}?", ty),
            Type::Func { params, ret } => write!(
                f,
                "fn({}) -> {}",
                params
                    .iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                ret
            ),
            Type::Infer => write!(f, "<infer>"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Mutability {
    Mutable,
    Immutable,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub name: Ident,
    pub ty: Option<Type>,
    pub mutability: Mutability,
    pub value: ExprNode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Func {
    pub name: Ident,
    pub visibility: Visibility,
    pub params: Vec<Param>,
    pub ret: Type,
    pub body: BlockNode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<StmtNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lit {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Nil,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    NotEq,
    LessThan,
    GreaterThan,
    LessThanEq,
    GreaterThanEq,
    And,
    Or,
    Xor,
    Coalesce,
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Rem => write!(f, "%"),
            BinaryOp::Eq => write!(f, "=="),
            BinaryOp::NotEq => write!(f, "!="),
            BinaryOp::LessThan => write!(f, "<"),
            BinaryOp::GreaterThan => write!(f, ">"),
            BinaryOp::LessThanEq => write!(f, "<="),
            BinaryOp::GreaterThanEq => write!(f, ">="),
            BinaryOp::And => write!(f, "&&"),
            BinaryOp::Or => write!(f, "||"),
            BinaryOp::Xor => write!(f, "^"),
            BinaryOp::Coalesce => write!(f, "??"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binary {
    pub left: Box<ExprNode>,
    pub op: BinaryOp,
    pub right: Box<ExprNode>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "-"),
            UnaryOp::Not => write!(f, "!"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Unary {
    pub op: UnaryOp,
    pub expr: Box<ExprNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub func: Box<ExprNode>,
    pub args: Vec<ExprNode>,
    pub type_args: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assign {
    pub target: Box<ExprNode>,
    pub op: AssignOp,
    pub value: Box<ExprNode>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AssignOp {
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
}

impl Display for AssignOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssignOp::Assign => write!(f, "="),
            AssignOp::AddAssign => write!(f, "+="),
            AssignOp::SubAssign => write!(f, "-="),
            AssignOp::MulAssign => write!(f, "*="),
            AssignOp::DivAssign => write!(f, "/="),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Return {
    pub value: Option<ExprNode>,
}
