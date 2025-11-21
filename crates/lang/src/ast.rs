use crate::span::Spanned;
use internment::Intern;
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub stmts: Vec<StmtNode>,
}

pub type ExprNode = Spanned<Expr>;
pub type StmtNode = Spanned<Stmt>;
pub type FuncNode = Spanned<Func>;
pub type BlockNode = Spanned<Block>;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Ident(pub Intern<String>);

impl Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    Bool,
    String,
    Void,
    Optional(Box<Type>),
    Func { params: Vec<Type>, ret: Box<Type> },
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
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Func(FuncNode),
    Expr(ExprNode),
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Visibility {
    Public,
    Private,
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

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Ident(Ident),
    Block(BlockNode),
    Lit(Lit),
    Call {
        func: Box<ExprNode>,
        args: Vec<ExprNode>,
        type_args: Vec<Type>,
    },
}
