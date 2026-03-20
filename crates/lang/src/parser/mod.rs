mod common;
mod decl;
mod expr;
mod ops;
mod pattern;
mod stmt;
mod types;

#[cfg(test)]
mod tests;

use crate::{
    ast::{self, ExprId, TypeVarId},
    lexer::SpannedToken,
    span::Spanned,
};
use chumsky::{
    Boxed,
    error::Rich,
    extra::{self, SimpleState},
    prelude::*,
};

use decl::{enum_declaration, extern_declaration, function, import_declaration, struct_declaration};
use stmt::statement;

#[derive(Debug, Default)]
pub(super) struct ParserState {
    next_expr_id: ExprId,
    next_type_var_id: TypeVarId,
}

impl ParserState {
    pub(super) fn new_expr_id(&mut self) -> ExprId {
        let id = ExprId(self.next_expr_id.0);
        self.next_expr_id = ExprId(id.0 + 1);
        id
    }

    pub(super) fn new_type_var_id(&mut self) -> TypeVarId {
        let id = TypeVarId(self.next_type_var_id.0);
        self.next_type_var_id = TypeVarId(id.0 + 1);
        id
    }
}

pub(super) type Input<'src> = &'src [SpannedToken];
pub(super) type Extra<'src> = extra::Full<Rich<'src, SpannedToken>, SimpleState<ParserState>, ()>;
pub(super) trait AnvParser<'src, T>:
    chumsky::Parser<'src, Input<'src>, T, Extra<'src>> + Clone + 'src
{
}
impl<'src, T, P> AnvParser<'src, T> for P where
    P: chumsky::Parser<'src, Input<'src>, T, Extra<'src>> + Clone + 'src
{
}

// It seems tht rustc chokes trying to compile the parser types, so we need to box them
// in order to reduce the chusmky generic types :(
// I feel this is fine for now, the parser should still be fast enough for my tiny lang
pub(super) type BoxedParser<'src, T> = Boxed<'src, 'src, Input<'src>, T, Extra<'src>>;

pub fn parse_ast(tokens: &[SpannedToken]) -> Result<ast::Program, Vec<Rich<'_, SpannedToken>>> {
    let mut state = SimpleState(ParserState::default());
    parser().parse_with_state(tokens, &mut state).into_result()
}

pub(crate) fn parse_type_str(s: &str) -> Result<ast::Type, String> {
    let tokens = crate::lexer::tokenize(s)
        .map_err(|_| format!("Failed to tokenize type string: '{s}'"))?;
    let mut state = SimpleState(ParserState::default());
    types::type_ident()
        .then_ignore(end())
        .parse_with_state(&tokens, &mut state)
        .into_result()
        .map_err(|_| format!("Failed to parse type string: '{s}'"))
}

fn parser<'src>() -> BoxedParser<'src, ast::Program> {
    let stmt = statement();
    let func_decl = function(stmt.clone()).map(|func_node| {
        let span = func_node.span;
        Spanned::new(ast::Stmt::Func(func_node), span)
    });
    let struct_decl = struct_declaration(stmt).map(|struct_node| {
        let span = struct_node.span;
        Spanned::new(ast::Stmt::Struct(struct_node), span)
    });
    let enum_decl = enum_declaration().map(|enum_node| {
        let span = enum_node.span;
        Spanned::new(ast::Stmt::Enum(enum_node), span)
    });

    choice((import_declaration(), extern_declaration(), func_decl, struct_decl, enum_decl))
        .repeated()
        .collect::<Vec<_>>()
        .map(|stmts| ast::Program { stmts })
        .then_ignore(end())
        .boxed()
}
