use crate::{
    intrinsic::{self, CompilationContext, IntrinsicError},
    lexer::{Delimiter, Keyword, Op, SpannedToken, Token},
    span::Span,
};

#[derive(Debug, PartialEq)]
pub(crate) enum CondError {
    UnknownPredicate {
        name: String,
        span: Span,
    },
    UnknownPredicateArg {
        predicate: String,
        arg: String,
        span: Span,
        valid: Vec<String>,
    },
    UnexpectedToken {
        span: Span,
        found: String,
    },
    UnclosedParen {
        span: Span,
    },
    EmptyCondition {
        span: Span,
    },
    UnmatchedEnd {
        span: Span,
    },
    UnterminatedIf {
        span: Span,
    },
    ElifAfterElse {
        span: Span,
    },
    DuplicateElse {
        span: Span,
    },
    ElifOutsideIf {
        span: Span,
    },
    ElseOutsideIf {
        span: Span,
    },
}

impl CondError {
    pub(crate) fn span(&self) -> Span {
        match self {
            Self::UnknownPredicate { span, .. }
            | Self::UnknownPredicateArg { span, .. }
            | Self::UnexpectedToken { span, .. }
            | Self::UnclosedParen { span, .. }
            | Self::EmptyCondition { span, .. }
            | Self::UnmatchedEnd { span, .. }
            | Self::UnterminatedIf { span, .. }
            | Self::ElifAfterElse { span, .. }
            | Self::DuplicateElse { span, .. }
            | Self::ElifOutsideIf { span, .. }
            | Self::ElseOutsideIf { span, .. } => *span,
        }
    }
}

impl std::fmt::Display for CondError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownPredicate { name, .. } => {
                write!(f, "unknown conditional predicate '{name}'")
            }
            Self::UnknownPredicateArg {
                predicate,
                arg,
                valid,
                ..
            } => {
                let valid_list = valid.join("' or '");
                write!(f, "unknown {predicate} '{arg}'; expected '{valid_list}'")
            }
            Self::UnexpectedToken { found, .. } => {
                write!(f, "unexpected token in #if condition: '{found}'")
            }
            Self::UnclosedParen { .. } => write!(f, "unclosed '(' in #if condition"),
            Self::EmptyCondition { .. } => write!(f, "expected condition after #if"),
            Self::UnmatchedEnd { .. } => write!(f, "#end without matching #if"),
            Self::UnterminatedIf { .. } => write!(f, "unterminated #if (missing #end)"),
            Self::ElifAfterElse { .. } => write!(f, "#elif after #else"),
            Self::DuplicateElse { .. } => write!(f, "duplicate #else in #if chain"),
            Self::ElifOutsideIf { .. } => write!(f, "#elif without matching #if"),
            Self::ElseOutsideIf { .. } => write!(f, "#else without matching #if"),
        }
    }
}

#[derive(Debug)]
enum Condition {
    Predicate {
        name: String,
        arg: String,
        span: Span,
    },
    Not(Box<Condition>),
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
}

struct CondParser<'a> {
    tokens: &'a [SpannedToken],
    pos: usize,
}

fn parse_condition(tokens: &[SpannedToken], start: usize) -> Result<(Condition, usize), CondError> {
    let mut parser = CondParser { tokens, pos: start };
    let cond = parser.parse()?;
    let consumed = parser.pos - start;
    Ok((cond, consumed))
}

impl<'a> CondParser<'a> {
    fn peek(&self) -> Option<&'a SpannedToken> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> &'a SpannedToken {
        let tok = &self.tokens[self.pos];
        self.pos += 1;
        tok
    }

    fn peek_token(&self) -> Option<&'a Token> {
        self.peek().map(|(t, _)| t)
    }

    fn parse(&mut self) -> Result<Condition, CondError> {
        if !self.at_condition_start() {
            let span = self.current_span();
            return Err(CondError::EmptyCondition { span });
        }
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Condition, CondError> {
        let mut left = self.parse_and()?;
        while matches!(self.peek_token(), Some(Token::Op(Op::Or))) {
            self.advance();
            let right = self.parse_and()?;
            left = Condition::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Condition, CondError> {
        let mut left = self.parse_unary()?;
        while matches!(self.peek_token(), Some(Token::Op(Op::And))) {
            self.advance();
            let right = self.parse_unary()?;
            left = Condition::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Condition, CondError> {
        if matches!(self.peek_token(), Some(Token::Op(Op::Not))) {
            self.advance();
            let inner = self.parse_unary()?;
            Ok(Condition::Not(Box::new(inner)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<Condition, CondError> {
        match self.peek_token() {
            Some(Token::Open(Delimiter::Parent)) => {
                let (_, open_span) = self.advance();
                let open_span = *open_span;
                let inner = self.parse_or()?;
                match self.peek_token() {
                    Some(Token::Close(Delimiter::Parent)) => {
                        self.advance();
                        Ok(inner)
                    }
                    _ => Err(CondError::UnclosedParen { span: open_span }),
                }
            }
            Some(Token::Ident(_)) => {
                let (tok, name_span) = self.advance();
                let name_span = *name_span;
                let Token::Ident(id) = tok else {
                    unreachable!()
                };
                let name = id.0.as_ref().clone();

                match self.peek_token() {
                    Some(Token::Open(Delimiter::Parent)) => {
                        self.advance();
                    }
                    _ => {
                        let span = self.current_span();
                        return Err(CondError::UnexpectedToken {
                            span,
                            found: self.describe_next(),
                        });
                    }
                }

                let arg = match self.peek_token() {
                    Some(Token::Ident(_)) => {
                        let (tok, _) = self.advance();
                        let Token::Ident(id) = tok else {
                            unreachable!()
                        };
                        id.0.as_ref().clone()
                    }
                    _ => {
                        let span = self.current_span();
                        return Err(CondError::UnexpectedToken {
                            span,
                            found: self.describe_next(),
                        });
                    }
                };

                let close_span = match self.peek_token() {
                    Some(Token::Close(Delimiter::Parent)) => {
                        let (_, s) = self.advance();
                        *s
                    }
                    _ => {
                        return Err(CondError::UnclosedParen { span: name_span });
                    }
                };

                let span = Span::new(name_span.start, close_span.end);
                Ok(Condition::Predicate { name, arg, span })
            }
            _ => {
                let span = self.current_span();
                Err(CondError::UnexpectedToken {
                    span,
                    found: self.describe_next(),
                })
            }
        }
    }

    fn at_condition_start(&self) -> bool {
        matches!(
            self.peek_token(),
            Some(Token::Ident(_) | Token::Open(Delimiter::Parent) | Token::Op(Op::Not))
        )
    }

    fn current_span(&self) -> Span {
        self.peek().map_or(
            self.tokens
                .last()
                .map_or(Span::new(0, 0), |(_, s)| Span::new(s.end, s.end)),
            |(_, s)| *s,
        )
    }

    fn describe_next(&self) -> String {
        self.peek()
            .map_or("end of input".to_string(), |(t, _)| t.to_string())
    }
}

fn evaluate_condition(cond: &Condition, ctx: &CompilationContext) -> Result<bool, CondError> {
    match cond {
        Condition::Predicate { name, arg, span } => {
            intrinsic::evaluate_predicate(name, arg, ctx, *span).map_err(|e| match e {
                IntrinsicError::UnknownIntrinsic { name, span } => {
                    CondError::UnknownPredicate { name, span }
                }
                IntrinsicError::UnknownValue {
                    predicate,
                    value,
                    valid,
                    span,
                } => CondError::UnknownPredicateArg {
                    predicate,
                    arg: value,
                    span,
                    valid,
                },
                other => CondError::UnexpectedToken {
                    span: other.span(),
                    found: other.to_string(),
                },
            })
        }
        Condition::Not(inner) => evaluate_condition(inner, ctx).map(|v| !v),
        Condition::And(a, b) => {
            if !evaluate_condition(a, ctx)? {
                return Ok(false);
            }
            evaluate_condition(b, ctx)
        }
        Condition::Or(a, b) => {
            if evaluate_condition(a, ctx)? {
                return Ok(true);
            }
            evaluate_condition(b, ctx)
        }
    }
}

struct FilterFrame {
    any_branch_taken: bool,
    current_branch_live: bool,
    if_span: Span,
    seen_else: bool,
}

impl FilterFrame {
    fn dead(if_span: Span) -> Self {
        Self {
            any_branch_taken: false,
            current_branch_live: false,
            if_span,
            seen_else: false,
        }
    }
}

fn is_live(stack: &[FilterFrame]) -> bool {
    stack.iter().all(|f| f.current_branch_live)
}

fn enclosing_live(stack: &[FilterFrame]) -> bool {
    if stack.len() <= 1 {
        return true;
    }
    stack[..stack.len() - 1]
        .iter()
        .all(|f| f.current_branch_live)
}

pub(crate) fn filter_tokens(
    tokens: &[SpannedToken],
    ctx: &CompilationContext,
) -> Result<Vec<SpannedToken>, Vec<CondError>> {
    let mut stack: Vec<FilterFrame> = vec![];
    let mut output: Vec<SpannedToken> = vec![];
    let mut errors: Vec<CondError> = vec![];
    let mut pos = 0;

    while pos < tokens.len() {
        let (ref tok, ref span) = tokens[pos];

        if matches!(tok, Token::Hash) {
            let next = tokens.get(pos + 1);
            match next.map(|(t, _)| t) {
                Some(Token::Keyword(Keyword::If)) => {
                    let if_span = *span;
                    pos += 2;

                    let enclosing_is_live = is_live(&stack);

                    let (cond, consumed) = match parse_condition(tokens, pos) {
                        Ok(result) => result,
                        Err(e) => {
                            errors.push(e);
                            stack.push(FilterFrame::dead(if_span));
                            continue;
                        }
                    };
                    pos += consumed;

                    let live = enclosing_is_live
                        && match evaluate_condition(&cond, ctx) {
                            Ok(v) => v,
                            Err(e) => {
                                errors.push(e);
                                false
                            }
                        };

                    if live {
                        stack.push(FilterFrame {
                            any_branch_taken: true,
                            current_branch_live: true,
                            if_span,
                            seen_else: false,
                        });
                    } else {
                        stack.push(FilterFrame::dead(if_span));
                    }
                    continue;
                }

                Some(Token::Ident(id)) if id.0.as_ref() == "elif" => {
                    let directive_span = *span;
                    pos += 2;

                    if stack.is_empty() {
                        errors.push(CondError::ElifOutsideIf {
                            span: directive_span,
                        });
                        if let Ok((_, consumed)) = parse_condition(tokens, pos) {
                            pos += consumed;
                        }
                        continue;
                    }

                    if stack.last().unwrap().seen_else {
                        errors.push(CondError::ElifAfterElse {
                            span: directive_span,
                        });
                        if let Ok((_, consumed)) = parse_condition(tokens, pos) {
                            pos += consumed;
                        }
                        continue;
                    }

                    let enclosing = enclosing_live(&stack);
                    let (cond, consumed) = match parse_condition(tokens, pos) {
                        Ok(result) => result,
                        Err(e) => {
                            errors.push(e);
                            stack.last_mut().unwrap().current_branch_live = false;
                            continue;
                        }
                    };
                    pos += consumed;

                    let top = stack.last_mut().unwrap();
                    if top.any_branch_taken || !enclosing {
                        top.current_branch_live = false;
                    } else {
                        let live = match evaluate_condition(&cond, ctx) {
                            Ok(v) => v,
                            Err(e) => {
                                errors.push(e);
                                false
                            }
                        };
                        top.current_branch_live = live;
                        if live {
                            top.any_branch_taken = true;
                        }
                    }
                    continue;
                }

                Some(Token::Keyword(Keyword::Else)) => {
                    let directive_span = *span;
                    pos += 2;

                    if stack.is_empty() {
                        errors.push(CondError::ElseOutsideIf {
                            span: directive_span,
                        });
                        continue;
                    }

                    let enclosing = enclosing_live(&stack);
                    let top = stack.last_mut().unwrap();
                    if top.seen_else {
                        errors.push(CondError::DuplicateElse {
                            span: directive_span,
                        });
                        continue;
                    }

                    top.seen_else = true;
                    if top.any_branch_taken || !enclosing {
                        top.current_branch_live = false;
                    } else {
                        top.current_branch_live = true;
                        top.any_branch_taken = true;
                    }
                    continue;
                }

                Some(Token::Ident(id)) if id.0.as_ref() == "end" => {
                    let directive_span = *span;
                    pos += 2;

                    if stack.is_empty() {
                        errors.push(CondError::UnmatchedEnd {
                            span: directive_span,
                        });
                    } else {
                        stack.pop();
                    }
                    continue;
                }

                _ => {
                    if is_live(&stack) {
                        output.push(tokens[pos].clone());
                    }
                    pos += 1;
                    continue;
                }
            }
        }

        if is_live(&stack) {
            output.push(tokens[pos].clone());
        }
        pos += 1;
    }

    for frame in &stack {
        errors.push(CondError::UnterminatedIf {
            span: frame.if_span,
        });
    }

    if errors.is_empty() {
        Ok(output)
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use internment::Intern;

    use super::*;
    use crate::{Profile, ast};

    fn ident_tok(name: &str, pos: usize) -> SpannedToken {
        (
            Token::Ident(ast::Ident(Intern::new(name.to_string()))),
            Span::new(pos, pos + name.len()),
        )
    }

    fn hash_tok(pos: usize) -> SpannedToken {
        (Token::Hash, Span::new(pos, pos + 1))
    }

    fn if_tok(pos: usize) -> SpannedToken {
        (Token::Keyword(Keyword::If), Span::new(pos, pos + 2))
    }

    fn else_tok(pos: usize) -> SpannedToken {
        (Token::Keyword(Keyword::Else), Span::new(pos, pos + 4))
    }

    fn open_paren(pos: usize) -> SpannedToken {
        (Token::Open(Delimiter::Parent), Span::new(pos, pos + 1))
    }

    fn close_paren(pos: usize) -> SpannedToken {
        (Token::Close(Delimiter::Parent), Span::new(pos, pos + 1))
    }

    fn not_tok(pos: usize) -> SpannedToken {
        (Token::Op(Op::Not), Span::new(pos, pos + 1))
    }

    fn and_tok(pos: usize) -> SpannedToken {
        (Token::Op(Op::And), Span::new(pos, pos + 2))
    }

    fn or_tok(pos: usize) -> SpannedToken {
        (Token::Op(Op::Or), Span::new(pos, pos + 2))
    }

    fn semi_tok(pos: usize) -> SpannedToken {
        (Token::Semicolon, Span::new(pos, pos + 1))
    }

    // build predicate tokens, name ( arg )
    fn pred_toks(name: &str, arg: &str, start: usize) -> Vec<SpannedToken> {
        vec![
            ident_tok(name, start),
            open_paren(start + name.len()),
            ident_tok(arg, start + name.len() + 1),
            close_paren(start + name.len() + 1 + arg.len()),
        ]
    }

    fn debug_ctx() -> CompilationContext {
        CompilationContext {
            profile: Profile::Debug,
            os: intrinsic::TargetOs::MacOs,
            arch: intrinsic::TargetArch::Aarch64,
            features: vec!["ecs".to_string()],
        }
    }

    fn release_ctx() -> CompilationContext {
        CompilationContext {
            profile: Profile::Release,
            os: intrinsic::TargetOs::Windows,
            arch: intrinsic::TargetArch::X86_64,
            features: vec![],
        }
    }

    fn pred(name: &str, arg: &str) -> Condition {
        Condition::Predicate {
            name: name.to_string(),
            arg: arg.to_string(),
            span: Span::new(0, 0),
        }
    }

    // condition parser tests

    #[test]
    fn parse_single_predicate() {
        let tokens = pred_toks("profile", "debug", 0);
        let (cond, consumed) = parse_condition(&tokens, 0).unwrap();
        assert_eq!(consumed, 4);
        assert!(matches!(
            cond,
            Condition::Predicate { ref name, ref arg, .. } if name == "profile" && arg == "debug"
        ));
    }

    #[test]
    fn parse_not() {
        let mut tokens = vec![not_tok(0)];
        tokens.extend(pred_toks("profile", "release", 1));
        let (cond, consumed) = parse_condition(&tokens, 0).unwrap();
        assert_eq!(consumed, 5);
        assert!(matches!(cond, Condition::Not(_)));
        if let Condition::Not(inner) = cond {
            assert!(matches!(
                *inner,
                Condition::Predicate { ref name, ref arg, .. } if name == "profile" && arg == "release"
            ));
        }
    }

    #[test]
    fn parse_and() {
        let mut tokens = pred_toks("os", "macos", 0);
        tokens.push(and_tok(10));
        tokens.extend(pred_toks("profile", "debug", 12));
        let (cond, consumed) = parse_condition(&tokens, 0).unwrap();
        assert_eq!(consumed, 9);
        assert!(matches!(cond, Condition::And(_, _)));
        if let Condition::And(left, right) = cond {
            assert!(
                matches!(*left, Condition::Predicate { ref name, ref arg, .. } if name == "os" && arg == "macos")
            );
            assert!(
                matches!(*right, Condition::Predicate { ref name, ref arg, .. } if name == "profile" && arg == "debug")
            );
        }
    }

    #[test]
    fn parse_or() {
        let mut tokens = pred_toks("os", "macos", 0);
        tokens.push(or_tok(10));
        tokens.extend(pred_toks("os", "ios", 12));
        let (cond, consumed) = parse_condition(&tokens, 0).unwrap();
        assert_eq!(consumed, 9);
        assert!(matches!(cond, Condition::Or(_, _)));
        if let Condition::Or(left, right) = cond {
            assert!(
                matches!(*left, Condition::Predicate { ref name, ref arg, .. } if name == "os" && arg == "macos")
            );
            assert!(
                matches!(*right, Condition::Predicate { ref name, ref arg, .. } if name == "os" && arg == "ios")
            );
        }
    }

    #[test]
    fn parse_complex_grouped() {
        // (os(macos) || os(ios)) && !profile(debug)
        let mut tokens = vec![open_paren(0)];
        tokens.extend(pred_toks("os", "macos", 1));
        tokens.push(or_tok(10));
        tokens.extend(pred_toks("os", "ios", 12));
        tokens.push(close_paren(20));
        tokens.push(and_tok(21));
        tokens.push(not_tok(23));
        tokens.extend(pred_toks("profile", "debug", 24));
        let (cond, _) = parse_condition(&tokens, 0).unwrap();
        assert!(matches!(cond, Condition::And(_, _)));
        if let Condition::And(left, right) = cond {
            assert!(matches!(*left, Condition::Or(_, _)));
            assert!(matches!(*right, Condition::Not(_)));
        }
    }

    #[test]
    fn parse_precedence() {
        // os(macos) || os(ios) && profile(debug)
        // && binds tighter => Or(os macos, And(os ios, profile debug))
        let mut tokens = pred_toks("os", "macos", 0);
        tokens.push(or_tok(10));
        tokens.extend(pred_toks("os", "ios", 12));
        tokens.push(and_tok(20));
        tokens.extend(pred_toks("profile", "debug", 22));
        let (cond, _) = parse_condition(&tokens, 0).unwrap();
        assert!(matches!(cond, Condition::Or(_, _)));
        if let Condition::Or(left, right) = cond {
            assert!(matches!(*left, Condition::Predicate { ref name, .. } if name == "os"));
            assert!(matches!(*right, Condition::And(_, _)));
        }
    }

    #[test]
    fn parse_empty_condition() {
        let tokens: Vec<SpannedToken> = vec![];
        let result = parse_condition(&tokens, 0);
        assert!(matches!(result, Err(CondError::EmptyCondition { .. })));
    }

    #[test]
    fn parse_unclosed_paren() {
        // ( os(macos) no closing paren
        let mut tokens = vec![open_paren(0)];
        tokens.extend(pred_toks("os", "macos", 1));
        let result = parse_condition(&tokens, 0);
        assert!(matches!(result, Err(CondError::UnclosedParen { .. })));
    }

    // condition evaluator

    #[test]
    fn eval_profile_match() {
        let ctx = debug_ctx();
        assert_eq!(
            evaluate_condition(&pred("profile", "debug"), &ctx),
            Ok(true)
        );
    }

    #[test]
    fn eval_profile_mismatch() {
        let ctx = release_ctx();
        assert_eq!(
            evaluate_condition(&pred("profile", "debug"), &ctx),
            Ok(false)
        );
    }

    #[test]
    fn eval_os_match() {
        let ctx = debug_ctx();
        assert_eq!(evaluate_condition(&pred("os", "macos"), &ctx), Ok(true));
    }

    #[test]
    fn eval_os_mismatch() {
        let ctx = release_ctx();
        assert_eq!(evaluate_condition(&pred("os", "macos"), &ctx), Ok(false));
    }

    #[test]
    fn eval_not() {
        let ctx = debug_ctx();
        let cond = Condition::Not(Box::new(pred("profile", "debug")));
        assert_eq!(evaluate_condition(&cond, &ctx), Ok(false));
    }

    #[test]
    fn eval_and_true() {
        let ctx = debug_ctx();
        let cond = Condition::And(
            Box::new(pred("os", "macos")),
            Box::new(pred("profile", "debug")),
        );
        assert_eq!(evaluate_condition(&cond, &ctx), Ok(true));
    }

    #[test]
    fn eval_and_false() {
        let ctx = debug_ctx();
        let cond = Condition::And(
            Box::new(pred("os", "macos")),
            Box::new(pred("profile", "release")),
        );
        assert_eq!(evaluate_condition(&cond, &ctx), Ok(false));
    }

    #[test]
    fn eval_or() {
        let ctx = debug_ctx();
        let cond = Condition::Or(
            Box::new(pred("os", "macos")),
            Box::new(pred("os", "windows")),
        );
        assert_eq!(evaluate_condition(&cond, &ctx), Ok(true));
    }

    #[test]
    fn eval_feature() {
        let ctx = debug_ctx();
        assert_eq!(evaluate_condition(&pred("feature", "ecs"), &ctx), Ok(true));
    }

    #[test]
    fn eval_feature_missing() {
        let ctx = release_ctx();
        assert_eq!(evaluate_condition(&pred("feature", "ecs"), &ctx), Ok(false));
    }

    #[test]
    fn eval_unknown_predicate() {
        let ctx = debug_ctx();
        let result = evaluate_condition(&pred("platform", "macos"), &ctx);
        assert!(matches!(result, Err(CondError::UnknownPredicate { .. })));
    }

    #[test]
    fn eval_unknown_os_value() {
        let ctx = debug_ctx();
        let result = evaluate_condition(&pred("os", "bsd"), &ctx);
        assert!(matches!(result, Err(CondError::UnknownPredicateArg { .. })));
    }

    // token filter tests

    fn build_if_block(
        pred_name: &str,
        pred_arg: &str,
        body: Vec<SpannedToken>,
    ) -> Vec<SpannedToken> {
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks(pred_name, pred_arg, 3));
        tokens.extend(body);
        tokens.push(hash_tok(50));
        tokens.push(ident_tok("end", 51));
        tokens
    }

    #[test]
    fn filter_live_branch() {
        let tokens = build_if_block("profile", "debug", vec![semi_tok(20)]);
        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        assert_eq!(output.len(), 1);
        assert!(matches!(output[0].0, Token::Semicolon));
    }

    #[test]
    fn filter_dead_branch() {
        let tokens = build_if_block("profile", "debug", vec![semi_tok(20)]);
        let output = filter_tokens(&tokens, &release_ctx()).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn filter_elif() {
        // #if profile(release) #elif profile(debug) #end
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks("profile", "release", 3));
        tokens.push(semi_tok(20));
        tokens.push(hash_tok(22));
        tokens.push(ident_tok("elif", 23));
        tokens.extend(pred_toks("profile", "debug", 27));
        tokens.push(semi_tok(40));
        tokens.push(hash_tok(42));
        tokens.push(ident_tok("end", 43));

        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        assert_eq!(output.len(), 1);
        assert!(matches!(output[0].0, Token::Semicolon));
    }

    #[test]
    fn filter_else_fallback() {
        // #if profile(release) #else #end
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks("profile", "release", 3));
        tokens.push(semi_tok(20));
        tokens.push(hash_tok(22));
        tokens.push(else_tok(23));
        tokens.push(semi_tok(28));
        tokens.push(hash_tok(30));
        tokens.push(ident_tok("end", 31));

        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        assert_eq!(output.len(), 1);
        assert!(matches!(output[0].0, Token::Semicolon));
    }

    #[test]
    fn filter_nested_live_live() {
        // #if profile(debug) #if os(macos) #end #end
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks("profile", "debug", 3));
        tokens.push(hash_tok(20));
        tokens.push(if_tok(21));
        tokens.extend(pred_toks("os", "macos", 23));
        tokens.push(semi_tok(35));
        tokens.push(hash_tok(37));
        tokens.push(ident_tok("end", 38));
        tokens.push(hash_tok(42));
        tokens.push(ident_tok("end", 43));

        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        assert_eq!(output.len(), 1);
        assert!(matches!(output[0].0, Token::Semicolon));
    }

    #[test]
    fn filter_nested_live_dead() {
        // #if profile(debug) #if os(windows) #end #end
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks("profile", "debug", 3));
        tokens.push(semi_tok(15));
        tokens.push(hash_tok(17));
        tokens.push(if_tok(18));
        tokens.extend(pred_toks("os", "windows", 20));
        tokens.push(semi_tok(35));
        tokens.push(hash_tok(37));
        tokens.push(ident_tok("end", 38));
        tokens.push(hash_tok(42));
        tokens.push(ident_tok("end", 43));

        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        assert_eq!(output.len(), 1);
        assert!(matches!(output[0].0, Token::Semicolon));
    }

    #[test]
    fn filter_nested_dead() {
        // #if profile(release) #if os(macos) #end #end
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks("profile", "release", 3));
        tokens.push(hash_tok(20));
        tokens.push(if_tok(21));
        tokens.extend(pred_toks("os", "macos", 23));
        tokens.push(semi_tok(35));
        tokens.push(hash_tok(37));
        tokens.push(ident_tok("end", 38));
        tokens.push(hash_tok(42));
        tokens.push(ident_tok("end", 43));

        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn filter_no_directives() {
        let tokens = vec![semi_tok(0), semi_tok(1)];
        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn filter_strips_directives() {
        let tokens = build_if_block("profile", "debug", vec![semi_tok(20)]);
        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        // No # or `if` or `end` tokens in output
        for (tok, _) in &output {
            assert!(!matches!(tok, Token::Hash));
            assert!(!matches!(tok, Token::Keyword(Keyword::If)));
        }
    }

    #[test]
    fn filter_intrinsic_passthrough() {
        // # profile ( debug ) no 'if' keyword after #, so it passes through
        let mut tokens = vec![hash_tok(0)];
        tokens.extend(pred_toks("profile", "debug", 1));

        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        // All 5 tokens pass through: # profile ( debug )
        assert_eq!(output.len(), 5);
        assert!(matches!(output[0].0, Token::Hash));
    }

    #[test]
    fn filter_intrinsic_in_dead_context() {
        // #if profile(release) #profile(debug) #end in dead context
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks("profile", "release", 3));
        tokens.push(hash_tok(20));
        tokens.extend(pred_toks("profile", "debug", 21));
        tokens.push(hash_tok(35));
        tokens.push(ident_tok("end", 36));

        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn filter_multiple_blocks() {
        // #if profile(debug) #end #if profile(release) #end
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks("profile", "debug", 3));
        tokens.push(semi_tok(15));
        tokens.push(hash_tok(17));
        tokens.push(ident_tok("end", 18));
        tokens.push(hash_tok(22));
        tokens.push(if_tok(23));
        tokens.extend(pred_toks("profile", "release", 25));
        tokens.push(semi_tok(40));
        tokens.push(hash_tok(42));
        tokens.push(ident_tok("end", 43));

        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        assert_eq!(output.len(), 1);
        assert!(matches!(output[0].0, Token::Semicolon));
    }

    #[test]
    fn filter_only_one_branch() {
        // #if profile(debug) #elif profile(debug) #end
        // First branch taken, second should not fire
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks("profile", "debug", 3));
        tokens.push(semi_tok(15));
        tokens.push(hash_tok(17));
        tokens.push(ident_tok("elif", 18));
        tokens.extend(pred_toks("profile", "debug", 22));
        tokens.push(semi_tok(35));
        tokens.push(hash_tok(37));
        tokens.push(ident_tok("end", 38));

        let output = filter_tokens(&tokens, &debug_ctx()).unwrap();
        assert_eq!(output.len(), 1);
        assert!(matches!(output[0].0, Token::Semicolon));
    }

    #[test]
    fn err_unmatched_end() {
        let tokens = vec![hash_tok(0), ident_tok("end", 1)];
        let result = filter_tokens(&tokens, &debug_ctx());
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, CondError::UnmatchedEnd { .. }))
        );
    }

    #[test]
    fn err_unterminated() {
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks("profile", "debug", 3));
        tokens.push(semi_tok(15));
        // No #end

        let result = filter_tokens(&tokens, &debug_ctx());
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, CondError::UnterminatedIf { .. }))
        );
    }

    #[test]
    fn err_else_after_else() {
        // #if profile(debug) #else #else #end
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks("profile", "debug", 3));
        tokens.push(hash_tok(15));
        tokens.push(else_tok(16));
        tokens.push(hash_tok(21));
        tokens.push(else_tok(22));
        tokens.push(hash_tok(27));
        tokens.push(ident_tok("end", 28));

        let result = filter_tokens(&tokens, &debug_ctx());
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, CondError::DuplicateElse { .. }))
        );
    }

    #[test]
    fn err_elif_after_else() {
        // #if profile(debug) #else #elif profile(release) #end
        let mut tokens = vec![hash_tok(0), if_tok(1)];
        tokens.extend(pred_toks("profile", "debug", 3));
        tokens.push(hash_tok(15));
        tokens.push(else_tok(16));
        tokens.push(hash_tok(21));
        tokens.push(ident_tok("elif", 22));
        tokens.extend(pred_toks("profile", "release", 26));
        tokens.push(hash_tok(42));
        tokens.push(ident_tok("end", 43));

        let result = filter_tokens(&tokens, &debug_ctx());
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, CondError::ElifAfterElse { .. }))
        );
    }

    #[test]
    fn err_elif_outside_if() {
        let mut tokens = vec![hash_tok(0), ident_tok("elif", 1)];
        tokens.extend(pred_toks("profile", "debug", 5));

        let result = filter_tokens(&tokens, &debug_ctx());
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, CondError::ElifOutsideIf { .. }))
        );
    }
}
