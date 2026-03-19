use crate::ast::{Ident, Type};
use internment::Intern;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Builtin {
    Println,
    Assert,
    AssertMsg,
    OrderedMap,
}

const ALL: &[Builtin] = &[
    Builtin::Println,
    Builtin::Assert,
    Builtin::AssertMsg,
    Builtin::OrderedMap,
];

impl Builtin {
    pub fn name(&self) -> &'static str {
        match self {
            Builtin::Println => "println",
            Builtin::Assert => "assert",
            Builtin::AssertMsg => "assert_msg",
            Builtin::OrderedMap => "ordered_map",
        }
    }

    pub fn ident(&self) -> Ident {
        Ident(Intern::new(self.name().to_string()))
    }

    pub fn params(&self) -> Vec<Type> {
        match self {
            Builtin::Println => vec![Type::Any],
            Builtin::Assert => vec![Type::Bool],
            Builtin::AssertMsg => vec![Type::Bool, Type::String],
            Builtin::OrderedMap => vec![],
        }
    }

    pub fn ret(&self) -> Type {
        match self {
            Builtin::Println => Type::Void,
            Builtin::Assert => Type::Void,
            Builtin::AssertMsg => Type::Void,
            Builtin::OrderedMap => Type::Map {
                key: Type::Infer.boxed(),
                value: Type::Infer.boxed(),
            },
        }
    }

    pub fn func_type(&self) -> Type {
        Type::Func {
            params: self.params(),
            ret: Box::new(self.ret()),
        }
    }

    pub fn all() -> &'static [Builtin] {
        ALL
    }

    pub fn from_name(name: &str) -> Option<Builtin> {
        ALL.iter().find(|b| b.name() == name).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn println_name() {
        assert_eq!(Builtin::Println.name(), "println");
    }

    #[test]
    fn println_params() {
        assert_eq!(Builtin::Println.params(), vec![Type::Any]);
    }

    #[test]
    fn println_ret() {
        assert_eq!(Builtin::Println.ret(), Type::Void);
    }

    #[test]
    fn println_func_type() {
        assert_eq!(
            Builtin::Println.func_type(),
            Type::Func {
                params: vec![Type::Any],
                ret: Box::new(Type::Void),
            }
        );
    }

    #[test]
    fn all_contains_all_builtins() {
        let all = Builtin::all();
        assert_eq!(all.len(), 4);
        assert_eq!(all[0], Builtin::Println);
        assert_eq!(all[1], Builtin::Assert);
        assert_eq!(all[2], Builtin::AssertMsg);
        assert_eq!(all[3], Builtin::OrderedMap);
    }

    #[test]
    fn ordered_map_name() {
        assert_eq!(Builtin::OrderedMap.name(), "ordered_map");
    }

    #[test]
    fn ordered_map_params() {
        assert_eq!(Builtin::OrderedMap.params(), vec![]);
    }

    #[test]
    fn ordered_map_ret() {
        assert_eq!(
            Builtin::OrderedMap.ret(),
            Type::Map {
                key: Type::Infer.boxed(),
                value: Type::Infer.boxed(),
            }
        );
    }

    #[test]
    fn from_name_ordered_map() {
        assert_eq!(
            Builtin::from_name("ordered_map"),
            Some(Builtin::OrderedMap)
        );
    }

    #[test]
    fn from_name_known() {
        assert_eq!(Builtin::from_name("println"), Some(Builtin::Println));
    }

    #[test]
    fn from_name_unknown() {
        assert_eq!(Builtin::from_name("unknown"), None);
    }

    #[test]
    fn ident_round_trips() {
        assert_eq!(Builtin::Println.ident().to_string(), "println");
    }

    #[test]
    fn assert_name() {
        assert_eq!(Builtin::Assert.name(), "assert");
    }

    #[test]
    fn assert_params() {
        assert_eq!(Builtin::Assert.params(), vec![Type::Bool]);
    }

    #[test]
    fn assert_ret() {
        assert_eq!(Builtin::Assert.ret(), Type::Void);
    }

    #[test]
    fn assert_msg_name() {
        assert_eq!(Builtin::AssertMsg.name(), "assert_msg");
    }

    #[test]
    fn assert_msg_params() {
        assert_eq!(Builtin::AssertMsg.params(), vec![Type::Bool, Type::String]);
    }

    #[test]
    fn assert_msg_ret() {
        assert_eq!(Builtin::AssertMsg.ret(), Type::Void);
    }

    #[test]
    fn from_name_assert() {
        assert_eq!(Builtin::from_name("assert"), Some(Builtin::Assert));
    }

    #[test]
    fn from_name_assert_msg() {
        assert_eq!(Builtin::from_name("assert_msg"), Some(Builtin::AssertMsg));
    }
}
