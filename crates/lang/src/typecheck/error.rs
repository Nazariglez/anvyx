use crate::{
    ast::{Ident, Type},
    span::Span,
};

#[derive(Debug, Clone, PartialEq)]
pub struct TypeErr {
    pub span: Span,
    pub kind: TypeErrKind,
    pub help: Option<String>,
    pub notes: Vec<String>,
    pub secondary: Vec<(Span, String)>,
}

impl TypeErr {
    pub fn new(span: Span, kind: TypeErrKind) -> Self {
        Self {
            span,
            kind,
            help: None,
            notes: vec![],
            secondary: vec![],
        }
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_secondary(mut self, span: Span, msg: impl Into<String>) -> Self {
        self.secondary.push((span, msg.into()));
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeErrKind {
    UnknownVariable {
        name: Ident,
    },
    MismatchedTypes {
        expected: Type,
        found: Type,
    },
    InvalidOperand {
        op: String,
        operand_type: Type,
    },
    UnknownFunction {
        name: Ident,
    },
    UnresolvedInfer,
    NotAFunction {
        expr_type: Type,
    },
    GenericArgNumMismatch {
        expected: usize,
        found: usize,
    },
    NotGenericFunction,
    IfConditionNotBool {
        found: Type,
    },
    IfMissingElse,
    WhileConditionNotBool {
        found: Type,
    },
    BreakOutsideLoop,
    ContinueOutsideLoop,
    TupleIndexOnNonTuple {
        found: Type,
        index: u32,
    },
    TupleIndexOutOfBounds {
        tuple_type: Type,
        index: u32,
        len: usize,
    },
    TuplePatternArityMismatch {
        expected: usize,
        found: usize,
    },
    NonTupleInTuplePattern {
        found: Type,
        pattern_arity: usize,
    },
    TuplePatternLabelMismatch {
        expected: Ident,
        found: Ident,
    },
    NamedPatternOnPositionalTuple,
    DuplicateTupleLabel {
        label: Ident,
    },
    NoSuchFieldOnTuple {
        field: Ident,
        tuple_type: Type,
    },
    FieldAccessOnNonNamedTuple {
        field: Ident,
        found: Type,
    },
    UnknownStruct {
        name: Ident,
    },
    StructMissingField {
        struct_name: Ident,
        field: Ident,
    },
    StructUnknownField {
        struct_name: Ident,
        field: Ident,
    },
    ExternUnknownField {
        type_name: Ident,
        field: Ident,
    },
    ExternUnknownMethod {
        type_name: Ident,
        method: Ident,
    },
    ExternInitNoInit {
        type_name: Ident,
    },
    ExternInitMissingField {
        type_name: Ident,
        field: Ident,
    },
    ExternInitUnknownField {
        type_name: Ident,
        field: Ident,
    },
    ExternInitDuplicateField {
        type_name: Ident,
        field: Ident,
    },
    StructDestructureUnknownField {
        type_name: Ident,
        field: Ident,
    },
    StructDestructureDuplicateField {
        type_name: Ident,
        field: Ident,
    },
    StructDuplicateField {
        struct_name: Ident,
        field: Ident,
    },
    FieldDefaultNotConst {
        struct_name: Ident,
        field: Ident,
    },
    FieldDefaultTypeMismatch {
        struct_name: Ident,
        field: Ident,
        expected: Type,
        found: Type,
    },
    FieldDefaultOnGenericType {
        struct_name: Ident,
        field: Ident,
    },
    RequiredParamAfterOptional {
        func: Ident,
        param: Ident,
    },
    ParamDefaultNotConst {
        func: Ident,
        param: Ident,
    },
    ParamDefaultTypeMismatch {
        func: Ident,
        param: Ident,
        expected: Type,
        found: Type,
    },
    ParamDefaultOnGenericType {
        func: Ident,
        param: Ident,
    },
    TooFewArguments {
        expected: usize,
        found: usize,
    },
    TooManyArguments {
        expected: usize,
        found: usize,
    },
    UnknownMethod {
        struct_name: Ident,
        method: Ident,
    },
    StaticMethodOnValue {
        struct_name: Ident,
        method: Ident,
    },
    InstanceMethodOnType {
        struct_name: Ident,
        method: Ident,
    },
    ReadonlySelfMutation {
        struct_name: Ident,
        field: Ident,
    },
    InvalidToStringSignature {
        struct_name: Ident,
        reason: String,
    },

    UnknownEnum {
        name: Ident,
    },
    UnknownEnumVariant {
        enum_name: Ident,
        variant_name: Ident,
    },
    EnumVariantArityMismatch {
        enum_name: Ident,
        variant_name: Ident,
        expected: usize,
        found: usize,
    },
    EnumVariantNotTuple {
        enum_name: Ident,
        variant_name: Ident,
    },
    EnumVariantNotStruct {
        enum_name: Ident,
        variant_name: Ident,
    },
    EnumVariantNotUnit {
        enum_name: Ident,
        variant_name: Ident,
    },
    EnumVariantMissingField {
        enum_name: Ident,
        variant_name: Ident,
        field: Ident,
    },
    EnumVariantUnknownField {
        enum_name: Ident,
        variant_name: Ident,
        field: Ident,
    },
    EnumVariantDuplicateField {
        enum_name: Ident,
        variant_name: Ident,
        field: Ident,
    },

    ForIterableNotSupported {
        found: Type,
    },
    ForStepNotInt {
        item_ty: Type,
        step_ty: Type,
    },
    ForMapStepNotAllowed,
    ForMapRevNotAllowed,
    ForRangeFromRevNotAllowed,
    ArrayAllNilAmbiguous,
    ArrayFillLengthNotLiteral,
    IndexOnNonArray {
        found: Type,
    },
    IndexNotInt {
        found: Type,
    },
    RangeIndexNotInt {
        found: Type,
    },
    RangeIndexOnMap,
    OptionalChainingOnNonOpt {
        found: Type,
    },

    UnsupportedMatchScrutinee {
        found: Type,
    },
    InvalidLiteralPattern {
        expected: Type,
        found: Type,
    },
    NonExhaustiveMatchNoCatchAll,
    NonExhaustiveMatch {
        missing: Vec<Ident>,
    },
    MatchArmTypeMismatch {
        expected: Type,
        found: Type,
    },
    MatchPatternEnumMismatch {
        expected_enum: Ident,
        pattern_enum: Ident,
    },
    NilPatternOnNonOptional {
        found: Type,
    },
    OptionalPatternOnNonOptional {
        found: Type,
    },
    NestedOptionalPattern,
    NonNumericRangePattern {
        found: Type,
    },
    RangePatternBoundTypeMismatch {
        start: Type,
        end: Type,
    },
    EmptyRangePattern,

    ImmutableAssignment {
        name: Ident,
    },
    VarParamNotLvalue {
        param: Ident,
    },
    VarParamImmutableBinding {
        param: Ident,
        binding: Ident,
    },
    MutatingMethodOnImmutable {
        struct_name: Ident,
        method: Ident,
    },

    MapEmptyLiteralNoContext,
    MapKeyFloat,
    MapKeyNotKeyable {
        found: Type,
    },
    MapOptionalKeyNotAllowed {
        found: Type,
    },
    MapDuplicateKey,

    InvalidCast {
        from: Type,
        to: Type,
    },

    MethodTypeParamShadowsStruct {
        struct_name: Ident,
        method: Ident,
        param: Ident,
    },

    NotEquatable {
        ty: Type,
    },

    UnknownModuleMember {
        module: Ident,
        member: Ident,
    },

    PrivateModuleMember {
        module: Ident,
        member: Ident,
    },

    ReExportCollision {
        name: Ident,
        first_source: String,
        second_source: String,
    },

    AnyTypeNotAllowed,

    AmbiguousOperator {
        op: String,
        left: Type,
        right: Type,
    },

    LetElseMustDiverge,
    LetElseIrrefutable,

    NotConstantExpression,
    CircularConstDependency {
        name: Ident,
    },
    ConstDivisionByZero,
    ConstIntegerOverflow,
    ConstTypeMismatch {
        expected: Type,
        got: Type,
    },
    ConstAssignment {
        name: Ident,
    },
    DuplicateConst {
        name: Ident,
    },
    DuplicateTypeDefinition {
        name: Ident,
    },
    ImportNameConflict {
        name: Ident,
        existing: &'static str,
    },
    ModuleBindingConflict {
        name: Ident,
    },

    AmbiguousExtendMethod {
        ty: Type,
        method: Ident,
        candidates: Vec<Ident>,
    },
    ExtendMethodConflict {
        ty: Type,
        method: Ident,
    },
    DuplicateExtendMethod {
        ty: Type,
        method: Ident,
    },
    ExtendMethodMissingSelf {
        method: Ident,
    },
    ExtendSelfTypeAnnotation {
        method: Ident,
    },
    ExtendUnsupportedType {
        ty: Type,
    },
    ExtendTypeParamCountMismatch {
        ty_name: Ident,
        expected: usize,
        found: usize,
    },
    ExtendTypeParamsOnNonGeneric {
        ty_name: Ident,
    },

    CannotInferLambdaParam {
        name: Ident,
    },
    MutateCapturedVar {
        name: Ident,
    },
    LambdaParamCountMismatch {
        expected: usize,
        found: usize,
    },
    MutableParamRequiresVarTarget {
        name: Ident,
    },
}
