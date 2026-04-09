use anvyx_lang::ast;

use super::Printer;

impl Printer<'_> {
    fn write_receiver(&mut self, recv: ast::MethodReceiver) {
        match recv {
            ast::MethodReceiver::Value => self.write("self"),
            ast::MethodReceiver::Var => self.write("var self"),
        }
    }

    pub(super) fn format_annotations(&mut self, annotations: &[ast::AnnotationNode]) {
        for ann in annotations {
            self.write_indent();
            self.write("@");
            self.write_fmt(ann.node.name);
            match &ann.node.args {
                ast::AnnotationArgs::None => {}
                ast::AnnotationArgs::Positional(lit) => {
                    self.write("(");
                    self.format_lit(lit);
                    self.write(")");
                }
                ast::AnnotationArgs::Named(pairs) => {
                    self.write("(");
                    for (i, (name, lit)) in pairs.iter().enumerate() {
                        if i > 0 {
                            self.write(", ");
                        }
                        self.write_fmt(name);
                        // bare flags like `@foo(flag)` are stored as Bool(true), skip the `= true`
                        if !matches!(lit, ast::Lit::Bool(true)) {
                            self.write(" = ");
                            self.format_lit(lit);
                        }
                    }
                    self.write(")");
                }
            }
            self.writeln();
        }
    }

    pub(super) fn format_doc_comment(&mut self, doc: Option<&String>) {
        if let Some(doc) = doc {
            for line in doc.split('\n') {
                self.write_indent();
                self.write("///");
                if !line.is_empty() {
                    self.write(" ");
                    self.write(line);
                }
                self.writeln();
            }
        }
    }

    pub(super) fn format_param(&mut self, param: &ast::Param) {
        if matches!(param.mutability, ast::Mutability::Mutable) {
            self.write("var ");
        }
        self.write_fmt(param.name);
        self.write(": ");
        if param.cast_accept {
            self.write("as ");
        }
        self.format_type(&param.ty);
        if let Some(default) = &param.default {
            self.write(" = ");
            self.format_expr(&default.node);
        }
    }

    pub(super) fn format_inline_params(&mut self, params: &[ast::Param]) {
        for (i, param) in params.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.format_param(param);
        }
    }

    pub(super) fn format_param_list(&mut self, params: &[ast::Param]) {
        if params.is_empty() {
            self.write("()");
            return;
        }
        self.format_comma_list("(", ")", params, Self::format_param);
    }

    pub(super) fn format_func(&mut self, func: &ast::Func) {
        self.populate_type_param_names(&func.type_params, &func.const_params);
        self.format_annotations(&func.annotations);
        self.format_doc_comment(func.doc.as_ref());
        self.write_indent();
        self.format_visibility(func.visibility);
        self.write("fn ");
        self.write_fmt(func.name);
        self.format_type_params(&func.type_params, &func.const_params);
        self.format_param_list(&func.params);
        self.format_return_type(&func.ret);
        self.write(" ");
        self.format_block(&func.body);
        self.writeln();
    }

    pub(super) fn format_extern_func(&mut self, ef: &ast::ExternFunc) {
        self.format_annotations(&ef.annotations);
        self.format_doc_comment(ef.doc.as_ref());
        self.write_indent();
        self.write("extern fn ");
        self.write_fmt(ef.name);
        self.format_param_list(&ef.params);
        self.format_return_type(&ef.ret);
        self.write(";");
        self.writeln();
    }

    fn format_extern_type_member(&mut self, member: &ast::ExternTypeMember) {
        match member {
            ast::ExternTypeMember::Field { doc, name, ty, .. } => {
                self.format_doc_comment(doc.as_ref());
                self.write_indent();
                self.write_fmt(name);
                self.write(": ");
                self.format_type(ty);
                self.write(";");
                self.writeln();
            }
            ast::ExternTypeMember::Method {
                doc,
                name,
                receiver,
                params,
                ret,
            } => {
                self.format_doc_comment(doc.as_ref());
                self.write_indent();
                self.write("fn ");
                self.write_fmt(name);
                self.write("(");
                self.write_receiver(*receiver);
                if !params.is_empty() {
                    self.write(", ");
                    self.format_inline_params(params);
                }
                self.write(")");
                self.format_return_type(ret);
                self.write(";");
                self.writeln();
            }
            ast::ExternTypeMember::StaticMethod {
                doc,
                name,
                params,
                ret,
            } => {
                self.format_doc_comment(doc.as_ref());
                self.write_indent();
                self.write("fn ");
                self.write_fmt(name);
                self.write("(");
                self.format_inline_params(params);
                self.write(")");
                self.format_return_type(ret);
                self.write(";");
                self.writeln();
            }
            ast::ExternTypeMember::Operator {
                op,
                other_ty,
                ret,
                self_on_right,
            } => {
                self.write_indent();
                self.write("op ");
                if *self_on_right {
                    self.format_type(other_ty);
                    self.write(" ");
                    self.write_fmt(op);
                    self.write(" Self");
                } else {
                    self.write("Self ");
                    self.write_fmt(op);
                    self.write(" ");
                    self.format_type(other_ty);
                }
                self.write(" -> ");
                self.format_type(ret);
                self.write(";");
                self.writeln();
            }
            ast::ExternTypeMember::UnaryOperator { op, ret } => {
                self.write_indent();
                self.write("op ");
                self.write_fmt(op);
                self.write("Self -> ");
                self.format_type(ret);
                self.write(";");
                self.writeln();
            }
        }
    }

    pub(super) fn format_extern_type(&mut self, et: &ast::ExternType) {
        self.format_annotations(&et.annotations);
        self.format_doc_comment(et.doc.as_ref());
        self.write_indent();
        self.write("extern type ");
        self.write_fmt(et.name);
        if !et.has_init && et.members.is_empty() {
            self.write(";");
            self.writeln();
            return;
        }
        self.write(" {");
        self.writeln();
        self.indent();
        if et.has_init {
            self.write_indent();
            self.write("init;");
            self.writeln();
        }
        for member in &et.members {
            self.format_extern_type_member(member);
        }
        self.dedent();
        self.write_indent();
        self.write("}");
        self.writeln();
    }

    pub(super) fn format_import(&mut self, import: &ast::Import) {
        self.write_indent();
        self.format_visibility(import.visibility);
        self.write("import ");
        for (i, segment) in import.path.iter().enumerate() {
            if i > 0 {
                self.write(".");
            }
            self.write_fmt(segment);
        }
        match &import.kind {
            ast::ImportKind::Module => {
                self.write(";");
            }
            ast::ImportKind::ModuleAs(alias) => {
                self.write(" as ");
                self.write_fmt(alias);
                self.write(";");
            }
            ast::ImportKind::Selective(items) => {
                let fits = self.try_single_line(|p| {
                    p.write(" { ");
                    for (i, item) in items.iter().enumerate() {
                        if i > 0 {
                            p.write(", ");
                        }
                        p.write_fmt(item.name);
                        if let Some(alias) = &item.alias {
                            p.write(" as ");
                            p.write_fmt(alias);
                        }
                    }
                    p.write(" }");
                });
                if fits {
                    self.write(";");
                } else {
                    self.write(" {");
                    self.writeln();
                    self.indent();
                    for item in items {
                        self.write_indent();
                        self.write_fmt(item.name);
                        if let Some(alias) = &item.alias {
                            self.write(" as ");
                            self.write_fmt(alias);
                        }
                        self.write(",");
                        self.writeln();
                    }
                    self.dedent();
                    self.write_indent();
                    self.write("};");
                }
            }
            ast::ImportKind::Wildcard => {
                self.write(" { * };");
            }
        }
        self.writeln();
    }

    pub(super) fn format_const(&mut self, cd: &ast::ConstDecl) {
        self.format_annotations(&cd.annotations);
        self.format_doc_comment(cd.doc.as_ref());
        self.write_indent();
        self.format_visibility(cd.visibility);
        self.write("const ");
        self.write_fmt(cd.name);
        if let Some(ty) = &cd.ty {
            self.write(": ");
            self.format_type(ty);
        }
        self.write(" = ");
        self.format_expr(&cd.value.node);
        self.write(";");
        self.writeln();
    }

    fn format_struct_field(&mut self, field: &ast::StructField) {
        self.format_annotations(&field.annotations);
        self.format_doc_comment(field.doc.as_ref());
        self.write_indent();
        self.write_fmt(field.name);
        self.write(": ");
        self.format_type(&field.ty);
        if let Some(default) = &field.default {
            self.write(" = ");
            self.format_expr(&default.node);
        }
        self.write(",");
        self.writeln();
    }

    pub(super) fn format_method(&mut self, method: &ast::Method) {
        // save and restore this state so the method's type params
        // don't overwrite the struct's
        let saved_type_vars = self.type_var_names.clone();
        let saved_const_params = self.const_param_names.clone();
        self.extend_type_param_names(&method.type_params, &method.const_params);

        self.format_annotations(&method.annotations);
        self.format_doc_comment(method.doc.as_ref());
        self.write_indent();
        self.format_visibility(method.visibility);
        self.write("fn ");
        self.write_fmt(method.name);
        self.format_type_params(&method.type_params, &method.const_params);
        let has_receiver = method.receiver.is_some();
        let params = &method.params;
        let receiver = method.receiver;
        if !has_receiver && params.is_empty() {
            self.write("()");
        } else {
            let single = self.try_single_line(|p| {
                p.write("(");
                if let Some(recv) = receiver {
                    p.write_receiver(recv);
                }
                if has_receiver && !params.is_empty() {
                    p.write(", ");
                }
                p.format_inline_params(params);
                p.write(")");
            });
            if !single {
                self.write("(");
                self.writeln();
                self.indent();
                if let Some(recv) = receiver {
                    self.write_indent();
                    self.write_receiver(recv);
                    self.write(",");
                    self.writeln();
                }
                for param in params {
                    self.write_indent();
                    self.format_param(param);
                    self.write(",");
                    self.writeln();
                }
                self.dedent();
                self.write_indent();
                self.write(")");
            }
        }
        self.format_return_type(&method.ret);
        self.write(" ");
        self.format_block(&method.body);
        self.writeln();

        self.type_var_names = saved_type_vars;
        self.const_param_names = saved_const_params;
    }

    pub(super) fn format_aggregate(&mut self, decl: &ast::StructDecl) {
        self.populate_type_param_names(&decl.type_params, &decl.const_params);
        self.format_annotations(&decl.annotations);
        self.format_doc_comment(decl.doc.as_ref());
        self.write_indent();
        self.format_visibility(decl.visibility);
        self.write(decl.kind.keyword());
        self.write(" ");
        self.write_fmt(decl.name);
        self.format_type_params(&decl.type_params, &decl.const_params);
        self.write(" {");
        self.writeln();
        self.indent();
        for field in &decl.fields {
            self.format_struct_field(field);
        }
        if !decl.fields.is_empty() && !decl.methods.is_empty() {
            self.writeln();
        }
        for method in &decl.methods {
            self.format_method(method);
        }
        self.dedent();
        self.write_indent();
        self.write("}");
        self.writeln();
    }

    fn format_enum_variant(&mut self, variant: &ast::EnumVariant) {
        self.format_annotations(&variant.annotations);
        self.format_doc_comment(variant.doc.as_ref());
        self.write_indent();
        self.write_fmt(variant.name);
        match &variant.kind {
            ast::VariantKind::Unit => {}
            ast::VariantKind::Tuple(types) => {
                self.write("(");
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.format_type(ty);
                }
                self.write(")");
            }
            ast::VariantKind::Struct(fields) => {
                self.write(" { ");
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write_fmt(field.name);
                    self.write(": ");
                    self.format_type(&field.ty);
                }
                self.write(" }");
            }
        }
        self.write(",");
        self.writeln();
    }

    pub(super) fn format_enum(&mut self, decl: &ast::EnumDecl) {
        self.populate_type_param_names(&decl.type_params, &decl.const_params);
        self.format_annotations(&decl.annotations);
        self.format_doc_comment(decl.doc.as_ref());
        self.write_indent();
        self.format_visibility(decl.visibility);
        self.write("enum ");
        self.write_fmt(decl.name);
        self.format_type_params(&decl.type_params, &decl.const_params);
        self.write(" {");
        self.writeln();
        self.indent();
        for variant in &decl.variants {
            self.format_enum_variant(variant);
        }
        self.dedent();
        self.write_indent();
        self.write("}");
        self.writeln();
    }

    fn write_extend_param(&mut self, param: &ast::Param) {
        if *param.name.0 == "self" {
            if matches!(param.mutability, ast::Mutability::Mutable) {
                self.write("var self");
            } else {
                self.write("self");
            }

            return;
        }

        self.format_param(param);
    }

    fn format_extend_method(&mut self, method: &ast::ExtendMethod) {
        self.format_annotations(&method.annotations);
        self.format_doc_comment(method.doc.as_ref());
        self.write_indent();
        self.write("fn ");
        self.write_fmt(method.name);
        if method.params.is_empty() {
            self.write("()");
        } else {
            self.format_comma_list("(", ")", &method.params, |p, param| {
                p.write_extend_param(param);
            });
        }
        self.format_return_type(&method.ret);
        self.write(" ");
        self.format_block(&method.body);
        self.writeln();
    }

    fn format_cast_from(&mut self, cf: &ast::CastFrom) {
        self.write_indent();
        self.write("cast from(");
        self.format_param(&cf.param);
        self.write(")");
        if let Some(ret) = &cf.ret {
            self.write(" -> ");
            self.format_type(ret);
        }
        self.write(" ");
        self.format_block(&cf.body);
        self.writeln();
    }

    fn format_extend_type(&mut self, ty: &ast::Type) {
        if let ast::Type::DataRef { name, type_args } = ty
            && type_args.is_empty()
        {
            self.write("dataref ");
            self.write_fmt(name);
            return;
        }
        self.format_type(ty);
    }

    pub(super) fn format_extend(&mut self, decl: &ast::ExtendDecl) {
        self.populate_type_param_names(&decl.type_params, &decl.const_params);
        self.write_indent();
        self.format_visibility(decl.visibility);

        // put type params after the type name for named targets (`extend Name<T>`).
        // for unnamed targets put them right after `extend` (`extend<T> [T]`)
        match &decl.ty {
            ast::Type::UnresolvedName(_) | ast::Type::DataRef { .. } => {
                self.write("extend ");
                self.format_extend_type(&decl.ty);
                self.format_type_params(&decl.type_params, &decl.const_params);
            }
            _ => {
                self.write("extend");
                self.format_type_params(&decl.type_params, &decl.const_params);
                self.write(" ");
                self.format_extend_type(&decl.ty);
            }
        }

        self.write(" {");
        self.writeln();
        self.indent();
        for method in &decl.methods {
            self.format_extend_method(&method.node);
        }
        for cf in &decl.cast_froms {
            self.format_cast_from(&cf.node);
        }
        self.dedent();
        self.write_indent();
        self.write("}");
        self.writeln();
    }
}
