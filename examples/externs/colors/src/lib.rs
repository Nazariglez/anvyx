use anvyx_lang::{export_fn, export_methods, export_type};

#[derive(Clone)]
#[export_type]
pub struct Color {
    #[field]
    pub r: f32,
    #[field]
    pub g: f32,
    #[field]
    pub b: f32,
    #[field]
    pub a: f32,
    label: String,
}

#[export_methods]
impl Color {
    #[init]
    pub fn init(r: f32, g: f32, b: f32, a: f32, label: String) -> Self {
        Self { r, g, b, a, label }
    }

    pub fn blue() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 1.0,
            a: 1.0,
            label: "blue".to_string(),
        }
    }

    pub fn to_grayscale(&self) -> Self {
        let gray = self.r * 0.299 + self.g * 0.587 + self.b * 0.114;
        Self {
            r: gray,
            g: gray,
            b: gray,
            a: self.a,
            label: format!("{} (grayscale)", self.label),
        }
    }

    pub fn darken(&mut self, amount: f32) {
        self.r *= 1.0 - amount;
        self.g *= 1.0 - amount;
        self.b *= 1.0 - amount;
    }

    #[getter]
    pub fn label(&self) -> String {
        self.label.clone()
    }

    #[setter]
    pub fn set_label(&mut self, v: String) {
        self.label = v;
    }
}

#[export_fn]
pub fn mix(a: &Color, b: &Color, t: f32) -> Color {
    Color {
        r: a.r * (1.0 - t) + b.r * t,
        g: a.g * (1.0 - t) + b.g * t,
        b: a.b * (1.0 - t) + b.b * t,
        a: a.a * (1.0 - t) + b.a * t,
        label: format!("mix({}, {})", a.label, b.label),
    }
}

anvyx_lang::provider!(types: [Color], mix);
