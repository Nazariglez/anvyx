use anvyx_lang::{export_fn, export_type};

#[export_type(name = "Point")]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[export_type(name = "Rect")]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

#[export_fn]
pub fn create_point(x: f64, y: f64) -> Point {
    Point { x, y }
}

#[export_fn]
pub fn point_x(p: &Point) -> f64 {
    p.x
}

#[export_fn]
pub fn point_y(p: &Point) -> f64 {
    p.y
}

#[export_fn]
pub fn move_point(p: &mut Point, dx: f64, dy: f64) {
    p.x += dx;
    p.y += dy;
}

#[export_fn]
pub fn destroy_point(p: Point) {
    let _ = p;
}

#[export_fn]
pub fn create_rect(x: f64, y: f64, w: f64, h: f64) -> Rect {
    Rect { x, y, w, h }
}

#[export_fn]
pub fn rect_area(r: &Rect) -> f64 {
    r.w * r.h
}

#[export_fn]
pub fn point_in_rect(p: &Point, r: &Rect) -> bool {
    p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h
}

#[export_fn]
pub fn destroy_rect(r: Rect) {
    let _ = r;
}

anvyx_lang::provider!(
    types: [Point, Rect],
    create_point,
    point_x,
    point_y,
    move_point,
    destroy_point,
    create_rect,
    rect_area,
    point_in_rect,
    destroy_rect
);

#[cfg(test)]
mod tests {
    use super::*;
    use anvyx_lang::{exports_to_json, Value};

    #[test]
    fn anvyx_externs_contains_all() {
        let externs = anvyx_externs();
        assert_eq!(externs.len(), 9);
        assert!(externs.contains_key("create_point"));
        assert!(externs.contains_key("point_x"));
        assert!(externs.contains_key("point_y"));
        assert!(externs.contains_key("move_point"));
        assert!(externs.contains_key("destroy_point"));
        assert!(externs.contains_key("create_rect"));
        assert!(externs.contains_key("rect_area"));
        assert!(externs.contains_key("point_in_rect"));
        assert!(externs.contains_key("destroy_rect"));
    }

    #[test]
    fn create_point_handler() {
        let externs = anvyx_externs();
        let result = externs["create_point"](vec![Value::Float(1.0), Value::Float(2.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle");
        };

        // Verify it's in the store by reading it back
        let x = externs["point_x"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(x, Value::Float(1.0));
    }

    #[test]
    fn point_x_handler() {
        let externs = anvyx_externs();
        let result = externs["create_point"](vec![Value::Float(3.5), Value::Float(7.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle");
        };

        let x = externs["point_x"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(x, Value::Float(3.5));

        let y = externs["point_y"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(y, Value::Float(7.0));
    }

    #[test]
    fn move_point_handler() {
        let externs = anvyx_externs();
        let result = externs["create_point"](vec![Value::Float(10.0), Value::Float(20.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle");
        };

        // Mutate the point
        externs["move_point"](vec![Value::ExternHandle(id), Value::Float(5.0), Value::Float(-3.0)]).unwrap();

        // Verify mutation persisted
        let x = externs["point_x"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(x, Value::Float(15.0));
        let y = externs["point_y"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(y, Value::Float(17.0));
    }

    #[test]
    fn destroy_point_handler() {
        let externs = anvyx_externs();
        let result = externs["create_point"](vec![Value::Float(1.0), Value::Float(2.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle");
        };

        // Destroy removes from store
        externs["destroy_point"](vec![Value::ExternHandle(id)]).unwrap();

        // Verify it's gone
        let result = externs["point_x"](vec![Value::ExternHandle(id)]);
        assert!(result.is_err());
    }

    #[test]
    fn point_in_rect_handler() {
        let externs = anvyx_externs();

        let p = externs["create_point"](vec![Value::Float(5.0), Value::Float(5.0)]).unwrap();
        let Value::ExternHandle(pid) = p else {
            panic!("expected ExternHandle for point");
        };

        let r = externs["create_rect"](vec![
            Value::Float(0.0),
            Value::Float(0.0),
            Value::Float(10.0),
            Value::Float(10.0),
        ]).unwrap();
        let Value::ExternHandle(rid) = r else {
            panic!("expected ExternHandle for rect");
        };

        // Point (5,5) is inside rect (0,0,10,10)
        let result = externs["point_in_rect"](vec![Value::ExternHandle(pid), Value::ExternHandle(rid)]).unwrap();
        assert_eq!(result, Value::Bool(true));

        // Move point outside
        externs["move_point"](vec![Value::ExternHandle(pid), Value::Float(20.0), Value::Float(0.0)]).unwrap();

        // Point (25,5) is outside rect (0,0,10,10)
        let result = externs["point_in_rect"](vec![Value::ExternHandle(pid), Value::ExternHandle(rid)]).unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn metadata_format() {
        let json = exports_to_json(ANVYX_EXPORTS, ANVYX_TYPE_EXPORTS);

        // Types present
        assert!(json.contains("\"Point\""));
        assert!(json.contains("\"Rect\""));

        // All function names present
        assert!(json.contains("\"create_point\""));
        assert!(json.contains("\"point_x\""));
        assert!(json.contains("\"point_y\""));
        assert!(json.contains("\"move_point\""));
        assert!(json.contains("\"destroy_point\""));
        assert!(json.contains("\"create_rect\""));
        assert!(json.contains("\"rect_area\""));
        assert!(json.contains("\"point_in_rect\""));
        assert!(json.contains("\"destroy_rect\""));

        // Verify create_point param/ret types
        assert!(json.contains("\"ret\":\"Point\""));
        assert!(json.contains("\"ret\":\"float\""));
        assert!(json.contains("\"ret\":\"bool\""));
    }
}
