use anvyx_lang::{export_methods, export_type};

#[export_type(name = "Point")]
pub struct Point {
    #[field] pub x: f64,
    #[field] pub y: f64,
}

#[export_methods]
impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    pub fn move_by(&mut self, dx: f64, dy: f64) {
        self.x += dx;
        self.y += dy;
    }
    pub fn distance_to(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

#[export_type(name = "Rect")]
pub struct Rect {
    #[field] pub x: f64,
    #[field] pub y: f64,
    #[field] pub w: f64,
    #[field] pub h: f64,
}

#[export_methods]
impl Rect {
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, w, h }
    }
    pub fn area(&self) -> f64 {
        self.w * self.h
    }
    pub fn contains(&self, p: &Point) -> bool {
        p.x >= self.x && p.x <= self.x + self.w && p.y >= self.y && p.y <= self.y + self.h
    }
}

anvyx_lang::provider!(types: [Point, Rect]);

#[cfg(test)]
mod tests {
    use super::*;
    use anvyx_lang::{Value, exports_to_json};

    #[test]
    fn anvyx_externs_contains_all() {
        let externs = anvyx_externs();
        assert_eq!(externs.len(), 20);
        // Point (8)
        assert!(externs.contains_key("Point::__init__"));
        assert!(externs.contains_key("Point::new"));
        assert!(externs.contains_key("Point::__get_x"));
        assert!(externs.contains_key("Point::__set_x"));
        assert!(externs.contains_key("Point::__get_y"));
        assert!(externs.contains_key("Point::__set_y"));
        assert!(externs.contains_key("Point::move_by"));
        assert!(externs.contains_key("Point::distance_to"));
        // Rect (12)
        assert!(externs.contains_key("Rect::__init__"));
        assert!(externs.contains_key("Rect::new"));
        assert!(externs.contains_key("Rect::__get_x"));
        assert!(externs.contains_key("Rect::__set_x"));
        assert!(externs.contains_key("Rect::__get_y"));
        assert!(externs.contains_key("Rect::__set_y"));
        assert!(externs.contains_key("Rect::__get_w"));
        assert!(externs.contains_key("Rect::__set_w"));
        assert!(externs.contains_key("Rect::__get_h"));
        assert!(externs.contains_key("Rect::__set_h"));
        assert!(externs.contains_key("Rect::area"));
        assert!(externs.contains_key("Rect::contains"));
    }

    #[test]
    fn point_new_handler() {
        let externs = anvyx_externs();
        let result =
            externs["Point::new"](vec![Value::Float(1.0), Value::Float(2.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle");
        };
        let x = externs["Point::__get_x"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(x, Value::Float(1.0));
    }

    #[test]
    fn point_getter_handlers() {
        let externs = anvyx_externs();
        let result =
            externs["Point::new"](vec![Value::Float(3.5), Value::Float(7.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle");
        };
        let x = externs["Point::__get_x"](vec![Value::ExternHandle(id.clone())]).unwrap();
        assert_eq!(x, Value::Float(3.5));
        let y = externs["Point::__get_y"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(y, Value::Float(7.0));
    }

    #[test]
    fn point_move_by_handler() {
        let externs = anvyx_externs();
        let result =
            externs["Point::new"](vec![Value::Float(10.0), Value::Float(20.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle");
        };
        externs["Point::move_by"](vec![
            Value::ExternHandle(id.clone()),
            Value::Float(5.0),
            Value::Float(-3.0),
        ])
        .unwrap();
        let x = externs["Point::__get_x"](vec![Value::ExternHandle(id.clone())]).unwrap();
        assert_eq!(x, Value::Float(15.0));
        let y = externs["Point::__get_y"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(y, Value::Float(17.0));
    }

    #[test]
    fn point_distance_to_handler() {
        let externs = anvyx_externs();
        let a = externs["Point::new"](vec![Value::Float(0.0), Value::Float(0.0)]).unwrap();
        let Value::ExternHandle(aid) = a else {
            panic!("expected ExternHandle");
        };
        let b = externs["Point::new"](vec![Value::Float(3.0), Value::Float(4.0)]).unwrap();
        let Value::ExternHandle(bid) = b else {
            panic!("expected ExternHandle");
        };
        let result = externs["Point::distance_to"](vec![
            Value::ExternHandle(aid),
            Value::ExternHandle(bid),
        ])
        .unwrap();
        assert_eq!(result, Value::Float(5.0));
    }

    #[test]
    fn rect_field_handlers() {
        let externs = anvyx_externs();
        let r = externs["Rect::new"](vec![
            Value::Float(1.0),
            Value::Float(2.0),
            Value::Float(3.0),
            Value::Float(4.0),
        ])
        .unwrap();
        let Value::ExternHandle(rid) = r else {
            panic!("expected ExternHandle for rect");
        };
        assert_eq!(
            externs["Rect::__get_x"](vec![Value::ExternHandle(rid.clone())]).unwrap(),
            Value::Float(1.0)
        );
        assert_eq!(
            externs["Rect::__get_y"](vec![Value::ExternHandle(rid.clone())]).unwrap(),
            Value::Float(2.0)
        );
        assert_eq!(
            externs["Rect::__get_w"](vec![Value::ExternHandle(rid.clone())]).unwrap(),
            Value::Float(3.0)
        );
        assert_eq!(
            externs["Rect::__get_h"](vec![Value::ExternHandle(rid.clone())]).unwrap(),
            Value::Float(4.0)
        );
        externs["Rect::__set_x"](vec![Value::ExternHandle(rid.clone()), Value::Float(10.0)])
            .unwrap();
        assert_eq!(
            externs["Rect::__get_x"](vec![Value::ExternHandle(rid)]).unwrap(),
            Value::Float(10.0)
        );
    }

    #[test]
    fn rect_contains_handler() {
        let externs = anvyx_externs();

        let p = externs["Point::new"](vec![Value::Float(5.0), Value::Float(5.0)]).unwrap();
        let Value::ExternHandle(pid) = p else {
            panic!("expected ExternHandle for point");
        };

        let r = externs["Rect::new"](vec![
            Value::Float(0.0),
            Value::Float(0.0),
            Value::Float(10.0),
            Value::Float(10.0),
        ])
        .unwrap();
        let Value::ExternHandle(rid) = r else {
            panic!("expected ExternHandle for rect");
        };

        let result =
            externs["Rect::contains"](vec![Value::ExternHandle(rid.clone()), Value::ExternHandle(pid.clone())])
                .unwrap();
        assert_eq!(result, Value::Bool(true));

        externs["Point::move_by"](vec![
            Value::ExternHandle(pid.clone()),
            Value::Float(20.0),
            Value::Float(0.0),
        ])
        .unwrap();

        let result =
            externs["Rect::contains"](vec![Value::ExternHandle(rid), Value::ExternHandle(pid)])
                .unwrap();
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn metadata_format() {
        let json = exports_to_json(ANVYX_EXPORTS, &anvyx_type_exports());

        assert!(json.contains("\"Point\""));
        assert!(json.contains("\"Rect\""));

        assert!(json.contains("\"fields\":["));
        assert!(json.contains("\"methods\":["));
        assert!(json.contains("\"statics\":["));

        assert!(json.contains("\"ret\":\"Point\""));
        assert!(json.contains("\"ret\":\"float\""));
        assert!(json.contains("\"ret\":\"bool\""));
        assert!(json.contains("\"init\":true"));
    }

    #[test]
    fn cleanup_point_dropped_when_handle_dropped() {
        let externs = anvyx_externs();
        let result = externs["Point::new"](vec![Value::Float(1.0), Value::Float(2.0)]).unwrap();
        let Value::ExternHandle(handle) = result else {
            panic!("expected ExternHandle");
        };
        let id = handle.id;
        drop(handle);
        __ANVYX_STORE_POINT.with(|s| assert!(s.borrow().borrow(id).is_err()));
    }

    #[test]
    fn cleanup_loop_no_leak() {
        let externs = anvyx_externs();
        for i in 0..50 {
            let _ = externs["Point::new"](vec![Value::Float(i as f64), Value::Float(0.0)]).unwrap();
        }
        __ANVYX_STORE_POINT.with(|s| assert_eq!(s.borrow().len(), 0));
    }

    #[test]
    fn point_auto_init_field_order() {
        let externs = anvyx_externs();
        // __init__ expects args in field declaration order: x, y
        let result = externs["Point::__init__"](vec![Value::Float(7.0), Value::Float(3.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle");
        };
        let x = externs["Point::__get_x"](vec![Value::ExternHandle(id.clone())]).unwrap();
        assert_eq!(x, Value::Float(7.0));
        let y = externs["Point::__get_y"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(y, Value::Float(3.0));
    }

    #[test]
    fn rect_init_round_trip() {
        let externs = anvyx_externs();
        let r = externs["Rect::__init__"](vec![
            Value::Float(10.0),
            Value::Float(20.0),
            Value::Float(100.0),
            Value::Float(50.0),
        ])
        .unwrap();
        let Value::ExternHandle(rid) = r else {
            panic!("expected ExternHandle");
        };
        assert_eq!(
            externs["Rect::__get_x"](vec![Value::ExternHandle(rid.clone())]).unwrap(),
            Value::Float(10.0)
        );
        assert_eq!(
            externs["Rect::__get_y"](vec![Value::ExternHandle(rid.clone())]).unwrap(),
            Value::Float(20.0)
        );
        assert_eq!(
            externs["Rect::__get_w"](vec![Value::ExternHandle(rid.clone())]).unwrap(),
            Value::Float(100.0)
        );
        assert_eq!(
            externs["Rect::__get_h"](vec![Value::ExternHandle(rid)]).unwrap(),
            Value::Float(50.0)
        );
    }

    #[test]
    fn point_init_move_destructure() {
        let externs = anvyx_externs();
        let result =
            externs["Point::__init__"](vec![Value::Float(10.0), Value::Float(20.0)]).unwrap();
        let Value::ExternHandle(id) = result else {
            panic!("expected ExternHandle");
        };
        externs["Point::move_by"](vec![
            Value::ExternHandle(id.clone()),
            Value::Float(5.0),
            Value::Float(-3.0),
        ])
        .unwrap();
        let x = externs["Point::__get_x"](vec![Value::ExternHandle(id.clone())]).unwrap();
        assert_eq!(x, Value::Float(15.0));
        let y = externs["Point::__get_y"](vec![Value::ExternHandle(id)]).unwrap();
        assert_eq!(y, Value::Float(17.0));
    }

    #[test]
    fn rect_init_contains_point() {
        let externs = anvyx_externs();
        let p = externs["Point::__init__"](vec![Value::Float(5.0), Value::Float(5.0)]).unwrap();
        let Value::ExternHandle(pid) = p else {
            panic!("expected ExternHandle for point");
        };
        let r = externs["Rect::__init__"](vec![
            Value::Float(0.0),
            Value::Float(0.0),
            Value::Float(10.0),
            Value::Float(10.0),
        ])
        .unwrap();
        let Value::ExternHandle(rid) = r else {
            panic!("expected ExternHandle for rect");
        };
        let result = externs["Rect::contains"](vec![
            Value::ExternHandle(rid),
            Value::ExternHandle(pid),
        ])
        .unwrap();
        assert_eq!(result, Value::Bool(true));
    }
}
