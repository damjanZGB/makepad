//! A self-intersecting contour must fill by the non-zero winding rule.
//!
//! The motivating case is the pentagram: five straight segments that cross each
//! other, which is the compact way SVG authors draw a five-pointed star (the
//! Turkish flag's star is exactly this). Under SVG's default `fill-rule:
//! nonzero` a browser fills it SOLID -- the centre pentagon has winding 2 and
//! each arm has winding 1, and both are non-zero.
//!
//! Two independent defects used to make this render as a trapezoid:
//!
//! 1. `SweepTessellator` only ever created events at input vertices, so two
//!    edges crossing in their interiors produced no event. Past such a crossing
//!    the active-edge list is out of order and every `upper_region_winding`
//!    downstream of it is wrong, so the regions handed to the fill rule are
//!    garbage. This is fill-rule independent -- it corrupts even-odd too.
//! 2. `Tessellator::fill` guessed the fill rule, defaulting to EVEN-ODD unless
//!    some subpath carried an explicit `PathCmd::Winding`. SVG's default is
//!    non-zero. Even with (1) fixed, even-odd fills a pentagram HOLLOW.
//!
//! These tests assert the rendered coverage rather than the triangle list,
//! because the triangulation is free to change and the coverage is not.

use makepad_svg::path::{FillRule, LineJoin, VectorPath};
use makepad_svg::tessellate::{Tessellator, VVertex};

/// The five points of the star in `resources/flags/TUR.svg`, in a plain
/// `0 0 1200 800` viewBox. Consecutive points are NOT adjacent star tips --
/// that is what makes the outline cross itself.
const PENTAGRAM: [(f32, f32); 5] = [
    (556.67, 400.0),
    (737.57, 341.23),
    (625.76, 495.11),
    (625.76, 304.89),
    (737.57, 458.77),
];

fn pentagram_path() -> VectorPath {
    let mut path = VectorPath::new();
    path.move_to(PENTAGRAM[0].0, PENTAGRAM[0].1);
    for p in &PENTAGRAM[1..] {
        path.line_to(p.0, p.1);
    }
    path.close();
    path
}

/// Reference answer, independent of the tessellator: the winding number of the
/// pentagram outline about a point, by the standard crossing count.
fn reference_winding(px: f32, py: f32) -> i32 {
    let mut winding = 0;
    for i in 0..PENTAGRAM.len() {
        let (x0, y0) = PENTAGRAM[i];
        let (x1, y1) = PENTAGRAM[(i + 1) % PENTAGRAM.len()];
        let side = (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0);
        if y0 <= py {
            if y1 > py && side > 0.0 {
                winding += 1;
            }
        } else if y1 <= py && side < 0.0 {
            winding -= 1;
        }
    }
    winding
}

/// True when `(px, py)` lands in any emitted triangle. Deliberately tests the
/// whole emitted set (body plus AA fringe): what matters is whether the pixel
/// gets painted, not which triangle paints it.
fn covered(verts: &[VVertex], indices: &[u32], px: f32, py: f32) -> bool {
    indices.chunks_exact(3).any(|tri| {
        let a = &verts[tri[0] as usize];
        let b = &verts[tri[1] as usize];
        let c = &verts[tri[2] as usize];
        let d1 = (px - b.x) * (a.y - b.y) - (a.x - b.x) * (py - b.y);
        let d2 = (px - c.x) * (b.y - c.y) - (b.x - c.x) * (py - c.y);
        let d3 = (px - a.x) * (c.y - a.y) - (c.x - a.x) * (py - a.y);
        let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
        let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
        !(has_neg && has_pos)
    })
}

fn tessellate(path: &VectorPath, fill_rule: FillRule) -> (Vec<VVertex>, Vec<u32>) {
    let mut tess = Tessellator::default();
    tess.set_fill_rule(Some(fill_rule));
    tess.flatten(path, 0.25);
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    tess.fill(1.0, LineJoin::Miter, 4.0, false, &mut verts, &mut indices);
    (verts, indices)
}

/// Points chosen so the expectation is derived, not asserted by hand: the
/// reference winding says which are interior under non-zero.
///
/// (656.67, 400) is the centre pentagon, winding 2 -- the region that even-odd
/// leaves hollow and the missing-crossing bug got wrong. The four others are
/// arm interiors at winding 1. The last two are outside the star entirely.
const PROBES: [(&str, f32, f32); 8] = [
    ("centre pentagon", 656.67, 400.0),
    ("left arm", 575.0, 400.0),
    ("upper-right arm", 700.0, 360.0),
    ("lower-right arm", 700.0, 440.0),
    ("top arm", 640.0, 330.0),
    ("bottom arm", 640.0, 470.0),
    ("left of the star", 500.0, 400.0),
    ("right of the star", 800.0, 400.0),
];

#[test]
fn the_probe_points_are_where_this_test_thinks_they_are() {
    // Guards the guard. If the reference winding disagrees with the intent
    // encoded in the names, every other assertion below is meaningless.
    assert_eq!(reference_winding(656.67, 400.0).abs(), 2, "centre pentagon");
    for (name, x, y) in &PROBES[1..6] {
        assert_eq!(reference_winding(*x, *y).abs(), 1, "{name} should be an arm");
    }
    for (name, x, y) in &PROBES[6..] {
        assert_eq!(reference_winding(*x, *y), 0, "{name} should be outside");
    }
}

#[test]
fn a_pentagram_fills_solid_under_non_zero_winding() {
    let (verts, indices) = tessellate(&pentagram_path(), FillRule::NonZero);
    assert!(!indices.is_empty(), "tessellation produced no triangles");

    for (name, x, y) in PROBES {
        let want = reference_winding(x, y) != 0;
        assert_eq!(
            covered(&verts, &indices, x, y),
            want,
            "{name} at ({x}, {y}): non-zero winding says filled={want}"
        );
    }
}

#[test]
fn a_pentagram_fills_hollow_under_even_odd() {
    // Even-odd stays a separately selectable rule and must keep its own answer:
    // the arms fill, the centre pentagon (winding 2) does not. This is the test
    // that fails if the crossing-event fix is "simplified" into forcing
    // non-zero everywhere.
    let (verts, indices) = tessellate(&pentagram_path(), FillRule::EvenOdd);
    assert!(!indices.is_empty(), "tessellation produced no triangles");

    for (name, x, y) in PROBES {
        let want = reference_winding(x, y) % 2 != 0;
        assert_eq!(
            covered(&verts, &indices, x, y),
            want,
            "{name} at ({x}, {y}): even-odd says filled={want}"
        );
    }
}

#[test]
fn a_non_self_intersecting_star_is_unaffected() {
    // The 10-vertex star renders correctly today. It has no crossings, so the
    // planarisation pass must be a no-op for it and both fill rules must agree.
    const STAR: [(f32, f32); 10] = [
        (600.0, 240.0),
        (634.0, 346.0),
        (745.0, 346.0),
        (655.0, 412.0),
        (689.0, 518.0),
        (600.0, 452.0),
        (511.0, 518.0),
        (545.0, 412.0),
        (455.0, 346.0),
        (566.0, 346.0),
    ];
    let mut path = VectorPath::new();
    path.move_to(STAR[0].0, STAR[0].1);
    for p in &STAR[1..] {
        path.line_to(p.0, p.1);
    }
    path.close();

    for rule in [FillRule::NonZero, FillRule::EvenOdd] {
        let (verts, indices) = tessellate(&path, rule);
        assert!(covered(&verts, &indices, 600.0, 380.0), "{rule:?} centre");
        assert!(covered(&verts, &indices, 600.0, 260.0), "{rule:?} top point");
        assert!(
            !covered(&verts, &indices, 600.0, 530.0),
            "{rule:?} below the star"
        );
    }
}

#[test]
fn two_nested_contours_wound_the_same_way_fill_solid_under_non_zero() {
    // The rule that separates non-zero from even-odd on non-self-intersecting
    // input, and the reason the fill rule cannot simply be guessed from the
    // geometry: same-direction nesting is a solid square under non-zero and a
    // square ring under even-odd. A browser draws the first.
    let mut path = VectorPath::new();
    path.rect(0.0, 0.0, 100.0, 100.0);
    path.rect(25.0, 25.0, 50.0, 50.0);

    let (verts, indices) = tessellate(&path, FillRule::NonZero);
    assert!(covered(&verts, &indices, 50.0, 50.0), "non-zero centre");

    let (verts, indices) = tessellate(&path, FillRule::EvenOdd);
    assert!(!covered(&verts, &indices, 50.0, 50.0), "even-odd centre");
}
