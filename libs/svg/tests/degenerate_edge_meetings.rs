//! Edges that meet at a shared point rather than crossing cleanly.
//!
//! `self_intersecting_fill.rs` covers the proper crossing -- two edges through
//! each other's interiors. This file covers everything else two edges can do,
//! because those cases were being decided by floating-point luck:
//!
//!   * a **T-junction**: one edge's endpoint lying on another edge's interior;
//!   * a **shared vertex**: two contours touching at exactly one point;
//!   * a **collinear overlap**: two edges lying along the same line, sharing a
//!     stretch of it;
//!   * a **zero-length edge**: a repeated point in the source path.
//!
//! WHY THESE ARE NOT EXOTIC. An SVG author who wants a notch, an inset panel, a
//! quartered shield or two abutting bands types the same coordinate twice, and
//! gets a T-junction or a collinear overlap. It reaches this tessellator as
//! f32 that is collinear only to within rounding, because the numbers have been
//! through a viewBox transform, a node transform and Bezier flattening on the
//! way in. Deciding collinearity by comparing that against literal zero is a
//! coin flip: measured over 40 scale/rotation/offset combinations of the single
//! notched chevron below, 5 mis-filled, each across ~3% of the shape's area,
//! with no pattern to which. That is a hole or a blob appearing on one
//! federation's crest and not another's -- the exact failure that only shows up
//! at a venue.
//!
//! HOW THESE TESTS JUDGE. Not by hand-written expectations, and not by the
//! triangle list, which is free to change. Each shape is rasterised on a grid
//! and compared against its own winding number computed independently in `f64`
//! from the source contours. Samples within a margin of any edge are skipped,
//! since the boundary is genuinely ambiguous there. Every shape is run through
//! several affine transforms, including the ones measured to fail, so the tests
//! pin behaviour across coordinate ranges rather than at one lucky scale.

use makepad_svg::path::{FillRule, LineJoin, VectorPath};
use makepad_svg::tessellate::{Tessellator, VVertex};

type Pt = (f64, f64);

/// Affine transforms each shape is checked under, as (scale, radians, offset).
///
/// The rotations are the point: an axis-aligned shape has horizontal and
/// vertical edges, for which the collinearity determinant contains an exactly
/// zero factor and therefore comes out exact no matter how ugly the
/// coordinates. Those cases passed even before the fix. Rotating puts every
/// edge on a slant, which is where real artwork lives and where the arithmetic
/// stops being exact. `(1.732, 0.785, ...)` and `(13.7, 0.785, ...)` are two of
/// the combinations measured to mis-fill.
const TRANSFORMS: [(f64, f64, Pt); 6] = [
    (1.0, 0.0, (0.0, 0.0)),
    (0.37, 0.1, (0.31415, 0.271828)),
    (1.7320508, 0.7853981, (0.0, 0.0)),
    (1.7320508, 0.1, (0.31415, 0.271828)),
    (13.7, 0.7853981, (0.31415, 0.271828)),
    (91.3, 0.7853981, (0.0, 0.0)),
];

fn transform(p: Pt, scale: f64, radians: f64, offset: Pt) -> Pt {
    let (cos, sin) = (radians.cos(), radians.sin());
    (
        (p.0 * cos - p.1 * sin) * scale + offset.0,
        (p.0 * sin + p.1 * cos) * scale + offset.1,
    )
}

/// The reference answer, computed in f64 straight from the source contours and
/// owing nothing to the tessellator.
fn reference_winding(contours: &[Vec<Pt>], px: f64, py: f64) -> i32 {
    let mut winding = 0;
    for contour in contours {
        for i in 0..contour.len() {
            let (x0, y0) = contour[i];
            let (x1, y1) = contour[(i + 1) % contour.len()];
            let side = (x1 - x0) * (py - y0) - (px - x0) * (y1 - y0);
            if y0 <= py {
                if y1 > py && side > 0.0 {
                    winding += 1;
                }
            } else if y1 <= py && side < 0.0 {
                winding -= 1;
            }
        }
    }
    winding
}

fn distance_to_nearest_edge(contours: &[Vec<Pt>], px: f64, py: f64) -> f64 {
    let mut best = f64::MAX;
    for contour in contours {
        for i in 0..contour.len() {
            let (x0, y0) = contour[i];
            let (x1, y1) = contour[(i + 1) % contour.len()];
            let (dx, dy) = (x1 - x0, y1 - y0);
            let len2 = dx * dx + dy * dy;
            let t = if len2 > 0.0 {
                (((px - x0) * dx + (py - y0) * dy) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let (qx, qy) = (x0 + dx * t, y0 + dy * t);
            best = best.min(((px - qx).powi(2) + (py - qy).powi(2)).sqrt());
        }
    }
    best
}

/// Which side of the directed line `q -> r` the sample falls on, with the
/// rounding error of that very expression as the threshold rather than a
/// comparison against literal zero -- the same filter, and the same reasoning,
/// as `orient` in the tessellator.
///
/// This is not decoration. Two triangles that share a diagonal both claim the
/// points ON that diagonal, and any rasteriser paints them; but evaluated in
/// f32 the sample can come out a hair outside BOTH, and the harness would then
/// report a hole that does not exist. Measured while writing these tests: two
/// samples of the collinear-overlap shape missed their triangle by 5.3e-7 and
/// 3.3e-8 of the triangle's own area, purely from this.
fn side_of(px: f32, py: f32, qx: f32, qy: f32, rx: f32, ry: f32) -> i32 {
    let left = (px - rx) * (qy - ry);
    let right = (qx - rx) * (py - ry);
    let det = left - right;
    let bound = 8.0 * f32::EPSILON * (left.abs() + right.abs());
    if det > bound {
        1
    } else if det < -bound {
        -1
    } else {
        0
    }
}

fn covered(verts: &[VVertex], indices: &[u32], px: f32, py: f32) -> bool {
    indices.chunks_exact(3).any(|tri| {
        let a = &verts[tri[0] as usize];
        let b = &verts[tri[1] as usize];
        let c = &verts[tri[2] as usize];
        let d1 = side_of(px, py, a.x, a.y, b.x, b.y);
        let d2 = side_of(px, py, b.x, b.y, c.x, c.y);
        let d3 = side_of(px, py, c.x, c.y, a.x, a.y);
        // A zero counts as on the boundary, which is compatible with either
        // orientation -- so the sample is inside unless two edges disagree.
        let has_neg = d1 < 0 || d2 < 0 || d3 < 0;
        let has_pos = d1 > 0 || d2 > 0 || d3 > 0;
        !(has_neg && has_pos)
    })
}

/// Tessellate `contours` and assert the filled area matches `fill_rule` applied
/// to the reference winding, everywhere away from the boundary.
///
/// AA is switched OFF (`aa = 0.0`). The fringe is deliberately allowed to spill
/// outside the shape -- up to the miter limit, four times the fringe width, at a
/// sharp corner -- so including it would flag correct antialiasing as a fill
/// error. That is not a hypothetical: with the fringe on, six of the forty
/// probe configurations reported exactly one outlying sample each, all of them
/// miter spill at the chevron's sharp points, and all of them present before and
/// after this fix. Winding correctness is what these tests are for, so the AA
/// geometry is taken out of the picture.
fn assert_fill_matches_reference(name: &str, contours: &[Vec<Pt>], fill_rule: FillRule) {
    for (scale, radians, offset) in TRANSFORMS {
        let moved: Vec<Vec<Pt>> = contours
            .iter()
            .map(|c| {
                c.iter()
                    .map(|&p| transform(p, scale, radians, offset))
                    .collect()
            })
            .collect();

        let mut path = VectorPath::new();
        for contour in &moved {
            path.move_to(contour[0].0 as f32, contour[0].1 as f32);
            for p in &contour[1..] {
                path.line_to(p.0 as f32, p.1 as f32);
            }
            path.close();
        }

        let mut tess = Tessellator::default();
        tess.set_fill_rule(Some(fill_rule));
        tess.flatten(&path, 0.25);
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        tess.fill(0.0, LineJoin::Miter, 4.0, false, &mut verts, &mut indices);

        let (mut min_x, mut min_y) = (f64::MAX, f64::MAX);
        let (mut max_x, mut max_y) = (f64::MIN, f64::MIN);
        for contour in &moved {
            for p in contour {
                min_x = min_x.min(p.0);
                min_y = min_y.min(p.1);
                max_x = max_x.max(p.0);
                max_y = max_y.max(p.1);
            }
        }
        let extent = (max_x - min_x).max(max_y - min_y);
        let margin = extent * 0.01;

        const GRID: usize = 48;
        let mut wrong = 0usize;
        let mut tested = 0usize;
        let mut first: Option<(f64, f64, bool)> = None;
        for gy in 0..GRID {
            for gx in 0..GRID {
                let px = min_x + (max_x - min_x) * (gx as f64 + 0.5) / GRID as f64;
                let py = min_y + (max_y - min_y) * (gy as f64 + 0.5) / GRID as f64;
                if distance_to_nearest_edge(&moved, px, py) < margin {
                    continue;
                }
                tested += 1;
                let winding = reference_winding(&moved, px, py);
                let want = match fill_rule {
                    FillRule::NonZero => winding != 0,
                    FillRule::EvenOdd => winding % 2 != 0,
                };
                let got = covered(&verts, &indices, px as f32, py as f32);
                if got != want {
                    wrong += 1;
                    first.get_or_insert((px, py, want));
                }
            }
        }

        assert!(tested > 100, "{name}: grid degenerated, only {tested} samples");
        assert_eq!(
            wrong,
            0,
            "{name} at scale={scale} radians={radians} offset={offset:?}: \
             {wrong} of {tested} samples disagree with the reference winding; \
             first at {:?} where filled should be {:?}",
            first.map(|f| (f.0, f.1)),
            first.map(|f| f.2),
        );
    }
}

/// A chevron with a triangular notch bitten out of it, where the notch's apex
/// sits on the interior of the chevron's slanted lower-left edge.
///
/// The T-junction is load-bearing: it is where the subtracting contour meets the
/// outer one, so getting it wrong does not nudge a boundary, it mis-classifies
/// whole regions. This is the shape whose 40 transforms were measured.
fn notched_chevron() -> Vec<Vec<Pt>> {
    let outer = vec![
        (0.0, 0.0),
        (100.0, 60.0),
        (200.0, 0.0),
        (200.0, 40.0),
        (100.0, 100.0),
        (0.0, 40.0),
    ];
    // (50, 30) is the midpoint of the outer edge (0,0) -> (100,60).
    let notch = vec![(50.0, 30.0), (80.0, 10.0), (20.0, 10.0)];
    vec![outer, notch]
}

#[test]
fn a_t_junction_does_not_corrupt_the_fill() {
    assert_fill_matches_reference("notched chevron", &notched_chevron(), FillRule::NonZero);
}

#[test]
fn a_t_junction_does_not_corrupt_the_even_odd_fill_either() {
    // The T-junction defect was never a fill-rule defect: it wrecks the region
    // structure before any rule is consulted. Pinning even-odd separately keeps
    // that honest, and keeps even-odd a real, independently correct rule.
    assert_fill_matches_reference("notched chevron", &notched_chevron(), FillRule::EvenOdd);
}

#[test]
fn two_contours_touching_at_a_single_shared_vertex() {
    // Both triangles name (50, 50) with the same literal, so the two contours
    // meet at a bit-identical point. Nothing should be split here: the sweep
    // already merges coincident events, and `spans_collinear_point` rejects
    // exact endpoint equality precisely so this case is left alone.
    let upper = vec![(0.0, 0.0), (100.0, 0.0), (50.0, 50.0)];
    let lower = vec![(50.0, 50.0), (100.0, 100.0), (0.0, 100.0)];
    assert_fill_matches_reference("bowtie on a shared vertex", &[upper, lower], FillRule::NonZero);
}

#[test]
fn a_repeated_point_does_not_break_the_fill() {
    // A zero-length edge, as a source path gets when the same coordinate is
    // written twice. `Tessellator::prepare_points` merges points closer than its
    // `dist_tol` before `fill()` ever sees them, so this should not even reach
    // the planarisation -- but that upstream defence is exactly the kind of
    // thing that gets refactored away, and a zero-length edge has a null
    // direction vector that would make `orient` read "collinear" against
    // everything. Both defences are pinned: the shape must simply fill.
    let mut path = VectorPath::new();
    path.move_to(0.0, 0.0);
    path.line_to(100.0, 0.0);
    path.line_to(100.0, 0.0); // exact duplicate
    path.line_to(50.0, 80.0);
    path.close();

    let mut tess = Tessellator::default();
    tess.set_fill_rule(Some(FillRule::NonZero));
    tess.flatten(&path, 0.25);
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    tess.fill(0.0, LineJoin::Miter, 4.0, false, &mut verts, &mut indices);

    assert!(!indices.is_empty(), "a repeated point emptied the triangulation");
    assert!(covered(&verts, &indices, 50.0, 20.0), "triangle interior");
    assert!(!covered(&verts, &indices, 5.0, 70.0), "outside the triangle");
}

#[test]
fn a_collinear_overlap_subtracts_correctly() {
    // A rectangle with a rectangular bite taken out of its right-hand side. The
    // bite is wound the opposite way, so under non-zero it subtracts -- and its
    // right edge lies ALONG the outer rectangle's right edge, sharing a stretch
    // of that line rather than crossing it, with a T-junction at each end.
    //
    // This is what an inset panel or an abutting band in a flag looks like, and
    // it is the case where "no single intersection point exists" used to make
    // the planarisation give up entirely.
    let outer = vec![(0.0, 0.0), (200.0, 0.0), (200.0, 100.0), (0.0, 100.0)];
    let bite = vec![(100.0, 25.0), (100.0, 75.0), (200.0, 75.0), (200.0, 25.0)];
    assert_fill_matches_reference("rect with a collinear bite", &[outer, bite], FillRule::NonZero);
}

#[test]
fn two_abutting_rectangles_fill_as_one_solid() {
    // Same-direction contours sharing part of an edge: the seam between them
    // carries +1 from one and -1 from the other and must cancel, leaving a solid
    // union with no crack and no doubled region. The shorter rectangle's corners
    // are T-junctions on the taller one's side.
    let tall = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
    let short = vec![(100.0, 25.0), (200.0, 25.0), (200.0, 75.0), (100.0, 75.0)];
    assert_fill_matches_reference("abutting rectangles", &[tall, short], FillRule::NonZero);
}
