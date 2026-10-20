"""
Compare two Huridocs API responses to check if AMP (mixed precision) changed output quality.

Usage:
    python compare_amp_results.py results_no_amp.json results_amp.json

What it checks:
    1. Same number of segments detected
    2. Same segment types (table, heading, text, etc.) on each page
    3. Same text content in each segment
    4. Bounding boxes are close (within a small tolerance for float16 rounding)
"""

import json
import sys


BBOX_TOLERANCE = 2.0  # pixels — float16 rounding can shift boxes by ~1-2px


def load_segments(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected JSON structure in {path}")


def group_by_page(segments):
    pages = {}
    for seg in segments:
        pn = seg["page_number"]
        pages.setdefault(pn, []).append(seg)
    for page_segs in pages.values():
        page_segs.sort(key=lambda s: (s["top"], s["left"]))
    return pages


def bbox_close(a, b, tol=BBOX_TOLERANCE):
    for key in ("left", "top", "width", "height"):
        if abs(a[key] - b[key]) > tol:
            return False
    return True


def find_best_match(seg, candidates):
    for i, c in enumerate(candidates):
        if c["type"] == seg["type"] and bbox_close(seg, c):
            return i
    for i, c in enumerate(candidates):
        if bbox_close(seg, c, tol=BBOX_TOLERANCE * 3):
            return i
    return None


def compare(baseline_path, amp_path):
    baseline = load_segments(baseline_path)
    amp = load_segments(amp_path)

    print(f"Baseline: {len(baseline)} segments from {baseline_path}")
    print(f"AMP:      {len(amp)} segments from {amp_path}")
    print()

    if len(baseline) != len(amp):
        print(f"!! SEGMENT COUNT DIFFERS: {len(baseline)} vs {len(amp)}")
    else:
        print("Segment count: SAME")

    baseline_pages = group_by_page(baseline)
    amp_pages = group_by_page(amp)

    all_page_nums = sorted(set(baseline_pages.keys()) | set(amp_pages.keys()))
    if set(baseline_pages.keys()) != set(amp_pages.keys()):
        missing_in_amp = set(baseline_pages.keys()) - set(amp_pages.keys())
        missing_in_baseline = set(amp_pages.keys()) - set(baseline_pages.keys())
        if missing_in_amp:
            print(f"!! Pages in baseline but NOT in AMP: {sorted(missing_in_amp)}")
        if missing_in_baseline:
            print(f"!! Pages in AMP but NOT in baseline: {sorted(missing_in_baseline)}")
    else:
        print(f"Pages detected: SAME ({len(all_page_nums)} pages)")
    print()

    total_matched = 0
    total_type_mismatches = 0
    total_text_mismatches = 0
    total_unmatched_baseline = 0
    total_unmatched_amp = 0
    max_bbox_drift = 0.0
    bbox_drifts = []
    issues = []

    for pn in all_page_nums:
        b_segs = list(baseline_pages.get(pn, []))
        a_segs = list(amp_pages.get(pn, []))

        if len(b_segs) != len(a_segs):
            issues.append(f"  Page {pn}: segment count {len(b_segs)} vs {len(a_segs)}")

        remaining_amp = list(range(len(a_segs)))

        for b_seg in b_segs:
            candidates = [a_segs[i] for i in remaining_amp]
            match_idx = find_best_match(b_seg, candidates)

            if match_idx is None:
                total_unmatched_baseline += 1
                issues.append(
                    f"  Page {pn}: baseline segment NOT FOUND in AMP — "
                    f"type={b_seg['type']}, text={b_seg['text'][:60]!r}"
                )
                continue

            a_seg = candidates[match_idx]
            remaining_amp.pop(match_idx)
            total_matched += 1

            if b_seg["type"] != a_seg["type"]:
                total_type_mismatches += 1
                issues.append(
                    f"  Page {pn}: TYPE changed '{b_seg['type']}' -> '{a_seg['type']}' "
                    f"for text={b_seg['text'][:60]!r}"
                )

            if b_seg["text"].strip() != a_seg["text"].strip():
                total_text_mismatches += 1
                b_text = b_seg["text"][:80]
                a_text = a_seg["text"][:80]
                issues.append(
                    f"  Page {pn}: TEXT changed\n"
                    f"    baseline: {b_text!r}\n"
                    f"    amp:      {a_text!r}"
                )

            for key in ("left", "top", "width", "height"):
                drift = abs(b_seg[key] - a_seg[key])
                if drift > 0:
                    bbox_drifts.append(drift)
                    max_bbox_drift = max(max_bbox_drift, drift)

        total_unmatched_amp += len(remaining_amp)
        for i in remaining_amp:
            a_seg = a_segs[i]
            issues.append(
                f"  Page {pn}: AMP has EXTRA segment — "
                f"type={a_seg['type']}, text={a_seg['text'][:60]!r}"
            )

    # Print summary
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Segments matched:        {total_matched}")
    print(f"Type mismatches:         {total_type_mismatches}")
    print(f"Text mismatches:         {total_text_mismatches}")
    print(f"Unmatched in baseline:   {total_unmatched_baseline}")
    print(f"Unmatched in AMP:        {total_unmatched_amp}")
    print()

    if bbox_drifts:
        avg_drift = sum(bbox_drifts) / len(bbox_drifts)
        print(f"Bounding box drift (pixels):")
        print(f"  Max:     {max_bbox_drift:.2f}")
        print(f"  Average: {avg_drift:.2f}")
        print(f"  Values with any drift: {len(bbox_drifts)}")
    else:
        print("Bounding box drift: ZERO (perfectly identical)")
    print()

    if issues:
        print(f"ISSUES FOUND ({len(issues)}):")
        print("-" * 60)
        for issue in issues:
            print(issue)
        print()

    # Final verdict
    is_ok = (
        total_type_mismatches == 0
        and total_text_mismatches == 0
        and total_unmatched_baseline == 0
        and total_unmatched_amp == 0
    )

    if is_ok and max_bbox_drift <= BBOX_TOLERANCE:
        print("VERDICT: PASS — AMP output is equivalent to baseline.")
        print("(Bounding box differences, if any, are within float16 rounding tolerance.)")
        return 0
    elif is_ok and max_bbox_drift > BBOX_TOLERANCE:
        print(f"VERDICT: WARNING — Same segments and text, but bounding boxes drifted up to {max_bbox_drift:.1f}px.")
        print("This is unusual for AMP. Check if results are still acceptable visually.")
        return 1
    else:
        print("VERDICT: FAIL — AMP changed the output meaningfully.")
        print("Consider disabling AMP (set CUDA_AMP=false) or investigating the differences above.")
        return 2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_amp_results.py <baseline.json> <amp.json>")
        print("  baseline.json = API response with CUDA_AMP=false")
        print("  amp.json      = API response with CUDA_AMP=true")
        sys.exit(1)

    sys.exit(compare(sys.argv[1], sys.argv[2]))
