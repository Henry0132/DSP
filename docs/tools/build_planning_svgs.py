"""Extract the paper's existing subgoal sequence into animatable SVG layers.

Usage: python docs/tools/build_planning_svgs.py --figure-dir /path/to/giant
Requires PyMuPDF. No trajectories are generated or interpolated.
"""

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import fitz

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)


def build(figure_dir, name, expected_count, output):
    with fitz.open(figure_dir / f"{name}_giant_value_function_task_1.pdf") as doc:
        page = doc[0]
        root = ET.fromstring(page.get_svg_image(text_as_path=True))
        drawings = page.get_drawings()
        markers = [d for d in drawings if d["fill"] == (0, 1, 1)]
        assert len(markers) == expected_count
        centers = [((d["rect"].x0 + d["rect"].x1) / 2,
                    (d["rect"].y0 + d["rect"].y1) / 2) for d in markers]
        parents = {child: parent for parent in root.iter() for child in parent}
        circles = [e for e in root.iter() if e.get("fill") == "#00ffff"]
        assert len(circles) == expected_count
        for i, circle in enumerate(circles):
            # The PDF exporter places the circle's fill and outline in one group.
            group = parents[circle]
            assert len(group) == 2 and group[1].get("stroke") == "#ffffff"
            group.set("data-step", str(i))

        # Keep the original glyph outlines, including the exact label positions.
        glyphs = [e for e in root.iter() if e.get("data-text", "").isdigit()
                  and e.get("transform", "").startswith("matrix(9,0,0,-9,")]
        assert "".join(e.get("data-text") for e in glyphs) == "".join(map(str, range(expected_count)))
        cursor = 0
        for i in range(expected_count):
            for glyph in glyphs[cursor:cursor + len(str(i))]:
                glyph.set("data-step", str(i))
            cursor += len(str(i))

        # Split only the existing dashed connector at its original vertices.
        connector = [d for d in drawings if d["color"] == (1, 1, 1)
                     and len(d["items"]) == expected_count - 1]
        assert len(connector) == 1
        for i, item in enumerate(connector[0]["items"]):
            assert item[0] == "l"
            assert abs(item[1].x - centers[i][0]) < .001
            assert abs(item[1].y - centers[i][1]) < .001
            assert abs(item[2].x - centers[i + 1][0]) < .001
            assert abs(item[2].y - centers[i + 1][1]) < .001
        paths = [e for e in root.iter() if e.get("stroke") == "#ffffff"
                 and e.get("stroke-dasharray")]
        assert len(paths) == 1
        original = paths[0]
        container = parents[original]
        container.remove(original)
        for i, (_, start, end) in enumerate(connector[0]["items"], 1):
            attributes = dict(original.attrib)
            attributes.pop("transform", None)
            attributes.update({"d": f"M{start.x},{start.y} L{end.x},{end.y}",
                               "data-step": str(i)})
            ET.SubElement(container, f"{{{SVG}}}path", attributes)

        if name == "HIQL":
            # Preserve the red annotation from the paper's annotated PDF.
            with fitz.open(figure_dir / "HIQL_giant_value_function_task_1_nn.pdf") as annotated:
                annotation_root = ET.fromstring(annotated[0].get_svg_image())
                box = [e for e in annotation_root.iter() if e.get("fill") == "#e52237"]
                assert len(box) == 1
                # Subgoals 12–18 form the local cluster; 19 moves away from it.
                # Reveal the annotation only after that local sequence is complete.
                group = ET.SubElement(root, f"{{{SVG}}}g", {
                    "data-step": "19", "data-annotation": "cluster",
                    "transform": f"scale({page.rect.width / annotated[0].rect.width} {page.rect.height / annotated[0].rect.height})"})
                group.append(copy.deepcopy(box[0]))

        # Both figures are inserted into the same HTML document.
        prefix = name.lower() + "-"
        for element in root.iter():
            for key, value in list(element.attrib.items()):
                if key == "id":
                    element.set(key, prefix + value)
                elif value.startswith("#") and key.endswith("href"):
                    element.set(key, "#" + prefix + value[1:])
                elif "url(#" in value:
                    element.set(key, value.replace("url(#", "url(#" + prefix))
        root.set("data-subgoal-count", str(expected_count))
        root.set("role", "img")
        root.set("aria-label", f"{name} numbered subgoal sequence from the paper, AntMaze Giant task 1")
        root.insert(0, ET.Comment("Extracted from the paper PDF; animation order is the original numbered subgoal order."))
        output.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
        print(f"{output.name}: {expected_count} original subgoals; connector coordinates verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    assets = Path(__file__).resolve().parents[1] / "assets"
    for name, count in [("DSP", 35), ("HIQL", 50)]:
        build(args.figure_dir, name, count, assets / f"{name.lower()}-planning-steps.svg")
