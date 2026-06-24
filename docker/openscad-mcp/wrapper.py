import logging
import os
import subprocess
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OpenSCADWrapper:
    """
    Wrapper for OpenSCAD command-line interface.
    Provides methods to generate SCAD code, STL files, and preview images.
    """

    def __init__(self, scad_dir: str, output_dir: str):
        self.scad_dir = scad_dir
        self.output_dir = output_dir
        self.stl_dir = os.path.join(output_dir, "stl")
        self.preview_dir = os.path.join(output_dir, "preview")

        os.makedirs(self.scad_dir, exist_ok=True)
        os.makedirs(self.stl_dir, exist_ok=True)
        os.makedirs(self.preview_dir, exist_ok=True)

        self.shape_templates = {
            "cube": self._cube_template,
            "sphere": self._sphere_template,
            "cylinder": self._cylinder_template,
            "box": self._box_template,
            "rounded_box": self._rounded_box_template,
        }

    def _preview_distance(self, parameters: Optional[Dict[str, Any]] = None) -> float:
        """Pick a camera distance that frames the model regardless of corner vs centered geometry."""
        params = parameters or {}
        dims: List[float] = []
        for key in ("width", "depth", "height"):
            if key in params:
                try:
                    dims.append(abs(float(params[key])))
                except (TypeError, ValueError):
                    pass
        for key in ("radius", "outer_radius", "major_radius"):
            if key in params:
                try:
                    dims.append(abs(float(params[key])) * 2)
                except (TypeError, ValueError):
                    pass
        span = max(dims) if dims else 25.0
        return max(span * 4.0, 80.0)

    def _camera_positions(self, distance: float) -> Dict[str, str]:
        """
        Gimbal cameras (tx,ty,tz,rx,ry,rz,dist).

        Pure top/right orthographic views look like flat color blocks in headless
        OpenSCAD, so those views use a slight tilt to reveal edges and hollow features.
        """
        d = f"{distance:.1f}"
        return {
            "front": f"0,0,0,12,0,8,{d}",
            "top": f"0,0,0,75,0,15,{d}",
            "right": f"0,0,0,15,80,0,{d}",
            "perspective": f"0,0,0,25,0,35,{d}",
        }

    def _build_preview_command(
        self,
        scad_file: str,
        preview_file: str,
        camera_position: str,
        parameters: Optional[Dict[str, Any]] = None,
        image_size: str = "800,600",
    ) -> List[str]:
        cmd = [
            "openscad",
            "--render",
            "--viewall",
            "--autocenter",
            "--camera",
            camera_position,
            "--imgsize",
            image_size,
            "-o",
            preview_file,
        ]
        if parameters:
            for key, value in parameters.items():
                cmd.extend(["-D", f"{key}={value}"])
        cmd.append(scad_file)
        return cmd

    def generate_scad_code(self, model_type: str, parameters: Dict[str, Any]) -> str:
        model_id = str(uuid.uuid4())
        scad_file = os.path.join(self.scad_dir, f"{model_id}.scad")

        template_func = self.shape_templates.get(model_type)
        if not template_func:
            raise ValueError(f"Unsupported model type: {model_type}")

        scad_code = template_func(parameters)

        with open(scad_file, "w", encoding="utf-8") as f:
            f.write(scad_code)

        logger.info("Generated SCAD file: %s", scad_file)
        return scad_file

    def generate_scad(self, scad_code: str, model_id: str) -> str:
        scad_file = os.path.join(self.scad_dir, f"{model_id}.scad")

        with open(scad_file, "w", encoding="utf-8") as f:
            f.write(scad_code)

        logger.info("Generated SCAD file: %s", scad_file)
        return scad_file

    def update_scad_code(self, model_id: str, parameters: Dict[str, Any]) -> str:
        scad_file = os.path.join(self.scad_dir, f"{model_id}.scad")
        if not os.path.exists(scad_file):
            raise FileNotFoundError(f"SCAD file not found: {scad_file}")

        with open(scad_file, encoding="utf-8") as f:
            scad_code = f.read()

        model_type = None
        for shape_type in self.shape_templates:
            if shape_type in scad_code.lower():
                model_type = shape_type
                break

        if not model_type:
            raise ValueError("Could not determine model type from existing SCAD file")

        new_scad_code = self.shape_templates[model_type](parameters)

        with open(scad_file, "w", encoding="utf-8") as f:
            f.write(new_scad_code)

        logger.info("Updated SCAD file: %s", scad_file)
        return scad_file

    def generate_stl(self, scad_file: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        model_id = os.path.basename(scad_file).split(".")[0]
        stl_file = os.path.join(self.stl_dir, f"{model_id}.stl")

        cmd = ["openscad", "-o", stl_file]

        if parameters:
            for key, value in parameters.items():
                cmd.extend(["-D", f"{key}={value}"])

        cmd.append(scad_file)

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Generated STL file: %s", stl_file)
            logger.debug(result.stdout)
            return stl_file
        except subprocess.CalledProcessError as e:
            logger.error("Error generating STL file: %s", e.stderr)
            raise RuntimeError(f"Failed to generate STL file: {e.stderr}") from e

    def generate_preview(
        self,
        scad_file: str,
        parameters: Optional[Dict[str, Any]] = None,
        camera_position: Optional[str] = None,
        image_size: str = "800,600",
    ) -> str:
        model_id = os.path.basename(scad_file).split(".")[0]
        preview_file = os.path.join(self.preview_dir, f"{model_id}.png")
        if camera_position is None:
            camera_position = self._camera_positions(self._preview_distance(parameters))["perspective"]

        cmd = self._build_preview_command(scad_file, preview_file, camera_position, parameters, image_size)

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Generated preview image: %s", preview_file)
            logger.debug(result.stdout)
            return preview_file
        except subprocess.CalledProcessError as e:
            logger.error("Error generating preview image: %s", e.stderr)
            logger.warning("Using placeholder image due to rendering error")
            return self._create_placeholder_image(preview_file)

    def _create_placeholder_image(self, output_path: str) -> str:
        try:
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (800, 600), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            draw.text((400, 300), "Preview not available", fill=(0, 0, 0))
            img.save(output_path)
            return output_path
        except Exception as e:
            logger.error("Error creating placeholder image: %s", e)
            return output_path

    def generate_multi_angle_previews(
        self, scad_file: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        distance = self._preview_distance(parameters)
        camera_positions = self._camera_positions(distance)
        model_id = os.path.basename(scad_file).split(".")[0]

        previews: Dict[str, str] = {}
        for view, camera_position in camera_positions.items():
            preview_file = os.path.join(self.preview_dir, f"{model_id}_{view}.png")
            cmd = self._build_preview_command(scad_file, preview_file, camera_position, parameters)

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info("Generated %s preview: %s", view, preview_file)
                previews[view] = preview_file
            except subprocess.CalledProcessError as e:
                logger.error("Error generating %s preview: %s", view, e.stderr)
                previews[view] = self._create_placeholder_image(preview_file)

        return previews

    def _cube_template(self, params: Dict[str, Any]) -> str:
        size_x = params.get("width", 10)
        size_y = params.get("depth", 10)
        size_z = params.get("height", 10)
        center = params.get("center", "false").lower() == "true"

        return f"""// Cube
width = {size_x};
depth = {size_y};
height = {size_z};
center = {str(center).lower()};

cube([width, depth, height], center=center);
"""

    def _sphere_template(self, params: Dict[str, Any]) -> str:
        radius = params.get("radius", 10)
        segments = params.get("segments", 32)

        return f"""// Sphere
radius = {radius};
$fn = {segments};

sphere(r=radius);
"""

    def _cylinder_template(self, params: Dict[str, Any]) -> str:
        radius = params.get("radius", 10)
        height = params.get("height", 20)
        center = params.get("center", "false").lower() == "true"
        segments = params.get("segments", 32)

        return f"""// Cylinder
radius = {radius};
height = {height};
center = {str(center).lower()};
$fn = {segments};

cylinder(h=height, r=radius, center=center);
"""

    def _box_template(self, params: Dict[str, Any]) -> str:
        width = params.get("width", 30)
        depth = params.get("depth", 20)
        height = params.get("height", 15)
        thickness = params.get("thickness", 2)

        return f"""// Hollow Box
width = {width};
depth = {depth};
height = {height};
thickness = {thickness};

module box(width, depth, height, thickness) {{
    difference() {{
        cube([width, depth, height]);
        translate([thickness, thickness, thickness])
        cube([width - 2*thickness, depth - 2*thickness, height - thickness]);
    }}
}}

box(width, depth, height, thickness);
"""

    def _rounded_box_template(self, params: Dict[str, Any]) -> str:
        width = params.get("width", 30)
        depth = params.get("depth", 20)
        height = params.get("height", 15)
        radius = params.get("radius", 3)
        segments = params.get("segments", 32)

        return f"""// Rounded Box
width = {width};
depth = {depth};
height = {height};
radius = {radius};
$fn = {segments};

module rounded_box(width, depth, height, radius) {{
    hull() {{
        translate([radius, radius, radius])
        sphere(r=radius);

        translate([width-radius, radius, radius])
        sphere(r=radius);

        translate([radius, depth-radius, radius])
        sphere(r=radius);

        translate([width-radius, depth-radius, radius])
        sphere(r=radius);

        translate([radius, radius, height-radius])
        sphere(r=radius);

        translate([width-radius, radius, height-radius])
        sphere(r=radius);

        translate([radius, depth-radius, height-radius])
        sphere(r=radius);

        translate([width-radius, depth-radius, height-radius])
        sphere(r=radius);
    }}
}}

rounded_box(width, depth, height, radius);
"""
