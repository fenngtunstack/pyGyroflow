"""WGSL shader builder for the GPU undistortion pipeline.

Reads the wgpu_undistort.wgsl template and injects lens distortion model
functions, selects buffer vs texture input path, and sets the scalar type.
This mirrors the shader compilation logic in Gyroflow's wgpu.rs.
"""

from __future__ import annotations

import re
from pathlib import Path

SHADER_DIR = Path(__file__).parent / "shaders"


def build_undistort_shader(
    distortion_model_wgsl: str,
    digital_lens_wgsl: str = "",
    use_buffer_input: bool = True,
    scalar_type: str = "f32",
) -> str:
    """Build the WGSL undistort shader by injecting distortion model functions.

    The wgpu_undistort.wgsl template has these placeholders:
    - LENS_MODEL_FUNCTIONS; -> replaced with distortion model WGSL code
    - {texture_input}...{/texture_input} -> removed for buffer input path
    - {buffer_input}...{/buffer_input} -> removed for texture input path
    - SCALAR -> replaced with scalar type

    Args:
        distortion_model_wgsl: WGSL code for distort_point and undistort_point.
        digital_lens_wgsl: WGSL code for digital lens, or empty for passthrough.
        use_buffer_input: True = use storage buffers, False = use textures.
        scalar_type: "f32" or "u32".

    Returns:
        Complete WGSL shader code ready for wgpu-py.
    """
    template = (SHADER_DIR / "wgpu_undistort.wgsl").read_text()

    # 1. Inject lens model functions
    lens_code = distortion_model_wgsl
    if digital_lens_wgsl:
        lens_code += "\n" + digital_lens_wgsl
    else:
        lens_code += """
fn digital_undistort_point(uv: vec2<f32>) -> vec2<f32> { return uv; }
fn digital_distort_point(uv: vec2<f32>) -> vec2<f32> { return uv; }
"""
    template = template.replace("LENS_MODEL_FUNCTIONS;", lens_code)

    # 2. Select input path -- remove the blocks we don't need.
    # The template uses {texture_input}..{/texture_input} and
    # {buffer_input}..{/buffer_input} as conditional blocks.
    # When using buffer input (compute pipeline), strip texture blocks.
    # When using texture input (render pipeline), strip buffer blocks.
    if use_buffer_input:
        template = _strip_blocks(template, "texture_input")
    else:
        template = _strip_blocks(template, "buffer_input")

    # 3. Replace scalar type placeholder
    template = template.replace("SCALAR", scalar_type)

    # 4. Remove @fragment from variable declarations (compat with wgpu >= 0.20).
    #    @fragment on var declarations is only valid in fragment render pipelines.
    #    For compute pipelines, plain var declarations are required.
    template = re.sub(r"@fragment\s+var", "var", template)

    return template


def _strip_blocks(text: str, tag: str) -> str:
    """Remove all {tag}...{/tag} blocks from *text*.

    The opening marker is ``{tag}`` and the closing marker is ``{/tag}``.
    Both markers and everything between them are removed. Multiple
    non-nested occurrences are handled in a loop.
    """
    open_marker = "{" + tag + "}"
    close_marker = "{/" + tag + "}"
    while open_marker in text:
        start = text.index(open_marker)
        end = text.index(close_marker) + len(close_marker)
        text = text[:start] + text[end:]
    return text
