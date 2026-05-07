"""
Prompt template loader.

Templates live in ``prompts/`` as ``.txt`` files. Variables use Jinja2 syntax
(e.g. ``{{ class_name }}``) and are filled in at render time.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def render_prompt(
    template_name: str,
    prompts_dir: Path | None = None,
    **kwargs,
) -> str:
    path = (prompts_dir or _PROMPTS_DIR) / template_name
    template_str = path.read_text(encoding="utf-8").strip()

    if kwargs:
        env = Environment(keep_trailing_newline=False)
        tmpl = env.from_string(template_str)
        return tmpl.render(**kwargs).strip()

    return template_str


def depth_prompt(**kw) -> str:
    return render_prompt("depth.txt", **kw)


def normals_prompt(**kw) -> str:
    return render_prompt("normals.txt", **kw)

