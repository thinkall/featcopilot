"""Generate the API reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

src = Path(__file__).parent.parent / "featcopilot"

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src.parent)
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.with_suffix("").parts)

    # Skip __pycache__ and private modules
    if "__pycache__" in parts or any(part.startswith("_") for part in parts[1:]):
        continue

    # Skip __init__ but include the package
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    if not parts:
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"# {parts[-1]}\n\n::: {identifier}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src.parent))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
