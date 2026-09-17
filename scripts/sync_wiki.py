"""Copia las notas de la bóveda de Obsidian a docs/content para publicarlas con Quartz.

La nota "Red neuronal from scratch" pasa a ser index.md (la portada del sitio) y los
wikilinks que apuntan a ella se reescriben para que sigan resolviendo. Todo lo demás
se copia tal cual: Quartz entiende wikilinks, callouts, frontmatter y LaTeX de Obsidian.

Uso (dentro de nix develop):
    python scripts/sync_wiki.py
"""

import re
import shutil
from pathlib import Path

VAULT = Path.home() / "Documents" / "Obsidian" / "Investi" / "RedNeuro"
CONTENT = Path(__file__).resolve().parent.parent / "docs" / "content"
HUB = "Red neuronal from scratch"


def rewrite_hub_links(text):
    # [[Red neuronal from scratch]]      -> [[index|Red neuronal from scratch]]
    # [[Red neuronal from scratch#Sec]]  -> [[index#Sec|Red neuronal from scratch]]
    def repl(m):
        heading = m.group(1) or ""
        return f"[[index{heading}|{HUB}]]"
    return re.sub(rf"\[\[{re.escape(HUB)}(#[^\]|]*)?\]\]", repl, text)


def drop_leading_h1(text):
    # Las notas empiezan con "# Título" en el cuerpo. Quartz ya muestra el título de la
    # página encima del contenido, así que ese H1 saldría repetido.
    return re.sub(r"^(---\n.*?\n---\n)\s*# [^\n]+\n", r"\1", text, count=1, flags=re.S)


def block_math(text):
    # Obsidian muestra `$$f$$` en una sola línea como bloque centrado, pero el parser de
    # Quartz lo toma como matemática inline. Con los $$ en líneas propias sí es bloque.
    return re.sub(r"^\$\$(.+?)\$\$\s*$", r"$$\n\1\n$$", text, flags=re.M)


def main():
    if CONTENT.exists():
        shutil.rmtree(CONTENT)
    CONTENT.mkdir(parents=True)
    for note in sorted(VAULT.glob("*.md")):
        text = block_math(drop_leading_h1(rewrite_hub_links(note.read_text())))
        if note.stem == HUB:
            text = re.sub(r"^---\n", f"---\ntitle: {HUB}\n", text, count=1)
            dest = CONTENT / "index.md"
        else:
            dest = CONTENT / note.name
        dest.write_text(text)
        print(f"{note.name} -> {dest.relative_to(CONTENT.parent)}")


if __name__ == "__main__":
    main()
