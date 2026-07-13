import fitz  # PyMuPDF
from pathlib import Path

pdf_path = Path("Sources\\blockchain\\T1087.pdf")


doc = fitz.open(pdf_path)

print("=" * 60)
print("PYMUPDF - EXTRACTION COMPLÈTE")
print("=" * 60)
print(f"Nombre de pages: {len(doc)}")

# === Méthode 1 : Texte brut simple ===
print("\n--- Méthode 1: get_text() ---")
full_text = ""
for page_num, page in enumerate(doc, 1):
    text = page.get_text()
    full_text += f"\n\n--- Page {page_num} ---\n{text}"
    
print(f"Longueur totale: {len(full_text)} caractères")
print(full_text[:2000])

# === Méthode 2 : Texte avec structure (blocks) ===
print("\n" + "=" * 60)
print("MÉTHODE 2: Extraction par blocs")
print("=" * 60)

for page_num, page in enumerate(doc, 1):
    blocks = page.get_text("blocks")
    print(f"\n--- Page {page_num} - {len(blocks)} blocs ---")
    for block in blocks:
        x0, y0, x1, y1, text, block_no, block_type = block
        if text.strip():
            print(f"[Bloc {block_no} | type={block_type}] {text[:200]}...")

# === Méthode 3 : Dictionnaire détaillé ===
print("\n" + "=" * 60)
print("MÉTHODE 3: Dictionnaire détaillé")
print("=" * 60)

for page_num, page in enumerate(doc, 1):
    dict_data = page.get_text("dict")
    print(f"\n--- Page {page_num} ---")
    for block in dict_data["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"]
                    font = span["font"]
                    size = span["size"]
                    if text.strip():
                        print(f"[{font} {size:.1f}pt] {text[:100]}")  # ✅ CORRIGÉ

# === Méthode 4 : HTML ===
print("\n" + "=" * 60)
print("MÉTHODE 4: Export HTML")
print("=" * 60)

html_output = ""
for page in doc:
    html_output += page.get_text("html")
print(html_output[:2000])

# === Sauvegarde ===
output_dir = Path("output_pymupdf")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "texte_complet.txt", "w", encoding="utf-8") as f:
    f.write(full_text)
print(f"\n✅ Texte sauvegardé dans: {output_dir / 'texte_complet.txt'}")

doc.close()