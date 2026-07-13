from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from pathlib import Path

pdf_path = Path("Sources\\blockchain\\T1087.pdf")

converter = DocumentConverter()
result = converter.convert(pdf_path)

# Export avec images et tableaux
print("=== Export Markdown avec images ===")
md = result.document.export_to_markdown(image_mode="referenced")
print(md)

# Sauvegarder le résultat
output_dir = Path("output_docling")
output_dir.mkdir(exist_ok=True)

# Sauvegarder le texte
with open(output_dir / "texte.txt", "w", encoding="utf-8") as f:
    f.write(result.document.export_to_text())

# Sauvegarder le markdown
with open(output_dir / "document.md", "w", encoding="utf-8") as f:
    f.write(result.document.export_to_markdown())

print(f"\nRésultats sauvegardés dans: {output_dir.absolute()}")