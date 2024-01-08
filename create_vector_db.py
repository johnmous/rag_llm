import ray
import matplotlib.pyplot as plt
from pathlib import Path
from rag.data import extract_sections


# Ray dataset
EFS_DIR = "/home/imoustakas/llm_playground/RAG/django_docs/"
DOCS_DIR = Path(EFS_DIR, "docs.djangoproject.com/en/5.0/")
ds = ray.data.from_items([{"path": path} for path in DOCS_DIR.rglob("*.html")
if not path.is_dir()])
print(f"{ds.count()} documents")

sample_html_fp = Path(EFS_DIR, "docs.djangoproject.com/en/5.0/faq/contributing/index.html")
# print(sample_html_fp)
print(extract_sections({"path": sample_html_fp})[4])

# Extract sections
sections_ds = ds.flat_map(extract_sections)
print(sections_ds.count())

section_lengths = []
for section in sections_ds.take_all():
    section_lengths.append(len(section["text"]))

# Plot
plt.figure(figsize=(12, 3))
plt.plot(section_lengths, marker='o')
plt.title("Section lengths")
plt.ylabel("# chars")
plt.savefig("section_lengths.png")
