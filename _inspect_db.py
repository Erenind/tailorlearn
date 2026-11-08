import json
import sqlite3
from pathlib import Path

c = sqlite3.connect("db/tailorlearn.sqlite3")
c.row_factory = sqlite3.Row
print("TABLES")
for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
    print(r[0])
print("---COUNTS---")
for t in [
    "material_sets",
    "documents",
    "sections",
    "units",
    "learning_datasets",
    "learning_requirements",
    "learning_materials",
    "learning_students",
    "learning_nodes",
    "learning_answers",
]:
    n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(t, n)
print("---MATERIALS---")
for r in c.execute(
    "SELECT id,name,material_type,subject,grade,stage FROM material_sets"
):
    print(dict(r))
print("---DATASETS---")
for r in c.execute("SELECT id,name,description FROM learning_datasets"):
    print(dict(r))
print("---SECTIONS---")
for r in c.execute(
    """
    SELECT s.id, d.file_name, s.title, s.level, s.page_start, s.page_end, length(s.content) AS n
    FROM sections s JOIN documents d ON d.id=s.document_id
    ORDER BY d.file_name, s.order_index
    """
):
    print(f"{r['file_name']}\tL{r['level']}\tp{r['page_start']}-{r['page_end']}\t{r['n']}\t{r['title']}")
print("---REQ SAMPLE---")
rows = c.execute("SELECT * FROM learning_requirements LIMIT 5").fetchall()
print("req count shown", len(rows))
if rows:
    print(list(rows[0].keys()))
    for r in rows:
        print(dict(r))
