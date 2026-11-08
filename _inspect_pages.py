import sqlite3

c = sqlite3.connect("db/tailorlearn.sqlite3")
c.row_factory = sqlite3.Row
print("page counts")
for r in c.execute(
    """
    SELECT d.file_name, COUNT(*) n
    FROM pages p JOIN documents d ON d.id=p.document_id
    GROUP BY d.file_name
    """
):
    print(dict(r))

print("\n=== STANDARD pages 54-66 first 400 chars ===")
doc = c.execute(
    "SELECT id FROM documents WHERE file_name LIKE '%课程标准%'"
).fetchone()["id"]
for r in c.execute(
    "SELECT page_number, substr(final_text,1,500) text FROM pages WHERE document_id=? AND page_number BETWEEN 54 AND 66 ORDER BY page_number",
    (doc,),
):
    print("\n--- page", r["page_number"], "---")
    print(r["text"])
