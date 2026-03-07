SELECT pt.text_chunk 
FROM pharaohs_texts as pt 
INNER JOIN pharaohs as p ON pt.pharaoh_id = p.id
WHERE p.name = :pharoah_name
ORDER BY pt.text_embedding <=> :embedding 
LIMIT :limit