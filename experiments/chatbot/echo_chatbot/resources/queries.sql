SELECT text_chunk 
FROM pharaohs_texts 
WHERE pharaoh_id = :p_id 
ORDER BY text_embedding <=> :embedding 
LIMIT :limit