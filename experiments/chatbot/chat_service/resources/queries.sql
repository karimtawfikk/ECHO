SELECT text_chunk
FROM {texts_table}
WHERE {entity_id_col} = :entity_id
ORDER BY text_embedding <=> :embedding
LIMIT 10;