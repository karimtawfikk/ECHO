WITH RankedTexts AS (
    SELECT 
        pt.text_chunk,
        pt.text_embedding <=> :embedding AS distance,
        COUNT(*) OVER() AS total_count
    FROM {texts_table} AS pt
    INNER JOIN {entities_table} AS p ON pt.{entity_id_col} = p.id
    WHERE p.name = :entity_name
)
SELECT text_chunk
FROM RankedTexts
WHERE true
ORDER BY distance

LIMIT LEAST(
    3,
    (
        SELECT COUNT(*) 
        FROM {texts_table} t2
        INNER JOIN {entities_table} e2 ON t2.{entity_id_col} = e2.id
        WHERE e2.name = :entity_name
    )
);