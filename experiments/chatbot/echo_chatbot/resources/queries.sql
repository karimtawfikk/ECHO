WITH RankedTexts AS (
    SELECT 
        pt.text_chunk,
        pt.text_embedding <=> :embedding AS distance,
        COUNT(*) OVER() as total_count
    FROM pharaohs_texts AS pt 
    INNER JOIN pharaohs AS p ON pt.pharaoh_id = p.id
    WHERE p.name = :pharoah_name
)
SELECT text_chunk
FROM RankedTexts
WHERE true
ORDER BY distance
-- This handles the "min(10, total)" logic:
LIMIT LEAST(10, (SELECT COUNT(*) FROM pharaohs_texts pt2 INNER JOIN pharaohs p2 ON pt2.pharaoh_id = p2.id WHERE p2.name = :pharoah_name));