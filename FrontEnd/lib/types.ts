// ─── API Types for E.C.H.O Recognition Flow ───────────────────────────────

export interface EntityImage {
    id: number;
    url: string | null;
    caption: string | null;
}

export interface RecognitionEntity {
    id: number;
    name: string;
    description: string | null;
    type: string | null;
    dynasty: string | null;
    period: string | null;
    location: string | null;
    images: EntityImage[];
    scripts: Record<string, unknown> | null;
}

export interface RecognitionResult {
    source: string;
    type: "pharaoh" | "landmark" | "error" | string;
    name: string;          // raw model label e.g. "Ramesses_II"
    category: string | null;
    confidence: number;
    binary_confidence: number;
    entity: RecognitionEntity | null;
    debug_info: Record<string, unknown> | null;
}

// Session storage payload — stored between Upload → Result
export interface PendingResult {
    result: RecognitionResult;
    imageDataUrl: string | null; // base64 DataURL of the uploaded image (kept small for preview only)
}
