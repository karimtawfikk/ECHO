# E.C.H.O. Backend (Neural Gate API)

This backend serves the real Keras AI models for E.C.H.O. It performs hierarchical inference (Binary -> Specialized) and eagerly hydrates metadata and related images/scripts from the exact matching PostgreSQL database records.

## Requirements

- Python 3.9+
- PostgreSQL database (`echo_db`)

## Setup Instructions

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   Ensure your `.env` file exists in the root directory (same level as `app/`) containing at least:
   ```env
   DATABASE_URL=postgresql://postgres:user@localhost:5432/echo_db
   ENVIRONMENT=development
   ```

4. **Run the API**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Verify AI Models**
   The startup logs should confirm that the Neural Gate is active. It expects the following files inside the `models_weights/` directory:
   - `binary_classifier.keras`
   - `pharaoh_identifier.keras`
   - `landmark_identifier.keras`
   - `binary_label_encoder.pkl`
   - `pharaoh_label_encoder.pkl`
   - `landmark_label_encoder.pkl`

## API Documentation

- **Interactive Docs:** `http://127.0.0.1:8000/docs`
- **Main Engine:** `POST /api/v1/recognize` (Multipart/form-data: `file=@image.jpg`)
- **Health:** Checks DB connectivity `GET /api/v1/health/db`
- **Debug:** `GET /api/v1/debug/pharaohs`
