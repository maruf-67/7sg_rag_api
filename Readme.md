# 7SG Q&A API with Web Interface

A Flask-based application for managing a 7SG vector database via a web page and answering questions via API using PostgreSQL with `pgvector`.

## Setup

1. **Activate Virtual Environment**:
   ```bash
   source 7sg_rag/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up PostgreSQL**:
   - Ensure PostgreSQL is running and the `7sg` database with `pgvector` exists:
     ```bash
     psql -h localhost -p 5432 -U your_username -d 7sg -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
     ```

4. **Configure Environment**:
   - Edit `.env`:
     ```
     POSTGRES_DSN=postgresql+psycopg2://your_username:your_password@localhost:5432/7sg
     GOOGLE_API_KEY=your_actual_api_key
     ```

5. **Create Templates Directory**:
   ```bash
   mkdir templates
   ```

6. **Run the App**:
   ```bash
   python app.py
   ```

## Usage

- **Web Interface**: Visit `http://localhost:5000`
  - Upload PDFs to update the vector store.
  - Check "Clear existing data" to reset the collection.

- **API Endpoints**:
  - **GET /api/status**:
    ```bash
    curl http://localhost:5000/api/status
    ```
  - **POST /api/ask**:
    ```bash
    curl -X POST -F "question=What is the 7SG framework?" http://localhost:5000/api/ask
    ```

## Notes

- The web interface manages the vector store; the API only answers questions.
- Ensure the existing `7sg` database is accessible.
- Replace `your_username`, `your_password`, and `your_actual_api_key` in `.env`.