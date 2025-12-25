# Service Management Guide

This guide explains how to start, stop, and manage all services required for the RAG system.

## Overview

The RAG system requires three services:

1. **PostgreSQL** (with pgvector) - Primary vector database
2. **Qdrant** - Alternative vector database
3. **Ollama** - Local LLM server

## Quick Start

### Windows

**Start all services:**
```batch
start_all_services.bat
```

**Stop all services:**
```batch
stop_all_services.bat
```

### Linux/Mac

**Start all services:**
```bash
chmod +x *.sh  # First time only
./start_all_services.sh
```

**Stop all services:**
```bash
./stop_all_services.sh
```

## Individual Services

### PostgreSQL

**What it does:**
- Primary vector database with pgvector extension
- Stores module, file, and task embeddings
- Supports 384-dimensional vectors (BAAI/bge-small-en-v1.5)
- Uses HNSW index for fast similarity search

**Windows:**
```batch
start_postgres.bat
```

**Linux/Mac:**
```bash
./start_postgres.sh
```

**Connection details:**
- Host: `localhost`
- Port: `5432`
- Database: `semantic_vectors`
- User: `postgres`
- Password: `postgres`
- Connection string: `postgresql://postgres:postgres@localhost:5432/semantic_vectors`

**Verify it's running:**
```bash
podman ps | grep semantic_vectors_db
```

**Access the database:**
```bash
podman exec -it semantic_vectors_db psql -U postgres -d semantic_vectors
```

### Qdrant

**What it does:**
- Alternative vector database
- Similar functionality to PostgreSQL
- Has web dashboard for visualization
- May have stability issues (why we added PostgreSQL)

**Windows:**
```batch
start_qdrant.bat
```

**Linux/Mac:**
```bash
./start_qdrant.sh
```

**Access points:**
- Dashboard: http://localhost:6333/dashboard
- API: http://localhost:6333

**Verify it's running:**
```bash
podman ps | grep qdrant
```

### Ollama

**What it does:**
- Local LLM server
- Runs models like qwen2.5-coder, codellama
- Provides AI recommendations in RAG system
- GPU acceleration if available

**Windows:**
```batch
start_ollama.bat
```

**Linux/Mac:**
```bash
./start_ollama.sh
```

**Access:**
- API: http://localhost:11434

**Default model:**
- `qwen2.5-coder:latest` (1.9 GB)
- Auto-downloaded on first start

**Verify it's running:**
```bash
curl http://localhost:11434/api/tags
```

**List available models:**
```bash
ollama list
```

**Pull additional models:**
```bash
ollama pull codellama
ollama pull deepseek-coder
```

## Service Status Checks

### Check all containers

**Windows:**
```batch
podman ps
```

**Linux/Mac:**
```bash
podman ps
```

**Expected output:**
```
CONTAINER ID  IMAGE                       PORTS                   NAMES
abc123...     pgvector/pgvector:pg16      0.0.0.0:5432->5432/tcp  semantic_vectors_db
def456...     qdrant/qdrant:latest        0.0.0.0:6333->6333/tcp  qdrant
```

### Check Ollama

```bash
ollama list
```

**Expected output:**
```
NAME                    ID              SIZE      MODIFIED
qwen2.5-coder:latest    abc123...       1.9 GB    2 minutes ago
```

## Troubleshooting

### PostgreSQL won't start

**Problem:** Port 5432 already in use

**Solution:**
```bash
# Check what's using the port
netstat -ano | findstr :5432    # Windows
lsof -i :5432                   # Linux/Mac

# Stop the conflicting service or change the port in postgres-compose.yml
```

**Problem:** Container exists but stopped

**Solution:**
```bash
podman start semantic_vectors_db
```

### Qdrant won't start

**Problem:** Port 6333 already in use

**Solution:**
```bash
# Check what's using the port
netstat -ano | findstr :6333    # Windows
lsof -i :6333                   # Linux/Mac

# Change the port in qdrant-compose.yml
```

### Ollama won't start

**Problem:** Ollama not installed

**Solution:**
1. Download from https://ollama.ai
2. Install and restart terminal
3. Run `ollama serve`

**Problem:** Model download fails

**Solution:**
```bash
# Try manual download
ollama pull qwen2.5-coder

# Check disk space (model is 1.9 GB)
```

**Problem:** Server doesn't respond

**Solution:**
```bash
# Kill existing process
taskkill /F /IM ollama.exe      # Windows
pkill ollama                     # Linux/Mac

# Restart
ollama serve
```

## Complete Workflow

### First Time Setup

**1. Start services:**
```batch
# Windows
start_all_services.bat

# Linux/Mac
./start_all_services.sh
```

**2. Wait for services (total ~1 minute):**
- PostgreSQL: 10 seconds
- Qdrant: 10 seconds
- Ollama: 30 seconds
- Model download (first time only): 2-3 minutes

**3. Verify all services:**
```bash
# Check containers
podman ps

# Check Ollama
ollama list

# Test PostgreSQL
podman exec -it semantic_vectors_db psql -U postgres -c "SELECT version();"

# Test Qdrant
curl http://localhost:6333/dashboard
```

**4. Run ETL:**
```batch
# Quick test (5-8 minutes)
cd exp3
run_etl_test_postgres.bat       # Windows
./run_etl_test_postgres.sh      # Linux/Mac

# Full production (60-70 minutes)
cd exp3
run_etl_postgres.bat            # Windows
./run_etl_postgres.sh           # Linux/Mac
```

**5. Launch UI:**
```bash
cd ragmcp
python gradio_ui.py
```

**6. Open browser:**
```
http://localhost:7860
```

### Daily Usage

**Start services:**
```batch
start_all_services.bat    # Windows
./start_all_services.sh   # Linux/Mac
```

**Launch UI:**
```bash
cd ragmcp
python gradio_ui.py
```

**Stop when done:**
```batch
stop_all_services.bat     # Windows
./stop_all_services.sh    # Linux/Mac
```

## Service Management Best Practices

### 1. Always start services before ETL or UI

```bash
# Wrong order
cd ragmcp
python gradio_ui.py  # Will fail - no database

# Right order
start_all_services.bat
cd ragmcp
python gradio_ui.py
```

### 2. Use stop script before shutdown

```bash
# Clean shutdown
stop_all_services.bat

# Avoids data corruption
# Releases ports properly
```

### 3. Check service status before troubleshooting

```bash
# Before reporting issues
podman ps          # Check containers
ollama list        # Check Ollama
curl http://localhost:11434/api/tags  # Test Ollama API
```

### 4. Monitor logs for errors

```bash
# PostgreSQL logs
podman logs semantic_vectors_db

# Qdrant logs
podman logs qdrant

# Ollama logs (if using start script)
cat ollama.log  # Linux/Mac
```

## Port Reference

| Service    | Port  | Protocol | URL                                    |
|------------|-------|----------|----------------------------------------|
| PostgreSQL | 5432  | TCP      | postgresql://localhost:5432            |
| Qdrant     | 6333  | HTTP     | http://localhost:6333                  |
| Ollama     | 11434 | HTTP     | http://localhost:11434                 |
| Gradio UI  | 7860  | HTTP     | http://localhost:7860                  |

## Docker Compose Files

All compose files are in the `exp3/` directory:

- `postgres-compose.yml` - PostgreSQL with pgvector
- `qdrant-compose.yml` - Qdrant vector database

## Next Steps

After starting services:

1. **Quick Test (10 minutes):**
   - See `QUICK_TEST_GUIDE.md`
   - Uses w1000 (last 1000 tasks)
   - Perfect for development

2. **Full Production (60-70 minutes):**
   - See `README.md` or `QUICKSTART_RAG.md`
   - Processes all ~9,799 tasks
   - Maximum accuracy

3. **Customize:**
   - Add more Ollama models
   - Adjust database settings
   - Configure LLM parameters

## Related Documentation

- `QUICK_TEST_GUIDE.md` - Fast testing in 10 minutes
- `QUICKSTART_RAG.md` - Complete RAG system guide
- `POSTGRES_SETUP.md` - PostgreSQL detailed setup
- `ragmcp/RAG_SYSTEM.md` - RAG architecture

---

**Service Management Made Easy! 🚀**
