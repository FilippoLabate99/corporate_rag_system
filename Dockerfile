# Usa una versione leggera e ufficiale di Python
FROM python:3.9-slim

# Imposta la cartella di lavoro dentro il container
WORKDIR /app

# Installa alcuni strumenti di base del sistema operativo necessari per compilare le librerie
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia solo il file dei requirements per ottimizzare la cache di Docker
COPY requirements.txt .

# Installa le dipendenze Python senza salvare la cache per tenere l'immagine leggera
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il resto del codice nella cartella di lavoro
COPY . .

# Dichiara la porta su cui comunicherà Streamlit
EXPOSE 8501

# Il comando magico per far partire l'applicazione
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]