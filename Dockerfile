# Usamos Python 3.7 slim para que la imagen no pese gigas innecesarios
FROM python:3.7-slim

# Instalamos dependencias del sistema necesarias para compilar librerías como Cython o Scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    &> /dev/null && rm -rf /var/lib/apt/lists/*

# Definimos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos SOLO el archivo de requisitos primero
# Esto aprovecha la caché de Docker y no reinstala todo cada vez que cambies tu código
COPY requirements.txt .

# Instalamos las librerías de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# NOTA: No hacemos "COPY . ." porque usaremos VOLÚMENES para el código.