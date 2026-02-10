# Usar Ubuntu 22.04 con CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Establecer variables de entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Establecer el directorio de trabajo
WORKDIR /app

# 1. Instalar Python 3.10 y dependencias básicas (NO necesitas PPA)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    python3-distutils \
    curl \
    wget \
    git \
    build-essential \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Configurar Python 3.10 como predeterminado
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# 3. Actualizar pip y herramientas Python
RUN python3 -m pip install --upgrade pip setuptools wheel

# 4. Instalar dependencias del sistema adicionales
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    liblzma-dev \
    tk-dev \
    xz-utils \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 5. Verificar instalación
RUN python3 --version && \
    python --version && \
    pip --version && \
    python3 -c "import ssl; print(f'SSL: {ssl.OPENSSL_VERSION}'); import setuptools; print('Setuptools OK')"