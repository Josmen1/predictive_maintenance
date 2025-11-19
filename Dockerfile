FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements & setup first (better caching)
COPY requirements.txt setup.py ./

RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project
COPY . .

# Install your project as a package via setup.py
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
