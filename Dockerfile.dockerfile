# Dockerfile

FROM python:3.10-slim

# Install basic OS dependencies
RUN apt-get update && apt-get install -y build-essential

# Set working directory
WORKDIR /app

# Copy requirement list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY src/ ./src
COPY data/ ./data
COPY results/ ./results

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit dashboard by default
CMD ["streamlit", "run", "src/dashboard.py"]
