# Using a small Python image so Docker isn't too big
FROM python:3.10-slim

# Installing basic build tools (needed for some pip packages like pandas etc)
RUN apt-get update && apt-get install -y build-essential

# This is the folder where everything will go inside the container
WORKDIR /app

# Copying my list of libraries and installing them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now I add the code and data, putting it all into /app
COPY src/ ./src
COPY data/ ./data
COPY results/ ./results

# For when I want to run Streamlit later
EXPOSE 8501

# By default this runs my CLI tool (see src/cli.py)
# Can override this to run Streamlit with --entrypoint
ENTRYPOINT ["python", "-m", "src.cli"]


