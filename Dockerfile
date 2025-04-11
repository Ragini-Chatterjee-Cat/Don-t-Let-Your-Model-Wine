# using a small python image so docker isnt too big
FROM python:3.10-slim

# installing basic build tools (needed for some pip packages like pandas etc)
RUN apt-get update && apt-get install -y build-essential

# This is the folder where everything will go inside the container
WORKDIR /app

# copying my list of libaries and installing them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# now I add the code and data, putting it all into /app
COPY src/ ./src
COPY data/ ./data
COPY results/ ./results

# for when I wanna run streamlit later
EXPOSE 8501

# By default this runs my cli tool (see src/cli.py)
# Can override this to run streamlit with --entrypoint
ENTRYPOINT ["python", "-m", "src.cli"]


