# base image — official Python 3.11 slim version
FROM python:3.11-slim

# set working directory inside container
WORKDIR /app

# copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of the project
COPY models/ ./models/
COPY api/ ./api/
COPY src/ ./src/

# expose port 5000
EXPOSE 5000

# command to run when container starts
CMD ["python", "api/app.py"]