# 
FROM python:3.11-bullseye

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y gcc python3-dev
RUN apt-get install -y python3-psutil
RUN pip install numpy
RUN pip install fastapi uvicorn

# 
WORKDIR /app

# 
COPY requirements_docker.txt .

# 
RUN pip install -r requirements_docker.txt

# 
COPY . .

EXPOSE 8099

# 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8099"]