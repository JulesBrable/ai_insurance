# app/Dockerfile

FROM python:3.9.16

WORKDIR ${HOME}/ai_insurance

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5010

HEALTHCHECK CMD curl --fail http://localhost:5010/_stcore/health

ENTRYPOINT ["bash", "-c", "./run.sh"]
