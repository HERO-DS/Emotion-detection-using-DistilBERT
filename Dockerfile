FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
