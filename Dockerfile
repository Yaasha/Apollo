FROM python:3.8

RUN apt install git

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
ENTRYPOINT ["python3", "-u", "/app/apollo.py"]
CMD ["run"]
