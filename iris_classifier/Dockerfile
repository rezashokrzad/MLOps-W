FROM python:3.11

WORKDIR /code

COPY . /code

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt 

EXPOSE 5000

CMD ["python", "main.py"]
