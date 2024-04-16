# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim

EXPOSE 8000

WORKDIR /app
COPY . /app

COPY requirements.txt .
# RUN python -m pip install -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt


CMD ["python", "manage.py", "runserver"]