FROM python:3.6-alpine

RUN adduser -D irishSignLanguage

WORKDIR /home/irishSignLanguage

COPY requirements.txt requirements.txt
RUN python -m venv venv
RUN venv/bin/pip install -r requirements.txt
RUN venv/bin/pip install gunicorn

COPY app app
COPY migrations migrations
COPY irishSignLanguage.py config.py boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP irishSignLanguage.py

RUN chown -R irishSignLanguage:irishSignLanguage ./
USER irishSignLanguage

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]
