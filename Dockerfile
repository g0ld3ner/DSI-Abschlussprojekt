FROM python:3.12.3-slim

RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY crontab.txt /etc/cron.d/strom-cron
RUN chmod 0644 /etc/cron.d/strom-cron

COPY . .

RUN touch /var/log/cron.log

EXPOSE 8501

CMD ["sh", "-c", "cron && streamlit run src/strom_app.py --server.address=0.0.0.0"]