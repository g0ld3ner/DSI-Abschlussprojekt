from datetime import datetime

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open("/app/logs/python_cron_test.log", "a") as f:
    f.write(f"Cron-Testlauf um {now}\n")