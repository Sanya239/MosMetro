FROM ubuntu:latest
LABEL authors="sanya"

# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальное (включая данные)
COPY . .

# Открываем порт (если Dash-приложение)
EXPOSE 8239

# Указываем команду запуска
CMD ["python","-u", "main.py"]
