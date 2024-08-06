# FROM tensorflow/tensorflow:2.17.0
FROM  tensorflow/tensorflow:2.17.0-gpu
WORKDIR /app

# Copy the requirements file
# COPY requirements.txt .

# Install the required Python packages
# RUN pip install torch
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN   pip3 install --ignore-installed blinker
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip nginx supervisor
COPY ./source /app/source
COPY ./damo1 /app/damo1
COPY ./input.png /app/
COPY ./main.py /app/
COPY ./cartoonifymod.py /app/
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/




EXPOSE 80

# Define environment variable
# ENV FLASK_APP=app.py

# Command to run the Flask app
# CMD ["python", "./main.py"]
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

