FROM ubuntu:20.04
RUN apt-get update && apt-get install \
  -y --no-install-recommends python3 python3-virtualenv

COPY . /app

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN ls -la
# Install dependencies:
RUN pip install -r requirements.txt

RUN python -m spacy download en
# Run the application:
CMD ["python", "app.py"]