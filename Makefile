env:
	/usr/bin/python3 -m venv venv
	venv/bin/python -m pip install -r requirements.txt
	make renv

renv: 
	. ./venv/bin/activate

run:
	make redis
	python replier_parser.py

freeze:
	venv/bin/python -m pip freeze

reqs:
	venv/bin/python -m pip freeze -> requirements.txt

redis:
	docker-compose down
	docker-compose up -d

make api:
	make redis
	python api.py

# docker mode
COMPOSE_FILE=docker-compose.yml 
CONTAINER_NAME=repliers-parser-api
IMAGE_NAME=repliers-parser-api

start:
	docker-compose -f $(COMPOSE_FILE) up -d --build

stop:
	docker-compose -f $(COMPOSE_FILE) stop

rm:
	docker-compose -f $(COMPOSE_FILE) down

start-interactive:
	docker-compose -f $(COMPOSE_FILE) run --rm $(CONTAINER_NAME)

logs:
	docker-compose -f $(COMPOSE_FILE) logs $(CONTAINER_NAME)

restart: stop start

rmi:
	docker rmi -f $(IMAGE_NAME)

clean: stop rm rmi