env:
	python -m venv venv
	venv/bin/python -m pip install -r requirements.txt
	make renv

renv: 
	. ./venv/bin/activate

run:
	venv/bin/python replier_parser.py

freeze:
	venv/bin/python -m pip freeze

reqs:
	venv/bin/python -m pip freeze -> requirements.txt

docker:
	docker-compose up -d