env:
	/usr/bin/python3 -m venv venv
	venv/bin/python -m pip install -r requirements.txt
	make renv

renv: 
	. ./venv/bin/activate

run:
	python replier_parser.py

freeze:
	venv/bin/python -m pip freeze

reqs:
	venv/bin/python -m pip freeze -> requirements.txt

redis:
	docker-compose up -d