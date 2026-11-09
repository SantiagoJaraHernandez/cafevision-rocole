#!/usr/bin/env bash
set -o errexit
pip install --upgrade pip
pip install -r requirements.txt
python descargar_modelo.py
python manage.py collectstatic --no-input
