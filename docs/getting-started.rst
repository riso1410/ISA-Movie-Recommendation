Getting started
===============

Install dependencies with:

.. code-block:: bash

   uv sync

Download the Kaggle dataset into ``data/raw/`` with these files:

- ``movies_metadata.csv``
- ``credits.csv``
- ``keywords.csv``
- ``links_small.csv``

Generate recommender models before starting the app:

.. code-block:: bash

   uv run python -m src.models.train_model

Run the web app:

.. code-block:: bash

   uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

The FastAPI app loads prebuilt models from ``models/`` at startup. It does not
train models during startup or while serving requests.
