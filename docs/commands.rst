Commands
========

Common project commands:

Setup and training
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv sync
   uv run python -m src.models.train_model

App runtime
^^^^^^^^^^^

.. code-block:: bash

   uv run uvicorn app.main:app --reload

Notebook and evaluation
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   uv run jupyter notebook
   uv run python -m src.models.evaluate_model
