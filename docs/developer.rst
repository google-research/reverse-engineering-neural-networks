Tests and code format
---------------------
Tests are run using `pytest <https://docs.pytest.org/en/stable/>`_. From the project root directory, simply run: ``pytest`` to run the tests. You will need to have pytest installed (try `pip install pytest` to install it).

Formatting checks are doing via `yapf <https://github.com/google/yapf>`_, enabled automatically by `pre-commit <https://pre-commit.com/>`_. To get this setup, first make sure pre-commit is installed (`pip install pre-commit`) and then run ``pre-commit install`` from the project root directory.

Building the docs
-----------------
To rebuild the documentation, first install the dependencies: ``pip install -r docs/requirements.txt``

First, generate the source API documentation by running ``sphinx-apidoc -f -o docs/source renn`` from the root directory. Then, the commands to build the docs are contained in the `docs/Makefile` file.

How to contribute
-----------------

Contributor License Agreement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

Code reviews
^^^^^^^^^^^^

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
`GitHub Help <https://help.github.com/articles/about-pull-requests/>`_ for more
information on using pull requests.

Community Guidelines
^^^^^^^^^^^^^^^^^^^^

This project follows `Google's Open Source Community
Guidelines <https://opensource.google/conduct/>`_.
