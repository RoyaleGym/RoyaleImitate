# Contributing

Thanks for helping. Bug reports, fixes, examples and docs are all welcome.

## Set up

Build RoyaleLearn from source first, then install this repository into the same virtual
environment. The steps are in the guide's
[Working on RoyaleImitate itself](docs/guide.md#working-on-royaleimitate-itself).

## Before you open a pull request

    python -m pytest -q -p no:randomly -rs
    python -m ruff check royaleimitate tests

- Add a test that fails without your change and passes with it.
- A skipped test is not a passing one. If a test skips on your machine, say which and why.
- Keep public text plain: short sentences, no internal names.

## Reporting a bug

Open an issue with the template. Include the commands you ran, the full error, your OS and
Python version, and `python -c "import royaleimitate; print(royaleimitate.__version__)"`.

## Licence

By contributing you agree your work is released under this repository's [licence](LICENSE).
