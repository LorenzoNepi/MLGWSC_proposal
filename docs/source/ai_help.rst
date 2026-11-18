.. _ai_help:

AI Contributions
================

Disclaimer
----------

Some parts of the code were corrected (and very rarely written) with the help of the Google Gemini AI model.
As we all might agree, being as powerful and revolutionary as it is, it would be foolish for programmers
not to use AI to make development faster and more efficient. However, I do believe that the use of such a tool
must be well documented, as it must be clear to collaborators and users which parts of the software
were hand-written and which ones were artificially generated. That is even more true in the case of this package
since it started as a university exam project, though in my opinion it should become the standard from here on out.

AI usage
--------

In the case of `MLGWSC_proposal`, only the free models of Google Gemini (such as *Hydrogen* as of November 2026,
with limited capabilities compared to their pay-to-use counterparts) were used to speed up the development process.
None of the (few) original ideas and implementation choices of the package come from AI, which was used more
as a reviewer or sanity check. Here are the most relevant Gemini contributions:

- General revision, linting and debugging alongside tools like `ruff`, `pylint`, `pytest` and `mypy`;
- Correction of the deprecated parts of the `dain.py` module to adapt the code from [DainRepo]_ to newer PyTorch
versions;
- Suggestions about RAM and VRAM usage (e.g. using `gc.collect()` before
`torch.cuda.empty_cache()` to free the GPU VRAM );
- Cells to automatically save training and inference results on GitHub
in the `inference.ipynb` and `training.ipynb` notebooks;
- Fast docstring generation (alongside VS CODE auto-generated docstrings) for some of the less complex functions;
- Help in writing the unit tests for `torch_segmentation()` and `fore_segmentation()` and the `inference.py` module
(e.g. usage of `tmp_path` and `capsys`);
- `uv` installation to speed up `nox` sessions.