# Contributing Guidelines

To keep this code vault clean, organized, and optimized, please follow these guidelines when adding new notebooks or scripts:

## 📓 Jupyter Notebooks (`.ipynb`)
1. **Clear Outputs:** Before committing a notebook, clear all cell outputs to reduce file size (under `Cell > All Output > Clear` in Jupyter). This keeps git diffs clean and avoids committing unnecessary cache/images.
2. **Naming Conventions:** Use clear snake_case or DescriptiveNames. E.g., `model_training.ipynb` instead of `Untitled.ipynb`.
3. **Weekly Coursework:** Save weekly notebooks inside dedicated course folders or prefix them with `week_` to maintain index scanning order.

## 🐍 Python Scripts (`.py`)
1. **Clean Code Structure:** Ensure all utilities, vision models, or UI scripts contain a docstring explaining their usage.
2. **Environment Independent:** Do not hardcode absolute paths. Always use relative paths or environment variables for reading datasets.
3. **No Large Assets:** Do not commit raw datasets, video files, or checkpoints. Ensure they are listed in `.gitignore`.
