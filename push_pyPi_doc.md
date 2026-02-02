# Script Documentation: `push_pyPi.sh`

This bash script automates the maintenance and publication lifecycle for the `pyphyschemtools` library. It ensures synchronization between the source code, version metadata, and documentation.

---

## 🛠 Functionalities

### 1. Version Management & Synchronization
The script acts as a single source of truth for versioning across multiple files:
* **PyPI Check**: Queries the Python Package Index via `curl` and `jq` to verify the current published version.
* **Semantic Increment**: Supports `patch`, `minor`, and `major` updates.
* **Automated Updates**:
    * `pyproject.toml`: Updates the `version` key.
    * `pyphyschemtools/__init__.py`: Updates `__version__` and sets `__last_update__` to the current date.
    * `docs/source/conf.py`: Synchronizes both `version` (X.Y) and `release` (X.Y.Z).

### 2. Quality Control (Documentation Guard)
To prevent publishing broken documentation, the script includes a validation gate:
* **Pre-commit Build**: Runs `make clean && make html` within the `docs/` directory.
* **Error Interception**: If Sphinx detects any **"Unexpected indentation"** or syntax errors in docstrings, the script terminates immediately (`exit 1`) and cancels the release.

### 3. Git Workflow Automation
* **Large File Filtering**: Scans the project for files larger than 49MB and adds them to a temporary `.LargeFiles` list to update `.gitignore`, preventing push rejections on GitHub.
* **Commit & Tag**: Automates the `git add`, `git commit` (with version bump message), and `git tag` process.
* **Remote Push**: Pushes both the code and the version tags to the remote repository.

### 4. Build & Distribution
* **Clean Room Build**: Deletes previous `build/`, `dist/`, and `*.egg-info` directories to avoid artifact contamination.
* **Package Generation**: Uses `python -m build` to create standard Source and Wheel distributions.
* **PyPI Upload**: Securely uploads the content of `dist/` to PyPI using `twine`.

### 5. Environment Refresh
* **Editable Install**: Concludes by running `pip install -e .` to ensure the local development environment is synced with the newly bumped version.

---

## 🚀 Usage

```bash
# Give execution rights (one-time)
chmod +x push_pyPi.sh

# Run the script
./push_pyPi.sh

The script supports optional commit comments via command-line arguments.

**Option 1: Version only (Default)** 
If you run the script without arguments, the git commit message will only contain the version bump details.

```bash
./push_pyPi.sh
```

**Option 2: Version + Custom Comment** 
To add context to your release (e.g., "fixed spectra bug"), pass the comment as a quoted string:

```bash
./push_pyPi.sh "Add support for TDDFT gaussian summing"
```

The resulting commit will look like: 
Bump version: 1.0.1 → 1.0.2 (Add support for TDDFT gaussian summing).