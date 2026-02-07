#!/bin/bash
#
# MAINTENANCE & PUBLICATION SCRIPT: pyphyschemtools
#
# This script automates the full release cycle:
# 1. Version Check: Compares local version vs. latest on PyPI.
# 2. Version Bump: Prompts for Patch, Minor, or Major increment.
# 3. Synchronization: Updates version and date in pyproject.toml & __init__.py.
# 4. Git Workflow: Automates Add, Commit, Tag, and Push to GitHub.
# 5. Clean & Build: Removes old artifacts and generates new Source/Wheel files.
# 6. PyPI Upload: Publishes the package to PyPI using 'twine'.
# 7. Local Refresh: Reinstalls the package in editable mode for testing.
################################################################################

# ANSI colors
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RED='\033[1;31m'
WHITE_BG_BLACK_TEXT='\033[30;47m'
RESET='\033[0m'

PYPROJECT="pyproject.toml"
DIST_DIR="dist"

project_name="pyphyschemtools"

USER_COMMENT="$1"

# nice utility
print_padded_line_wbg() {
    # Usage: print_padded_line "your message" width
    local msg="$1"
    local width="$2"
    local msg_len=${#msg}
    local pad_len=0
    if (( msg_len < width )); then
        pad_len=$((width - msg_len))
        pad=$(printf '%*s' "$pad_len" "")
        msg="$msg$pad"
    fi
    echo -e "${WHITE_BG_BLACK_TEXT}${msg}${RESET}"
}
#
# Clear separator line
SEPARATOR_RAW="---------------------------------------------------------------------------------------------"
SEPARATOR_WIDTH=${#SEPARATOR_RAW}
SEPARATOR="${WHITE_BG_BLACK_TEXT}${SEPARATOR_RAW}${RESET}"

# Validate pyproject.toml syntax before proceeding
if ! python3 -c "import tomllib; tomllib.load(open('$PYPROJECT', 'rb'))" 2>/dev/null; then
    echo -e "${RED}❌ Syntax error in $PYPROJECT detected! Please fix it before running the script.${RESET}"
    exit 1
fi

# Read current version from pyproject.toml
CURRENT_VERSION=$(grep "^version" $PYPROJECT | head -n1 | cut -d '"' -f2)
echo -e "$SEPARATOR"
print_padded_line_wbg "                    Current version in pyproject.toml: $CURRENT_VERSION" "$SEPARATOR_WIDTH"
echo -e "$SEPARATOR"
echo

# Check if any .tar.gz exists in dist/
if [ -d "$DIST_DIR" ]; then
    echo -e "$SEPARATOR"
    ARCHIVES=$(ls dist/*.tar.gz 2>/dev/null)
    if [ -z "$ARCHIVES" ]; then
        print_padded_line_wbg "${YELLOW}No tar.gz archive found.${RESET}" "$SEPARATOR_WIDTH"
    else
        print_padded_line_wbg "Archives found in dist/: $ARCHIVES" "$SEPARATOR_WIDTH"
    fi
    echo -e "$SEPARATOR"
else
    echo -e "$SEPARATOR"
    echo -e "${WHITE_BG_BLACK_TEXT}No dist/ directory.  ${RESET}"
    echo -e "$SEPARATOR"
fi
echo

# Get the latest published version on PyPI (optional)
PACKAGE_NAME=$(grep "^name" $PYPROJECT | head -n1 | cut -d '"' -f2)
echo -e "$SEPARATOR"
print_padded_line_wbg "Querying PyPI for $PACKAGE_NAME..." "$SEPARATOR_WIDTH"
echo -e "$SEPARATOR"
LATEST_PYPI=$(curl -s https://pypi.org/pypi/$PACKAGE_NAME/json | jq -r '.info.version')

if [ "$LATEST_PYPI" != "null" ]; then
    echo -e "${CYAN}Latest published version on PyPI:${RESET} ${YELLOW}$LATEST_PYPI${RESET}"
else
    echo -e "${RED}Package does not exist on PyPI (or PyPI error).${RESET}"
fi

# Ask whether to increment the version
echo -e "$SEPARATOR"
print_padded_line_wbg "Do you want to increment the $CURRENT_VERSION version? (y/n) "  "$SEPARATOR_WIDTH"
echo -e "$SEPARATOR"
read -r REPLY < /dev/tty

if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    echo -e "${CYAN}Which level? ([p]atch / [m]inor / [M]ajor)${RESET}"
    read -r LEVEL

    IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

    case $LEVEL in
        p)
            PATCH=$((PATCH + 1))
            ;;
        m)
            MINOR=$((MINOR + 1))
            PATCH=0
            ;;
        M)
            MAJOR=$((MAJOR + 1))
            MINOR=0
            PATCH=0
            ;;
        *)
            echo -e "${RED}Unknown type, version not modified.${RESET}"
            ;;
    esac

    NEW_VERSION="$MAJOR.$MINOR.$PATCH"
    echo -e "${GREEN}Updating version $CURRENT_VERSION to: $NEW_VERSION${RESET}"
    # Update pyproject.toml in place
    sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" $PYPROJECT
    echo "     - version update in  pyproject.toml   ... Done"
    # Update  $projectname/__init__.py in place
    if grep -q "^__version__ *= *" "${project_name}/__init__.py"; then
        sed -i "s/^__version__ *= *.*/__version__ = \"$NEW_VERSION\"/" "${project_name}/__init__.py"
    else
        echo "__version__ = \"$NEW_VERSION\"" >> "${project_name}/__init__.py"
    fi
    echo "     - __version__ variable updated in project_name/__init__.py  ... Done"
    # Update __last_update__ field automatically
    today=$(date +%Y-%m-%d)
    if grep -q "^__last_update__ *= *" "${project_name}/__init__.py"; then
        sed -i "s/^__last_update__ *= *.*/__last_update__ = \"$today\"/" "${project_name}/__init__.py"
    else
        echo "__last_update__ = \"$today\"" >> "${project_name}/__init__.py"
    fi
    # Update version in docs/source/conf.py
    CONF_PY="docs/source/conf.py"
    if [ -f "$CONF_PY" ]; then
        # On extrait X.Y pour le paramètre 'version' de Sphinx
        VERSION_XY=$(echo "$NEW_VERSION" | cut -d'.' -f1,2)
        
        # Mise à jour de 'version' (ex: 1.2)
        if grep -q "^version =" "$CONF_PY"; then
            sed -i "s/^version *= *['\"].*['\"]/version = '$VERSION_XY'/" "$CONF_PY"
        else
            sed -i "/^project *=/a version = '$VERSION_XY'" "$CONF_PY"
        fi
        # 2. Update 'release' (supports both ' and " quotes)
        sed -i "s/^release *= *['\"].*['\"]/release = '$NEW_VERSION'/" "$CONF_PY"
            echo "     - version & release updated in docs/source/conf.py ... Done"
    else
        echo -e "${YELLOW}     - Warning: docs/source/conf.py not found, skipping documentation update.${RESET}"
    fi
    # --- DOC VALIDATION ---
    echo -e "${CYAN}Checking documentation health before commit...${RESET}"
    (cd docs && make clean && make html > /dev/null 2>&1)
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Sphinx build failed! Please fix docstrings before pushing.${RESET}"
        exit 1
    fi
    echo "     - Documentation build ... OK"

    # --- GIT SECTION ---
    echo -e "$SEPARATOR"
    print_padded_line_wbg "Git commit and tag...  " "$SEPARATOR_WIDTH"
    echo -e "$SEPARATOR"
    echo

    # Detect large files (> 49MB) before publishing
    find . -size +49M -print | sed 's|^\./||' > .LargeFiles
    cat .gitignore_base .LargeFiles > .gitignore
    git add -A
    print_padded_line_wbg "Git status before commit:" "$SEPARATOR_WIDTH"
    git status
    echo "Proceed with commit? (y/n)"
    read -r CONFIRM
    if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
        # Prepare the commit message
        COMMIT_MSG="Bump version: $CURRENT_VERSION → $NEW_VERSION"
        
        if [ -n "$USER_COMMENT" ]; then
            # If an argument was provided, append it
            COMMIT_MSG="$COMMIT_MSG ($USER_COMMENT)"
        else
            # Inform the user that no extra comment is being added
            echo -e "${YELLOW}     - No extra comment provided. Using default version message.${RESET}"
        fi

	echo -e "${RED}Commit message: $COMMIT_MSG${RESET}"
        git commit -m "$COMMIT_MSG"
        git tag "v$NEW_VERSION"
        git push
        git push --tags
    else
        echo -e "${RED}Commit cancelled. Reverting version numbers only...${RESET}"
	sed -i "s/^version = \"$NEW_VERSION\"/version = \"$CURRENT_VERSION\"/" "$PYPROJECT"
	echo "     - pyproject.toml version number reverted"
	sed -i "s/^__version__ = \"$NEW_VERSION\"/__version__ = \"$CURRENT_VERSION\"/" "${project_name}/__init__.py"
	echo "     - pyphyschemtools/__init__.py version number reverted"
	if [ -f "docs/source/conf.py" ]; then
            VERSION_XY_OLD=$(echo "$CURRENT_VERSION" | cut -d'.' -f1,2)
	    sed -i "s/^version = '$VERSION_XY'/version = '$VERSION_XY_OLD'/" "docs/source/conf.py"
            sed -i "s/^release = '$NEW_VERSION'/release = '$CURRENT_VERSION'/" "docs/source/conf.py"
	    echo "     - docs/source/conf.py version number reverted"
	fi
	echo -e "${GREEN}Rollback complete. Version numbers are back to $CURRENT_VERSION.${RESET}"
        echo -e "${GREEN}All your other changes (dependencies, README, notebooks) are safe.${RESET}"
        exit 1 
    fi
    echo

    echo -e "$SEPARATOR"
    print_padded_line_wbg "Removing old builds: rm -rf build dist *.egg-info" "$SEPARATOR_WIDTH"
    echo -e "$SEPARATOR"
    rm -rf build dist "${project_name}.egg-info"
    echo

    # Build the package
    echo -e "$SEPARATOR"
    print_padded_line_wbg "Building the package: python -m build" "$SEPARATOR_WIDTH"
    echo -e "$SEPARATOR"
    python -m build
    echo

    # Upload to PyPI
    echo -e "$SEPARATOR"
    print_padded_line_wbg "Uploading to PyPI: twine upload dist/*" "$SEPARATOR_WIDTH"
    echo -e "$SEPARATOR"
    twine upload dist/*
    echo

    # Reinstall in editable mode
    echo -e "$SEPARATOR"
    print_padded_line_wbg "Reinstalling in editable mode: pip install -e ." "$SEPARATOR_WIDTH"
    echo -e "$SEPARATOR"
    pip install -e .
    echo

    echo -e "$SEPARATOR"
    echo -e "${GREEN}🎉 Process completed!${RESET}"
    echo -e "$SEPARATOR"
else
    echo -e "${YELLOW}Version $CURRENT_VERSION kept unchanged.${RESET}"
fi

