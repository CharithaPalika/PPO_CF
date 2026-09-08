#!/bin/bash
# Convert this project's own text files from Windows CRLF to Unix LF.
#
#   cd ~/ppo_cf && bash slurm/fix_line_endings.sh
#
# WHY. Copying from Windows brings \r\n with it. sbatch refuses such a script
# outright ("Batch script contains DOS line breaks"), which is the friendly case.
# The unfriendly case is a SOURCED file -- _prelude.sh, _submit_lib.sh, env.sh --
# where bash does not refuse anything: the \r becomes part of the last word on
# every line, so `set -euo pipefail\r` and `VENV="...".../bin/activate\r` fail
# somewhere that looks unrelated to line endings.
#
# ON THE LOGIN NODE ON PURPOSE. This is `sed` over ~60 small files -- the same
# order of work as `ls`, and a prerequisite for being able to submit anything at
# all. It is a control-plane action, like sbatch itself.
#
# NOT `find . -type f -exec dos2unix {} +`: that walks .venv/ (thousands of
# installed package files) and runs/ (hundreds of MB of results).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

mapfile -d '' FILES < <(
    find . \
        \( -path ./.venv -o -path ./.git -o -path ./runs -o -path ./figures \
           -o -path ./artifacts -o -path ./to_delete -o -name __pycache__ \) -prune -o \
        -type f \( -name '*.sh' -o -name '*.sbatch' -o -name '*.py' -o -name '*.yaml' \
                   -o -name '*.yml' -o -name '*.md' -o -name '*.txt' -o -name '*.cfg' \
                   -o -name '*.ipynb' -o -name '.gitignore' -o -name '.gitattributes' \) \
        -print0
)

if [ ${#FILES[@]} -eq 0 ]; then
    echo "no text files found -- are you in the project root?" >&2
    exit 1
fi

# Only touch files that actually contain a CR, so mtimes stay meaningful.
changed=0
for f in "${FILES[@]}"; do
    if grep -qU $'\r' "$f" 2>/dev/null; then
        if command -v dos2unix >/dev/null; then
            dos2unix -q "$f"
        else
            sed -i 's/\r$//' "$f"
        fi
        echo "  fixed $f"
        changed=$((changed + 1))
    fi
done

echo "${#FILES[@]} text files checked, $changed converted to LF"
[ "$changed" -eq 0 ] && echo "(already clean)"
exit 0
