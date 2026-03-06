set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python -m scripts.rec_zoo.runner "$@"
