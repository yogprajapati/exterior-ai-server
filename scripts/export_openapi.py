"""Write the API's OpenAPI schema to a file (used to generate the client's TypeScript types)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app

Path(sys.argv[1]).write_text(json.dumps(app.openapi(), indent=1))
