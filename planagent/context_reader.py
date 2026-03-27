import ast
import json
import hashlib
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STACK_SIGNALS = {
    "requirements.txt": "Python",
    "pyproject.toml": "Python",
    "setup.py": "Python",
    "package.json": "JavaScript/Node",
    "go.mod": "Go",
    "Cargo.toml": "Rust",
}

FRAMEWORK_SIGNALS = {
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "express": "Express",
    "koa": "Koa",
    "hapi": "Hapi",
    "nestjs": "NestJS",
    "nextjs": "Next.js",
    "react": "React",
    "vue": "Vue",
    "angular": "Angular",
    "svelte": "Svelte",
    "rust": "Rust",
    "go": "Go",
    "gin": "Gin",
    "echo": "Echo",
    "actix": "Actix",
}

IGNORE = {".git", "venv", "env", ".venv", "node_modules", "__pycache__",
          ".planagent", ".tox", ".mypy_cache", ".pytest_cache", "dist",
          "build", ".eggs", "*.egg-info", "site-packages",
          ".next", ".nuxt", "coverage", ".coverage", "htmlcov",
          ".idea", ".vscode", ".DS_Store"}

# Prefixes that indicate virtual-env directories to always skip
_IGNORE_PREFIXES = ("venv", ".venv", "env")

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
    ".java", ".rb", ".php", ".cs", ".cpp", ".c", ".h",
}

CACHE_FILE = "context_cache.json"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_context(project_root: str, state: dict) -> dict:
    """Main entry: load from cache or do a full deep scan."""
    root = Path(project_root)
    state["project_root"] = str(root)
    cache_path = root / ".planagent" / CACHE_FILE

    # Try loading existing cache
    cached = _load_cache(cache_path)
    if cached:
        changed, removed = _detect_changes(root, cached)
        if not changed and not removed:
            # Cache is fresh — skip scan entirely
            state["scenario"] = cached["scenario"]
            state["existing_summary"] = cached["summary"]
            state["context_index"] = cached["index"]
            state["context_tier1"] = cached["tier1_summary"]
            state["cache_hit"] = True
            return state
        # Incremental update — only re-scan changed files
        index = cached["index"]
        index = _incremental_update(root, index, changed, removed)
    else:
        # Full first-time scan
        index = _full_scan(root)

    if not index["files"]:
        state["scenario"] = "empty"
        _save_cache(cache_path, state, {}, "")
        return state

    state["scenario"] = "existing"
    summary = _build_summary(root, index)
    tier1 = _build_tier1_summary(summary, index)

    state["existing_summary"] = summary
    state["context_index"] = index
    state["context_tier1"] = tier1
    state["cache_hit"] = False

    # Persist cache for next run
    _save_cache(cache_path, state, index, tier1)
    return state


def read_file_on_demand(project_root: str, relative_path: str) -> str | None:
    """Phase C: read a specific file when the index doesn't have enough detail."""
    target = Path(project_root) / relative_path
    if not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8", errors="ignore")[:8000]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Full scan — runs once on first use
# ---------------------------------------------------------------------------

def _full_scan(root: Path) -> dict:
    """Walk every file, extract signatures, build a complete index."""
    files_info = {}
    all_files = _collect_files(root)

    for f in all_files:
        rel = str(f.relative_to(root))
        finfo = _analyze_file(root, f)
        files_info[rel] = finfo

    return {
        "files": files_info,
        "scanned_at": datetime.now().isoformat(),
    }


def _collect_files(root: Path) -> list[Path]:
    """Collect all non-ignored files."""
    results = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        # Check against ignore patterns
        parts = f.relative_to(root).parts
        if any(_matches_ignore(p) for p in parts):
            continue
        results.append(f)
    return results


def _matches_ignore(part: str) -> bool:
    """Check if a path component matches any ignore pattern."""
    lower = part.lower()
    # Prefix match for virtual env variants (venv313, .venv311, env3, etc.)
    for prefix in _IGNORE_PREFIXES:
        if lower == prefix or (lower.startswith(prefix) and not lower[len(prefix):].isalpha()):
            return True
    for pattern in IGNORE:
        if pattern.startswith("*"):
            if part.endswith(pattern[1:]):
                return True
        elif part == pattern:
            return True
    return False


def _analyze_file(root: Path, filepath: Path) -> dict:
    """Extract metadata + code signatures from a single file."""
    rel = str(filepath.relative_to(root))
    stat = filepath.stat()
    info = {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "hash": _file_hash(filepath),
        "type": _classify_file(filepath),
    }

    # Deep analysis for code files
    if filepath.suffix in CODE_EXTENSIONS:
        info["signatures"] = _extract_signatures(filepath)

    return info


def _file_hash(filepath: Path) -> str:
    """Fast hash of file contents for change detection."""
    h = hashlib.md5()
    try:
        h.update(filepath.read_bytes())
    except Exception:
        return ""
    return h.hexdigest()


def _classify_file(filepath: Path) -> str:
    """Classify file by its role in the project."""
    name = filepath.name.lower()
    suffix = filepath.suffix.lower()

    if name in ("readme.md", "readme.rst", "readme.txt"):
        return "docs"
    if name in ("dockerfile", "docker-compose.yml", "docker-compose.yaml"):
        return "infra"
    if name in (".env", ".env.example"):
        return "config"
    if name in STACK_SIGNALS:
        return "manifest"
    if "test" in name or "spec" in name:
        return "test"
    if "migration" in str(filepath).lower():
        return "migration"
    if suffix in CODE_EXTENSIONS:
        return "code"
    if suffix in (".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"):
        return "config"
    return "other"


# ---------------------------------------------------------------------------
# AST-based signature extraction (Python focus, extensible)
# ---------------------------------------------------------------------------

def _extract_signatures(filepath: Path) -> dict:
    """Extract function/class/import signatures without sending full source."""
    suffix = filepath.suffix.lower()
    if suffix == ".py":
        return _extract_python_signatures(filepath)
    # For non-Python files, do a lightweight line-based extraction
    return _extract_generic_signatures(filepath)


def _extract_python_signatures(filepath: Path) -> dict:
    """Use Python's ast module to extract classes, functions, imports."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, Exception):
        return {}

    classes = []
    functions = []
    imports = []
    routes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.dump(base))
            classes.append({
                "name": node.name,
                "methods": methods[:15],
                "bases": bases,
                "line": node.lineno,
            })

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip methods already captured in classes
            if isinstance(getattr(node, '_parent', None), ast.ClassDef):
                continue
            args = [a.arg for a in node.args.args if a.arg != "self"]
            functions.append({
                "name": node.name,
                "args": args[:10],
                "line": node.lineno,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
            })

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    # Detect route decorators (FastAPI, Flask, Django, etc.)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                dec_str = ast.dump(dec)
                if any(kw in dec_str.lower() for kw in
                       ["route", "get", "post", "put", "delete", "patch",
                        "api_view", "action"]):
                    routes.append({
                        "function": node.name,
                        "line": node.lineno,
                    })

    sigs = {}
    if classes:
        sigs["classes"] = classes[:20]
    if functions:
        sigs["functions"] = functions[:30]
    if imports:
        sigs["imports"] = list(set(imports))[:30]
    if routes:
        sigs["routes"] = routes[:30]
    return sigs


def _extract_generic_signatures(filepath: Path) -> dict:
    """Lightweight extraction for JS/TS/Go/Rust — keyword-based."""
    try:
        lines = filepath.read_text(encoding="utf-8", errors="ignore").splitlines()[:200]
    except Exception:
        return {}

    exports = []
    imports = []
    functions = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Imports
        if stripped.startswith(("import ", "from ", "require(", "use ")):
            imports.append(stripped[:100])
        # Function/class definitions
        if any(stripped.startswith(kw) for kw in
               ["function ", "class ", "export ", "def ", "fn ",
                "func ", "pub fn ", "pub struct ", "type ", "interface "]):
            functions.append({"line": i, "sig": stripped[:120]})
        # Route patterns
        if any(kw in stripped.lower() for kw in
               ["router.", "app.get", "app.post", "app.put", "app.delete",
                "@get", "@post", "@put", "@delete"]):
            exports.append({"line": i, "sig": stripped[:120]})

    sigs = {}
    if imports:
        sigs["imports"] = imports[:20]
    if functions:
        sigs["functions"] = functions[:20]
    if exports:
        sigs["routes"] = exports[:20]
    return sigs


# ---------------------------------------------------------------------------
# Incremental update — only re-scan changed files
# ---------------------------------------------------------------------------

def _detect_changes(root: Path, cached: dict) -> tuple[list[str], list[str]]:
    """Compare cached file hashes/mtimes against current filesystem."""
    old_index = cached.get("index", {}).get("files", {})
    current_files = _collect_files(root)
    current_rels = {str(f.relative_to(root)) for f in current_files}
    old_rels = set(old_index.keys())

    changed = []
    removed = list(old_rels - current_rels)

    for f in current_files:
        rel = str(f.relative_to(root))
        if rel not in old_index:
            changed.append(rel)  # new file
        else:
            old_mtime = old_index[rel].get("mtime", 0)
            if f.stat().st_mtime != old_mtime:
                changed.append(rel)

    return changed, removed


def _incremental_update(root: Path, index: dict, changed: list[str],
                        removed: list[str]) -> dict:
    """Update only the changed/new files in the index, remove deleted ones."""
    files_info = index.get("files", {})

    for rel in removed:
        files_info.pop(rel, None)

    for rel in changed:
        filepath = root / rel
        if filepath.is_file():
            files_info[rel] = _analyze_file(root, filepath)

    index["files"] = files_info
    index["scanned_at"] = datetime.now().isoformat()
    return index


# ---------------------------------------------------------------------------
# Summary builders (tiered)
# ---------------------------------------------------------------------------

def _build_summary(root: Path, index: dict) -> dict:
    """Level 2 summary: structured project metadata."""
    files = index.get("files", {})
    file_paths = [root / rel for rel in files]

    all_classes = []
    all_functions = []
    all_routes = []
    all_imports = set()
    file_types = {}

    for rel, info in files.items():
        ftype = info.get("type", "other")
        file_types[ftype] = file_types.get(ftype, 0) + 1

        sigs = info.get("signatures", {})
        for cls in sigs.get("classes", []):
            all_classes.append(f"{rel}::{cls['name']}")
        for fn in sigs.get("functions", []):
            name = fn.get("name", fn.get("sig", "unknown")) if isinstance(fn, dict) else str(fn)
            all_functions.append(f"{rel}::{name}")
        for route in sigs.get("routes", []):
            if isinstance(route, dict):
                all_routes.append(f"{rel}::{route.get('function', route.get('sig', ''))}")
        for imp in sigs.get("imports", []):
            if isinstance(imp, str):
                all_imports.add(imp)

    return {
        "language": _detect_language(file_paths),
        "framework": _detect_framework(file_paths),
        "file_count": len(files),
        "has_tests": any(info.get("type") == "test" for info in files.values()),
        "top_folders": _top_folders(root, file_paths),
        "file_types": file_types,
        "classes": all_classes[:30],
        "functions": all_functions[:50],
        "routes": all_routes[:30],
        "key_imports": sorted(all_imports)[:40],
    }


def _build_tier1_summary(summary: dict, index: dict) -> str:
    """Level 1 summary: compressed natural-language (~200-500 tokens).
    This is what goes into the system prompt every turn."""
    parts = []
    parts.append(f"Language: {summary['language']}, Framework: {summary['framework']}")
    parts.append(f"Files: {summary['file_count']}, Tests: {'yes' if summary['has_tests'] else 'no'}")

    ft = summary.get("file_types", {})
    if ft:
        parts.append(f"File breakdown: {', '.join(f'{k}={v}' for k, v in ft.items())}")

    folders = summary.get("top_folders", [])
    if folders:
        parts.append(f"Top folders: {', '.join(folders)}")

    classes = summary.get("classes", [])
    if classes:
        parts.append(f"Key classes ({len(classes)}): {', '.join(classes[:15])}")

    routes = summary.get("routes", [])
    if routes:
        parts.append(f"Routes/endpoints ({len(routes)}): {', '.join(routes[:10])}")

    fns = summary.get("functions", [])
    if fns:
        parts.append(f"Top-level functions ({len(fns)}): {', '.join(fns[:15])}")

    imps = summary.get("key_imports", [])
    if imps:
        parts.append(f"Key dependencies: {', '.join(imps[:20])}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Language / framework detection (improved)
# ---------------------------------------------------------------------------

def _detect_language(files) -> str:
    for f in files:
        name = f.name if isinstance(f, Path) else Path(f).name
        if name in STACK_SIGNALS:
            return STACK_SIGNALS[name]
    return "unknown"


def _detect_framework(files) -> str:
    for f in files:
        fpath = f if isinstance(f, Path) else Path(f)
        if fpath.name in ("requirements.txt", "package.json", "Cargo.toml",
                          "go.mod", "pyproject.toml"):
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore").lower()
                for kw, name in FRAMEWORK_SIGNALS.items():
                    if kw in content:
                        return name
            except Exception:
                pass
    return "unknown"


def _top_folders(root, files) -> list[str]:
    seen = set()
    for f in files:
        try:
            fpath = f if isinstance(f, Path) else Path(f)
            parts = fpath.relative_to(root).parts
            if len(parts) > 1:
                seen.add(parts[0])
        except ValueError:
            pass
    return sorted(seen)[:12]


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------

def _load_cache(cache_path: Path) -> dict | None:
    """Load cached context index from disk."""
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if "index" in data and "files" in data.get("index", {}):
            return data
    except (json.JSONDecodeError, Exception):
        pass
    return None


def _save_cache(cache_path: Path, state: dict, index: dict, tier1: str) -> None:
    """Persist context index to disk for future runs."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "scenario": state.get("scenario", "unknown"),
        "summary": state.get("existing_summary", {}),
        "index": index,
        "tier1_summary": tier1,
        "cached_at": datetime.now().isoformat(),
    }
    try:
        cache_path.write_text(json.dumps(data, indent=2, default=str),
                              encoding="utf-8")
    except Exception:
        pass