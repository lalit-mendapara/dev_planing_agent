import ast
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # fallback
    except ImportError:
        tomllib = None  # type: ignore[assignment]

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
    """Walk every file, extract signatures, build a complete index.
    Also extracts project-level context: README, config, infra, dependency graph,
    test map, entry points, and manifest metadata."""
    files_info = {}
    all_files = _collect_files(root)

    for f in all_files:
        rel = str(f.relative_to(root))
        finfo = _analyze_file(root, f)
        files_info[rel] = finfo

    index = {
        "files": files_info,
        "scanned_at": datetime.now().isoformat(),
    }

    # Project-level extractions (run once, cached)
    readme = _extract_readme(root)
    if readme:
        index["readme"] = readme

    env_keys = _extract_env_keys(root)
    if env_keys:
        index["env_keys"] = env_keys

    manifest = _extract_manifest_metadata(root)
    if manifest:
        index["manifest"] = manifest

    infra = _extract_infra_info(root)
    if infra:
        index["infra"] = infra

    dep_graph = _build_dependency_graph(root, files_info)
    if dep_graph:
        index["dependency_graph"] = dep_graph

    test_map = _extract_test_map(root, files_info)
    if test_map:
        index["test_map"] = test_map

    entry_points = _detect_entry_points(root, files_info)
    if entry_points:
        index["entry_points"] = entry_points

    # Comprehensive tech stack detection (runs after manifest + files are ready)
    tech_stack = _detect_full_tech_stack(root, files_info, index)
    if tech_stack:
        index["tech_stack"] = tech_stack

    # Automatic feature discovery (synthesize all signals into app features)
    discovered_features = _discover_features(root, files_info, index)
    if discovered_features:
        index["discovered_features"] = discovered_features

    return index


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
    """Use Python's ast module to extract comprehensive code intelligence:
    classes, functions, imports, routes, docstrings, decorators, constants,
    global variables, enums, type annotations, class attributes, and more."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, Exception):
        return {}

    # Annotate parent references for scope detection
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node  # type: ignore[attr-defined]

    # Module-level docstring
    module_doc = ast.get_docstring(tree)

    classes = []
    functions = []
    imports = []
    import_from = []  # structured: {module, names}
    routes = []
    constants = []
    global_vars = []
    decorators_all = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_info = {"name": n.name, "line": n.lineno}
                    # Method decorators (property, staticmethod, classmethod, etc.)
                    mdecs = [_decorator_name(d) for d in n.decorator_list]
                    if mdecs:
                        method_info["decorators"] = mdecs[:5]
                    # Method docstring
                    mdoc = ast.get_docstring(n)
                    if mdoc:
                        method_info["doc"] = mdoc[:120]
                    methods.append(method_info)

            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(_attr_name(base))

            # Class-level attributes (assignments in class body)
            class_attrs = []
            for n in node.body:
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    attr = {"name": n.target.id}
                    if n.annotation:
                        attr["type"] = _annotation_str(n.annotation)
                    class_attrs.append(attr)
                elif isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            class_attrs.append({"name": t.id})

            # Class decorators
            cls_decs = [_decorator_name(d) for d in node.decorator_list]

            # Class docstring
            cls_doc = ast.get_docstring(node)

            cls_info = {
                "name": node.name,
                "methods": methods[:20],
                "bases": bases,
                "line": node.lineno,
            }
            if class_attrs:
                cls_info["attributes"] = class_attrs[:20]
            if cls_decs:
                cls_info["decorators"] = cls_decs[:5]
            if cls_doc:
                cls_info["doc"] = cls_doc[:200]

            # Detect if this is an Enum
            if any(b in ("Enum", "IntEnum", "StrEnum", "Flag") for b in bases):
                cls_info["is_enum"] = True
                # Extract enum members
                members = []
                for n in node.body:
                    if isinstance(n, ast.Assign):
                        for t in n.targets:
                            if isinstance(t, ast.Name):
                                members.append(t.id)
                if members:
                    cls_info["enum_members"] = members[:30]

            # Detect if this is a DB model (Django, SQLAlchemy, Pydantic, etc.)
            _DB_BASES = {"Model", "Base", "DeclarativeBase", "BaseModel",
                         "Document", "EmbeddedDocument", "Table"}
            if any(b in _DB_BASES for b in bases):
                cls_info["is_model"] = True
                # Extract field definitions
                fields = []
                for n in node.body:
                    if isinstance(n, ast.Assign):
                        for t in n.targets:
                            if isinstance(t, ast.Name) and not t.id.startswith("_"):
                                field_info = {"name": t.id}
                                # Try to get field type from assignment
                                if isinstance(n.value, ast.Call):
                                    field_info["field_type"] = _call_name(n.value)
                                fields.append(field_info)
                    elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                        field_info = {"name": n.target.id}
                        if n.annotation:
                            field_info["type"] = _annotation_str(n.annotation)
                        if n.value and isinstance(n.value, ast.Call):
                            field_info["field_type"] = _call_name(n.value)
                        fields.append(field_info)
                if fields:
                    cls_info["model_fields"] = fields[:30]

            classes.append(cls_info)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip methods already captured in classes
            parent = getattr(node, '_parent', None)
            if isinstance(parent, ast.ClassDef):
                continue

            args = []
            for a in node.args.args:
                if a.arg == "self":
                    continue
                arg_info = a.arg
                if a.annotation:
                    arg_info += f": {_annotation_str(a.annotation)}"
                args.append(arg_info)

            fn_info = {
                "name": node.name,
                "args": args[:10],
                "line": node.lineno,
                "is_async": isinstance(node, ast.AsyncFunctionDef),
            }

            # Return type annotation
            if node.returns:
                fn_info["returns"] = _annotation_str(node.returns)

            # Function docstring
            fn_doc = ast.get_docstring(node)
            if fn_doc:
                fn_info["doc"] = fn_doc[:150]

            # All decorators on the function
            fn_decs = [_decorator_name(d) for d in node.decorator_list]
            if fn_decs:
                fn_info["decorators"] = fn_decs[:5]

            functions.append(fn_info)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
                names = [a.name for a in node.names] if node.names else []
                import_from.append({"module": node.module, "names": names[:10]})

    # Module-level constants (UPPER_CASE assignments at top level)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    if t.id.isupper() or (t.id.startswith("_") and t.id[1:].isupper()):
                        val_repr = _const_value_repr(node.value)
                        constants.append({"name": t.id, "value": val_repr, "line": node.lineno})
                    elif not t.id.startswith("_"):
                        global_vars.append({"name": t.id, "line": node.lineno})
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name.isupper():
                constants.append({"name": name, "line": node.lineno,
                                  "type": _annotation_str(node.annotation)})

    # Detect route decorators (FastAPI, Flask, Django, etc.)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                dec_str = ast.dump(dec)
                if any(kw in dec_str.lower() for kw in
                       ["route", "get", "post", "put", "delete", "patch",
                        "api_view", "action", "websocket"]):
                    route_info = {
                        "function": node.name,
                        "line": node.lineno,
                        "decorator": _decorator_name(dec),
                    }
                    # Try to extract path from decorator arguments
                    if isinstance(dec, ast.Call) and dec.args:
                        if isinstance(dec.args[0], ast.Constant):
                            route_info["path"] = str(dec.args[0].value)
                    routes.append(route_info)

    # Collect ALL decorator names for decorator pattern analysis
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                dname = _decorator_name(dec)
                if dname:
                    decorators_all.append(dname)

    sigs = {}
    if module_doc:
        sigs["module_doc"] = module_doc[:300]
    if classes:
        sigs["classes"] = classes[:25]
    if functions:
        sigs["functions"] = functions[:40]
    if imports:
        sigs["imports"] = list(set(imports))[:40]
    if import_from:
        sigs["import_details"] = import_from[:30]
    if routes:
        sigs["routes"] = routes[:30]
    if constants:
        sigs["constants"] = constants[:25]
    if global_vars:
        sigs["globals"] = global_vars[:15]
    if decorators_all:
        sigs["decorator_patterns"] = list(set(decorators_all))[:20]
    return sigs


# ---------------------------------------------------------------------------
# AST helper utilities
# ---------------------------------------------------------------------------

def _decorator_name(node) -> str:
    """Extract a human-readable decorator name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _attr_name(node)
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


def _attr_name(node) -> str:
    """Reconstruct dotted attribute name like 'app.route'."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _annotation_str(node) -> str:
    """Convert a type annotation AST node to a readable string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Attribute):
        return _attr_name(node)
    if isinstance(node, ast.Subscript):
        base = _annotation_str(node.value)
        sl = _annotation_str(node.slice)
        return f"{base}[{sl}]"
    if isinstance(node, ast.Tuple):
        return ", ".join(_annotation_str(e) for e in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return f"{_annotation_str(node.left)} | {_annotation_str(node.right)}"
    return "..."


def _call_name(node: ast.Call) -> str:
    """Extract the callable name from an ast.Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return _attr_name(node.func)
    return ""


def _const_value_repr(node) -> str:
    """Get a compact string representation of a constant's value."""
    if isinstance(node, ast.Constant):
        r = repr(node.value)
        return r[:80] if len(r) > 80 else r
    if isinstance(node, ast.Dict):
        return "{dict}"
    if isinstance(node, ast.List):
        return f"[list:{len(node.elts)} items]"
    if isinstance(node, ast.Set):
        return f"{{set:{len(node.elts)} items}}"
    if isinstance(node, ast.Tuple):
        return f"(tuple:{len(node.elts)} items)"
    if isinstance(node, ast.Call):
        return f"{_call_name(node)}(...)"
    return "..."


def _extract_generic_signatures(filepath: Path) -> dict:
    """Deep extraction for JS/TS/Go/Rust/Java/C# — keyword + pattern based.
    Reads up to 2000 lines and captures: imports, exports, functions, classes,
    interfaces, types, constants, routes, middleware, and JSDoc/GoDoc comments."""
    try:
        all_lines = filepath.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines = all_lines[:2000]  # cap for very large files
    except Exception:
        return {}

    exports = []
    imports = []
    functions = []
    classes = []
    interfaces = []
    constants = []
    routes = []
    middleware = []
    type_defs = []
    doc_comments = []

    prev_comment = ""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Collect doc comments (JSDoc /** ... */, /// for Rust, // for Go)
        if stripped.startswith(("/**", "///", "// ")):
            prev_comment = stripped[:150]
            continue
        if stripped.startswith("*") and prev_comment:
            prev_comment += " " + stripped.lstrip("* ")[:100]
            continue

        # Imports
        if stripped.startswith(("import ", "from ", "use ", "require(")):
            imports.append(stripped[:120])
        elif "require(" in stripped and ("const " in stripped or "let " in stripped or "var " in stripped):
            imports.append(stripped[:120])

        # Function / method definitions
        _FN_STARTS = ("function ", "async function ", "export function ",
                       "export async function ", "export default function ",
                       "def ", "fn ", "pub fn ", "pub async fn ",
                       "func ", "func (", "private ", "public ", "protected ")
        if any(stripped.startswith(kw) for kw in _FN_STARTS):
            fn_info = {"line": i, "sig": stripped[:150]}
            if prev_comment:
                fn_info["doc"] = prev_comment[:150]
            functions.append(fn_info)

        # Arrow functions assigned to const/let (JS/TS pattern)
        if ("=>" in stripped and
                any(stripped.startswith(kw) for kw in ("const ", "let ", "export const ", "export let "))):
            fn_info = {"line": i, "sig": stripped[:150]}
            if prev_comment:
                fn_info["doc"] = prev_comment[:150]
            functions.append(fn_info)

        # Class definitions
        if any(stripped.startswith(kw) for kw in
               ("class ", "export class ", "export default class ",
                "abstract class ", "pub struct ", "struct ", "data class ")):
            cls_info = {"line": i, "sig": stripped[:150]}
            if prev_comment:
                cls_info["doc"] = prev_comment[:150]
            classes.append(cls_info)

        # Interface / type definitions
        if any(stripped.startswith(kw) for kw in
               ("interface ", "export interface ", "type ", "export type ",
                "trait ", "pub trait ", "enum ", "pub enum ")):
            t_info = {"line": i, "sig": stripped[:150]}
            if prev_comment:
                t_info["doc"] = prev_comment[:150]
            type_defs.append(t_info)

        # Constants (UPPER_CASE or const assignments)
        if any(stripped.startswith(kw) for kw in ("const ", "export const ", "final ", "static final ")):
            # Check if it's UPPER_CASE constant
            parts = stripped.split("=", 1)
            if len(parts) >= 1:
                name_part = parts[0].strip().split()[-1] if parts[0].strip().split() else ""
                if name_part and (name_part.isupper() or name_part == name_part.upper()):
                    constants.append({"line": i, "sig": stripped[:120]})

        # Route patterns (Express, Koa, Hapi, Gin, Echo, Actix, etc.)
        _ROUTE_KW = ["router.", "app.get", "app.post", "app.put", "app.delete",
                      "app.patch", "app.use", "app.all",
                      "@get", "@post", "@put", "@delete", "@patch",
                      "@controller", "@requestmapping",
                      "r.get", "r.post", "r.put", "r.delete",
                      ".handle(", ".get(\"", ".post(\"", ".put(\"",
                      "group.get", "group.post"]
        if any(kw in stripped.lower() for kw in _ROUTE_KW):
            routes.append({"line": i, "sig": stripped[:150]})

        # Middleware patterns
        _MW_KW = ["app.use(", "middleware", "interceptor", ".use(",
                  "@middleware", "@guard", "@pipe", "@filter"]
        if any(kw in stripped.lower() for kw in _MW_KW):
            middleware.append({"line": i, "sig": stripped[:120]})

        # Reset comment if this line is not a comment
        if not stripped.startswith(("//", "*", "/**", "///", "#")):
            prev_comment = ""

    sigs = {}
    if imports:
        sigs["imports"] = imports[:30]
    if functions:
        sigs["functions"] = functions[:40]
    if classes:
        sigs["classes"] = classes[:20]
    if type_defs:
        sigs["types"] = type_defs[:20]
    if constants:
        sigs["constants"] = constants[:15]
    if routes:
        sigs["routes"] = routes[:30]
    if middleware:
        sigs["middleware"] = middleware[:10]
    return sigs


# ---------------------------------------------------------------------------
# README content extraction
# ---------------------------------------------------------------------------

def _extract_readme(root: Path) -> str | None:
    """Read and return the first ~2000 chars of the project README."""
    for name in ("README.md", "README.rst", "README.txt", "readme.md"):
        readme = root / name
        if readme.is_file():
            try:
                content = readme.read_text(encoding="utf-8", errors="ignore")
                # Return a meaningful chunk — enough to understand project purpose
                return content[:2000]
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Config / env extraction — keys only, NOT values (security)
# ---------------------------------------------------------------------------

def _extract_env_keys(root: Path) -> list[str]:
    """Extract variable names from .env / .env.example files (not values)."""
    keys = []
    for name in (".env", ".env.example", ".env.sample", ".env.local"):
        env_file = root / name
        if env_file.is_file():
            try:
                for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key = line.split("=", 1)[0].strip()
                        if key:
                            keys.append(key)
            except Exception:
                pass
            break  # only read the first found
    return keys[:50]


def _extract_manifest_metadata(root: Path) -> dict:
    """Extract metadata from pyproject.toml, package.json, Cargo.toml, go.mod."""
    meta = {}

    # pyproject.toml
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and tomllib is not None:
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            proj = data.get("project", data.get("tool", {}).get("poetry", {}))
            if proj:
                meta["name"] = proj.get("name", "")
                meta["version"] = proj.get("version", "")
                meta["description"] = proj.get("description", "")
                # Scripts / entry points
                scripts = proj.get("scripts", {})
                if scripts:
                    meta["scripts"] = {k: v for k, v in list(scripts.items())[:10]}
                # Dependencies
                deps = proj.get("dependencies", {})
                if isinstance(deps, dict):
                    meta["dependencies"] = list(deps.keys())[:30]
                elif isinstance(deps, list):
                    meta["dependencies"] = [str(d) for d in deps[:30]]
                # Dev dependencies
                dev_deps = proj.get("optional-dependencies", {}).get("dev", [])
                if not dev_deps:
                    dev_deps = data.get("tool", {}).get("poetry", {}).get(
                        "group", {}).get("dev", {}).get("dependencies", {})
                if dev_deps:
                    if isinstance(dev_deps, dict):
                        meta["dev_dependencies"] = list(dev_deps.keys())[:20]
                    else:
                        meta["dev_dependencies"] = [str(d) for d in dev_deps[:20]]
        except Exception:
            pass

    # package.json
    pkg_json = root / "package.json"
    if pkg_json.is_file():
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            meta["name"] = data.get("name", "")
            meta["version"] = data.get("version", "")
            meta["description"] = data.get("description", "")
            # npm scripts
            scripts = data.get("scripts", {})
            if scripts:
                meta["scripts"] = {k: v for k, v in list(scripts.items())[:15]}
            # Dependencies
            deps = data.get("dependencies", {})
            if deps:
                meta["dependencies"] = list(deps.keys())[:40]
            dev_deps = data.get("devDependencies", {})
            if dev_deps:
                meta["dev_dependencies"] = list(dev_deps.keys())[:30]
            # Main / entry
            if data.get("main"):
                meta["entry"] = data["main"]
        except Exception:
            pass

    # requirements.txt — parse dependency names
    req_txt = root / "requirements.txt"
    if req_txt.is_file() and "dependencies" not in meta:
        try:
            deps = []
            for line in req_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line and not line.startswith(("#", "-")):
                    # Extract package name (before ==, >=, etc.)
                    pkg = re.split(r"[>=<!~\[]", line)[0].strip()
                    if pkg:
                        deps.append(pkg)
            if deps:
                meta["dependencies"] = deps[:40]
        except Exception:
            pass

    # Cargo.toml
    cargo = root / "Cargo.toml"
    if cargo.is_file() and tomllib is not None:
        try:
            data = tomllib.loads(cargo.read_text(encoding="utf-8"))
            pkg = data.get("package", {})
            meta["name"] = pkg.get("name", "")
            meta["version"] = pkg.get("version", "")
            meta["description"] = pkg.get("description", "")
            deps = data.get("dependencies", {})
            if deps:
                meta["dependencies"] = list(deps.keys())[:30]
        except Exception:
            pass

    # go.mod
    gomod = root / "go.mod"
    if gomod.is_file():
        try:
            content = gomod.read_text(encoding="utf-8", errors="ignore")
            # Module name
            m = re.search(r"^module\s+(.+)$", content, re.MULTILINE)
            if m:
                meta["name"] = m.group(1).strip()
            # Go version
            m = re.search(r"^go\s+(.+)$", content, re.MULTILINE)
            if m:
                meta["go_version"] = m.group(1).strip()
            # Dependencies (require block)
            deps = re.findall(r"^\s+(\S+)\s+v", content, re.MULTILINE)
            if deps:
                meta["dependencies"] = deps[:30]
        except Exception:
            pass

    return meta


# ---------------------------------------------------------------------------
# Infra extraction — Dockerfile, docker-compose, CI/CD
# ---------------------------------------------------------------------------

def _extract_infra_info(root: Path) -> dict:
    """Extract infrastructure configuration from Docker, CI/CD, Makefile."""
    infra = {}

    # Dockerfile
    for dfile_name in ("Dockerfile", "dockerfile", "Dockerfile.dev", "Dockerfile.prod"):
        dfile = root / dfile_name
        if dfile.is_file():
            try:
                content = dfile.read_text(encoding="utf-8", errors="ignore")
                info = {"file": dfile_name}
                # Base image
                m = re.search(r"^FROM\s+(.+?)(?:\s+AS|\s*$)", content, re.MULTILINE | re.IGNORECASE)
                if m:
                    info["base_image"] = m.group(1).strip()
                # Exposed ports
                ports = re.findall(r"^EXPOSE\s+(.+)$", content, re.MULTILINE | re.IGNORECASE)
                if ports:
                    info["ports"] = [p.strip() for p in ports]
                # Entrypoint / CMD
                m = re.search(r"^(?:CMD|ENTRYPOINT)\s+(.+)$", content, re.MULTILINE | re.IGNORECASE)
                if m:
                    info["entrypoint"] = m.group(1).strip()[:150]
                infra.setdefault("dockerfiles", []).append(info)
            except Exception:
                pass

    # docker-compose
    for dc_name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
        dc = root / dc_name
        if dc.is_file():
            try:
                content = dc.read_text(encoding="utf-8", errors="ignore")
                # Extract service names (YAML key under services:)
                services = re.findall(r"^\s{2}(\w[\w-]*):", content, re.MULTILINE)
                if services:
                    infra["compose_services"] = services[:20]
            except Exception:
                pass
            break

    # CI/CD — GitHub Actions
    gh_dir = root / ".github" / "workflows"
    if gh_dir.is_dir():
        workflows = []
        for wf in gh_dir.glob("*.yml"):
            try:
                content = wf.read_text(encoding="utf-8", errors="ignore")
                m = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
                name = m.group(1).strip() if m else wf.name
                # Extract job names
                jobs = re.findall(r"^\s{2}(\w[\w-]*):", content, re.MULTILINE)
                workflows.append({"file": wf.name, "name": name, "jobs": jobs[:10]})
            except Exception:
                workflows.append({"file": wf.name})
        if workflows:
            infra["github_actions"] = workflows[:10]

    # Makefile targets
    makefile = root / "Makefile"
    if makefile.is_file():
        try:
            content = makefile.read_text(encoding="utf-8", errors="ignore")
            targets = re.findall(r"^([\w-]+):", content, re.MULTILINE)
            # Filter out common non-target patterns
            targets = [t for t in targets if not t.startswith((".", "_"))]
            if targets:
                infra["makefile_targets"] = targets[:20]
        except Exception:
            pass

    return infra


# ---------------------------------------------------------------------------
# Internal dependency graph — which project file imports which
# ---------------------------------------------------------------------------

def _build_dependency_graph(root: Path, files_info: dict) -> dict:
    """Build a map of internal file-to-file imports.
    Only tracks imports within the project (not external packages)."""
    graph = {}  # {file_rel: [imported_file_rels]}

    # Build a lookup: module_name -> relative file path
    module_to_file = {}
    for rel in files_info:
        p = Path(rel)
        if p.suffix == ".py":
            # Convert path to module: planagent/cli.py -> planagent.cli
            mod = str(p.with_suffix("")).replace("/", ".").replace("\\", ".")
            module_to_file[mod] = rel
            # Also register the last component: cli -> planagent/cli.py
            module_to_file[p.stem] = rel
        elif p.suffix in (".js", ".ts", ".jsx", ".tsx"):
            # Register by stem and relative path without extension
            module_to_file[str(p.with_suffix(""))] = rel
            module_to_file["./" + str(p.with_suffix(""))] = rel

    for rel, info in files_info.items():
        sigs = info.get("signatures", {})
        imports = sigs.get("imports", [])
        import_details = sigs.get("import_details", [])

        deps = set()

        # Python: check import_details for internal modules
        for imp_d in import_details:
            mod = imp_d.get("module", "")
            # Check if this module maps to a project file
            if mod in module_to_file:
                deps.add(module_to_file[mod])
            # Try partial match (e.g., "planagent.llm" matches)
            for mkey, mfile in module_to_file.items():
                if mod == mkey or mod.endswith("." + mkey.split(".")[-1]):
                    deps.add(mfile)

        # For simple imports list
        for imp in imports:
            if isinstance(imp, str):
                # JS/TS relative imports
                if imp.startswith(("./", "../")):
                    for mkey, mfile in module_to_file.items():
                        if imp.endswith(Path(mkey).stem):
                            deps.add(mfile)
                elif imp in module_to_file:
                    deps.add(module_to_file[imp])

        deps.discard(rel)  # no self-references
        if deps:
            graph[rel] = sorted(deps)

    return graph


# ---------------------------------------------------------------------------
# Test intelligence — map test files to what they test
# ---------------------------------------------------------------------------

def _extract_test_map(root: Path, files_info: dict) -> dict:
    """Map test files to the modules/features they test."""
    test_map = {}

    for rel, info in files_info.items():
        if info.get("type") != "test":
            continue
        sigs = info.get("signatures", {})

        test_info = {"file": rel, "tests": []}

        # Extract test function/class names
        for fn in sigs.get("functions", []):
            name = fn.get("name", fn.get("sig", ""))
            if isinstance(name, str) and (name.startswith("test_") or name.startswith("Test")):
                t = {"name": name}
                doc = fn.get("doc")
                if doc:
                    t["doc"] = doc[:100]
                test_info["tests"].append(t)

        for cls in sigs.get("classes", []):
            name = cls.get("name", cls.get("sig", ""))
            if isinstance(name, str) and name.startswith("Test"):
                methods = cls.get("methods", [])
                test_methods = []
                if isinstance(methods, list):
                    for m in methods:
                        mname = m.get("name", m) if isinstance(m, dict) else str(m)
                        if isinstance(mname, str) and mname.startswith("test_"):
                            test_methods.append(mname)
                test_info["tests"].append({
                    "class": name,
                    "methods": test_methods[:15],
                })

        # Infer what module this test covers from filename/imports
        tested_module = None
        # test_foo.py -> foo.py
        fname = Path(rel).stem
        if fname.startswith("test_"):
            tested_module = fname[5:]
        elif fname.endswith("_test"):
            tested_module = fname[:-5]

        # Also check imports for internal modules
        imports = sigs.get("imports", [])
        tested_imports = []
        for imp in imports:
            if isinstance(imp, str):
                # Check if it's an internal import
                for frel in files_info:
                    if frel != rel and Path(frel).stem in imp:
                        tested_imports.append(frel)

        if tested_module:
            test_info["tests_module"] = tested_module
        if tested_imports:
            test_info["tests_files"] = tested_imports[:5]

        if test_info["tests"]:
            test_map[rel] = test_info

    return test_map


# ---------------------------------------------------------------------------
# Entry point detection
# ---------------------------------------------------------------------------

_ENTRY_POINT_NAMES = {
    "main.py", "app.py", "manage.py", "server.py", "wsgi.py", "asgi.py",
    "index.py", "run.py", "__main__.py",
    "main.go", "main.rs", "main.js", "main.ts",
    "index.js", "index.ts", "server.js", "server.ts",
    "app.js", "app.ts",
}

_ENTRY_POINT_PATTERNS = {
    "if __name__", "app.run(", "uvicorn.run(", "serve(",
    "createServer(", "listen(",
    "func main()", "fn main()",
}


def _detect_entry_points(root: Path, files_info: dict) -> list[dict]:
    """Detect application entry points."""
    entries = []

    for rel, info in files_info.items():
        filepath = root / rel
        is_entry = False
        reason = ""

        # Check filename
        if filepath.name in _ENTRY_POINT_NAMES:
            is_entry = True
            reason = f"entry point file: {filepath.name}"

        # Check for entry point patterns in code
        if not is_entry and filepath.suffix in CODE_EXTENSIONS:
            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")[:3000]
                for pattern in _ENTRY_POINT_PATTERNS:
                    if pattern in content:
                        is_entry = True
                        reason = f"contains: {pattern}"
                        break
            except Exception:
                pass

        # Check manifest scripts that reference this file
        sigs = info.get("signatures", {})
        if sigs.get("module_doc"):
            # Docstrings mentioning "entry" or "main"
            doc = sigs["module_doc"].lower()
            if any(kw in doc for kw in ("entry point", "main module", "application entry")):
                is_entry = True
                reason = "docstring indicates entry point"

        if is_entry:
            entries.append({"file": rel, "reason": reason})

    return entries[:10]


# ---------------------------------------------------------------------------
# Automatic feature discovery — synthesize raw signals into app features
# ---------------------------------------------------------------------------

# Maps: signal keywords → (feature_name, confidence_source)
# Routes, folder names, class names, function names, imports, env keys
# are all used as evidence to discover features.

_FEATURE_ROUTE_SIGNALS: dict[str, str] = {
    # Auth
    "/login": "User Authentication",
    "/logout": "User Authentication",
    "/register": "User Registration",
    "/signup": "User Registration",
    "/auth": "User Authentication",
    "/token": "Token-based Authentication",
    "/refresh": "Token-based Authentication",
    "/forgot-password": "Password Reset",
    "/reset-password": "Password Reset",
    "/verify-email": "Email Verification",
    "/confirm": "Email Verification",
    "/oauth": "OAuth / Social Login",
    "/sso": "Single Sign-On",
    "/2fa": "Two-Factor Authentication",
    "/mfa": "Two-Factor Authentication",

    # User management
    "/users": "User Management",
    "/profile": "User Profile",
    "/account": "Account Management",
    "/settings": "User Settings",
    "/preferences": "User Preferences",
    "/avatar": "User Profile",
    "/roles": "Role Management",
    "/permissions": "Permission Management",

    # Content / CRUD
    "/posts": "Content Management",
    "/articles": "Content Management",
    "/blog": "Blog System",
    "/comments": "Commenting System",
    "/reviews": "Review System",
    "/ratings": "Rating System",
    "/likes": "Like/Reaction System",
    "/favorites": "Favorites/Bookmarks",
    "/bookmarks": "Favorites/Bookmarks",
    "/tags": "Tagging System",
    "/categories": "Category Management",

    # E-commerce
    "/products": "Product Catalog",
    "/cart": "Shopping Cart",
    "/checkout": "Checkout Flow",
    "/orders": "Order Management",
    "/inventory": "Inventory Management",
    "/wishlist": "Wishlist",
    "/coupons": "Coupon/Discount System",
    "/discounts": "Coupon/Discount System",

    # Payments
    "/payments": "Payment Processing",
    "/billing": "Billing System",
    "/invoices": "Invoice Management",
    "/subscriptions": "Subscription Management",
    "/plans": "Plan/Pricing Management",
    "/refunds": "Refund Processing",
    "/payouts": "Payout System",
    "/transactions": "Transaction History",
    "/webhook": "Webhook Integration",

    # Communication
    "/messages": "Messaging System",
    "/chat": "Chat System",
    "/conversations": "Chat System",
    "/notifications": "Notification System",
    "/emails": "Email System",
    "/sms": "SMS Notifications",

    # File management
    "/upload": "File Upload",
    "/uploads": "File Upload",
    "/files": "File Management",
    "/media": "Media Management",
    "/images": "Image Management",
    "/documents": "Document Management",
    "/attachments": "File Attachments",

    # Search & discovery
    "/search": "Search",
    "/explore": "Discovery/Explore",
    "/feed": "Activity Feed",
    "/timeline": "Timeline/Feed",
    "/trending": "Trending Content",
    "/recommendations": "Recommendation Engine",
    "/suggest": "Autocomplete/Suggestions",

    # Admin
    "/admin": "Admin Panel",
    "/dashboard": "Dashboard",
    "/analytics": "Analytics",
    "/reports": "Reporting",
    "/stats": "Statistics/Metrics",
    "/audit": "Audit Trail",
    "/logs": "Activity Logging",

    # Social
    "/follow": "Follow System",
    "/friends": "Friend System",
    "/connections": "Connection System",
    "/groups": "Group Management",
    "/teams": "Team Management",
    "/invitations": "Invitation System",
    "/invite": "Invitation System",

    # Location / maps
    "/locations": "Location Management",
    "/maps": "Map Integration",
    "/addresses": "Address Management",
    "/geocode": "Geocoding",

    # Scheduling
    "/events": "Event Management",
    "/calendar": "Calendar System",
    "/bookings": "Booking System",
    "/appointments": "Appointment Scheduling",
    "/schedule": "Scheduling System",
    "/availability": "Availability Management",

    # API
    "/api/v": "Versioned API",
    "/graphql": "GraphQL API",
    "/health": "Health Check Endpoint",
    "/status": "Status Endpoint",
    "/docs": "API Documentation",

    # Misc
    "/export": "Data Export",
    "/import": "Data Import",
    "/webhooks": "Webhook Integration",
    "/integrations": "Third-party Integrations",
    "/config": "Configuration Management",
}

# Folder/module name → feature mapping
_FEATURE_FOLDER_SIGNALS: dict[str, str] = {
    "auth": "User Authentication",
    "authentication": "User Authentication",
    "accounts": "Account Management",
    "users": "User Management",
    "profiles": "User Profile",
    "permissions": "Permission Management",
    "rbac": "Role-Based Access Control",
    "roles": "Role Management",
    "payments": "Payment Processing",
    "billing": "Billing System",
    "subscriptions": "Subscription Management",
    "orders": "Order Management",
    "cart": "Shopping Cart",
    "checkout": "Checkout Flow",
    "products": "Product Catalog",
    "catalog": "Product Catalog",
    "inventory": "Inventory Management",
    "notifications": "Notification System",
    "messaging": "Messaging System",
    "chat": "Chat System",
    "emails": "Email System",
    "mailer": "Email System",
    "mail": "Email System",
    "search": "Search",
    "analytics": "Analytics",
    "reports": "Reporting",
    "dashboard": "Dashboard",
    "admin": "Admin Panel",
    "uploads": "File Upload",
    "media": "Media Management",
    "storage": "File Storage",
    "tasks": "Background Tasks",
    "jobs": "Background Jobs",
    "workers": "Background Workers",
    "scheduler": "Task Scheduling",
    "cron": "Scheduled Jobs",
    "cache": "Caching Layer",
    "middleware": "Middleware Layer",
    "webhooks": "Webhook Integration",
    "integrations": "Third-party Integrations",
    "api": "REST API",
    "graphql": "GraphQL API",
    "websocket": "WebSocket Support",
    "realtime": "Realtime Features",
    "events": "Event System",
    "logging": "Logging System",
    "audit": "Audit Trail",
    "migrations": "Database Migrations",
    "seeds": "Database Seeding",
    "fixtures": "Test Fixtures",
    "tests": "Test Suite",
    "docs": "Documentation",
    "i18n": "Internationalization",
    "locales": "Localization",
    "templates": "Template Rendering",
    "export": "Data Export",
    "import": "Data Import",
    "feed": "Activity Feed",
    "social": "Social Features",
    "comments": "Commenting System",
    "reviews": "Review System",
    "ratings": "Rating System",
    "tags": "Tagging System",
    "categories": "Category Management",
    "bookings": "Booking System",
    "appointments": "Appointment Scheduling",
    "calendar": "Calendar System",
    "maps": "Map Integration",
    "geo": "Geolocation",
    "location": "Location Services",
}

# Import-based feature signals (library → feature)
_FEATURE_IMPORT_SIGNALS: dict[str, str] = {
    "stripe": "Payment Processing (Stripe)",
    "razorpay": "Payment Processing (Razorpay)",
    "paypal": "Payment Processing (PayPal)",
    "braintree": "Payment Processing (Braintree)",
    "sendgrid": "Email Notifications (SendGrid)",
    "mailgun": "Email Notifications (Mailgun)",
    "nodemailer": "Email Notifications",
    "resend": "Email Notifications (Resend)",
    "postmark": "Email Notifications (Postmark)",
    "celery": "Background Task Processing",
    "dramatiq": "Background Task Processing",
    "rq": "Background Task Processing",
    "bull": "Background Job Queue",
    "bullmq": "Background Job Queue",
    "socket.io": "Realtime (WebSocket)",
    "websockets": "Realtime (WebSocket)",
    "channels": "Realtime (Django Channels)",
    "pusher": "Realtime (Pusher)",
    "ably": "Realtime (Ably)",
    "sse-starlette": "Server-Sent Events",
    "elasticsearch": "Full-text Search (Elasticsearch)",
    "opensearch": "Full-text Search (OpenSearch)",
    "meilisearch": "Full-text Search (Meilisearch)",
    "algolia": "Search (Algolia)",
    "whoosh": "Full-text Search",
    "boto3": "AWS Integration",
    "google-cloud": "Google Cloud Integration",
    "azure": "Azure Integration",
    "cloudinary": "Image/Media Management (Cloudinary)",
    "pillow": "Image Processing",
    "sharp": "Image Processing",
    "pdfkit": "PDF Generation",
    "weasyprint": "PDF Generation",
    "reportlab": "PDF Generation",
    "twilio": "SMS/Voice (Twilio)",
    "apscheduler": "Task Scheduling",
    "django-celery-beat": "Periodic Task Scheduling",
    "cron": "Scheduled Jobs",
    "sentry-sdk": "Error Monitoring (Sentry)",
    "sentry": "Error Monitoring (Sentry)",
    "opentelemetry": "Distributed Tracing",
    "prometheus": "Metrics Collection",
    "graphene": "GraphQL API",
    "strawberry": "GraphQL API",
    "ariadne": "GraphQL API",
    "apollo-server": "GraphQL API",
    "grpcio": "gRPC API",
    "grpc": "gRPC API",
    "passlib": "Password Hashing",
    "bcrypt": "Password Hashing",
    "pyjwt": "JWT Authentication",
    "python-jose": "JWT Authentication",
    "jsonwebtoken": "JWT Authentication",
    "passport": "Authentication (Passport.js)",
    "next-auth": "Authentication (NextAuth)",
    "django-allauth": "Social Authentication",
    "social-auth": "Social Authentication",
    "oauth": "OAuth Integration",
    "oauthlib": "OAuth Integration",
    "firebase-admin": "Firebase Integration",
    "supabase": "Supabase Integration",
}

# Env key patterns → feature hints
_FEATURE_ENV_SIGNALS: dict[str, str] = {
    "STRIPE": "Payment Processing",
    "PAYPAL": "Payment Processing",
    "RAZORPAY": "Payment Processing",
    "SENDGRID": "Email Notifications",
    "MAILGUN": "Email Notifications",
    "SMTP": "Email Sending",
    "MAIL": "Email System",
    "TWILIO": "SMS/Voice Notifications",
    "S3_BUCKET": "Cloud File Storage (S3)",
    "AWS_S3": "Cloud File Storage (S3)",
    "CLOUDINARY": "Media Management",
    "SENTRY": "Error Monitoring",
    "REDIS": "Caching/Queue",
    "CELERY": "Background Tasks",
    "ELASTICSEARCH": "Search",
    "ALGOLIA": "Search",
    "FIREBASE": "Firebase Integration",
    "SUPABASE": "Supabase Integration",
    "PUSHER": "Realtime Notifications",
    "WEBHOOK": "Webhook Integration",
    "OAUTH": "OAuth Integration",
    "GOOGLE_CLIENT": "Google OAuth",
    "GITHUB_CLIENT": "GitHub OAuth",
    "FACEBOOK_APP": "Facebook OAuth",
    "RECAPTCHA": "CAPTCHA Protection",
    "OPENAI": "AI/LLM Integration",
    "LANGCHAIN": "AI/LLM Integration",
}

# Function/class name pattern → feature (regex patterns)
_FEATURE_NAME_PATTERNS: list[tuple[str, str]] = [
    (r"(?:send|dispatch)_?(?:email|mail|notification)", "Email Notifications"),
    (r"(?:send|dispatch)_?sms", "SMS Notifications"),
    (r"(?:send|push)_?notification", "Push Notifications"),
    (r"(?:upload|save)_?(?:file|image|media|document|avatar)", "File Upload"),
    (r"(?:resize|crop|thumbnail|optimize)_?image", "Image Processing"),
    (r"generate_?(?:pdf|report|invoice|receipt)", "PDF/Report Generation"),
    (r"(?:export|download)_?(?:csv|excel|pdf|data)", "Data Export"),
    (r"(?:import|ingest)_?(?:csv|excel|data|bulk)", "Data Import"),
    (r"(?:cache|invalidate_cache|clear_cache)", "Caching"),
    (r"(?:rate_limit|throttle)", "Rate Limiting"),
    (r"(?:search|full_text|fuzzy_search|autocomplete)", "Search"),
    (r"(?:paginate|pagination|cursor)", "Pagination"),
    (r"(?:soft_delete|restore|trash|archive)", "Soft Delete/Archive"),
    (r"(?:audit|log_action|track_event|activity_log)", "Audit Trail"),
    (r"(?:encrypt|decrypt|hash_password|verify_password)", "Security/Encryption"),
    (r"(?:generate_token|verify_token|refresh_token)", "Token Management"),
    (r"(?:schedule|cron|periodic|run_at)", "Task Scheduling"),
    (r"(?:notify|broadcast|publish|emit_event)", "Event/Notification System"),
    (r"(?:webhook|handle_webhook|process_webhook)", "Webhook Processing"),
    (r"(?:validate|sanitize|clean_input)", "Input Validation"),
    (r"(?:migrate|seed|rollback)", "Database Migrations"),
    (r"(?:backup|snapshot|dump)", "Backup System"),
    (r"(?:translate|localize|i18n|gettext)", "Internationalization"),
]


def _discover_features(root: Path, files_info: dict, index: dict) -> list[dict]:
    """Discover application features by synthesizing multiple signals:
    1. Route/endpoint patterns
    2. Folder/module names
    3. Library imports
    4. Environment variable names
    5. Function/class name patterns
    6. Model names and fields
    7. Decorator patterns

    Returns a list of dicts: [{name, confidence, evidence: [str]}]
    confidence: 'high' (3+ signals), 'medium' (2), 'low' (1)
    """
    # feature_name → set of evidence strings
    evidence_map: dict[str, set[str]] = {}

    def _add(feature: str, source: str):
        evidence_map.setdefault(feature, set()).add(source)

    # --- 1. Route/endpoint analysis ---
    for rel, info in files_info.items():
        sigs = info.get("signatures", {})
        for route in sigs.get("routes", []):
            if isinstance(route, dict):
                path = route.get("path", "")
                sig = route.get("sig", "")
                route_str = path or sig
            else:
                route_str = str(route)
            route_lower = route_str.lower()
            for pattern, feature in _FEATURE_ROUTE_SIGNALS.items():
                if pattern in route_lower:
                    _add(feature, f"route: {route_str[:80]}")
                    break

    # --- 2. Folder/module names ---
    seen_folders: set[str] = set()
    for rel in files_info:
        parts = Path(rel).parts
        for part in parts[:-1]:  # skip the filename itself
            lower = part.lower()
            if lower not in seen_folders:
                seen_folders.add(lower)
                if lower in _FEATURE_FOLDER_SIGNALS:
                    _add(_FEATURE_FOLDER_SIGNALS[lower],
                         f"folder: {part}/")

    # --- 3. Import-based feature detection ---
    all_imports: set[str] = set()
    for rel, info in files_info.items():
        sigs = info.get("signatures", {})
        for imp in sigs.get("imports", []):
            if isinstance(imp, str):
                all_imports.add(imp.lower())
                # Add root package: "stripe.webhook" → "stripe"
                root_pkg = imp.split(".")[0].lower()
                all_imports.add(root_pkg)

    manifest = index.get("manifest", {})
    for dep in manifest.get("dependencies", []):
        all_imports.add(dep.lower())

    for signal, feature in _FEATURE_IMPORT_SIGNALS.items():
        if signal.lower() in all_imports:
            _add(feature, f"import: {signal}")

    # --- 4. Environment variable signals ---
    env_keys = index.get("env_keys", [])
    for key in env_keys:
        key_upper = key.upper()
        for pattern, feature in _FEATURE_ENV_SIGNALS.items():
            if pattern in key_upper:
                _add(feature, f"env: {key}")
                break

    # --- 5. Function/class name pattern matching ---
    for rel, info in files_info.items():
        sigs = info.get("signatures", {})
        # Check function names
        for fn in sigs.get("functions", []):
            fname = fn.get("name", "") if isinstance(fn, dict) else str(fn)
            fname_lower = fname.lower()
            for pattern, feature in _FEATURE_NAME_PATTERNS:
                if re.search(pattern, fname_lower):
                    _add(feature, f"function: {fname} in {rel}")
                    break

        # Check class names
        for cls in sigs.get("classes", []):
            if isinstance(cls, dict):
                cname = cls.get("name", "")
                # Model-based feature detection
                if cls.get("is_model"):
                    model_name = cname.lower()
                    _MODEL_FEATURES = {
                        "user": "User Management",
                        "profile": "User Profile",
                        "product": "Product Catalog",
                        "order": "Order Management",
                        "orderitem": "Order Management",
                        "payment": "Payment Processing",
                        "transaction": "Transaction History",
                        "invoice": "Invoice Management",
                        "subscription": "Subscription Management",
                        "notification": "Notification System",
                        "message": "Messaging System",
                        "comment": "Commenting System",
                        "review": "Review System",
                        "rating": "Rating System",
                        "tag": "Tagging System",
                        "category": "Category Management",
                        "address": "Address Management",
                        "cart": "Shopping Cart",
                        "cartitem": "Shopping Cart",
                        "wishlist": "Wishlist",
                        "booking": "Booking System",
                        "appointment": "Appointment Scheduling",
                        "event": "Event Management",
                        "ticket": "Ticketing System",
                        "coupon": "Coupon/Discount System",
                        "discount": "Coupon/Discount System",
                        "refund": "Refund Processing",
                        "invitation": "Invitation System",
                        "team": "Team Management",
                        "organization": "Organization Management",
                        "role": "Role Management",
                        "permission": "Permission Management",
                        "auditlog": "Audit Trail",
                        "activitylog": "Activity Logging",
                        "file": "File Management",
                        "media": "Media Management",
                        "image": "Image Management",
                        "document": "Document Management",
                        "page": "CMS/Page Management",
                        "post": "Content Management",
                        "article": "Content Management",
                        "blog": "Blog System",
                    }
                    for model_key, feature in _MODEL_FEATURES.items():
                        if model_name == model_key or model_name.endswith(model_key):
                            _add(feature, f"model: {cname} in {rel}")
                            break

                # Enum-based feature hints
                if cls.get("is_enum"):
                    enum_name = cname.lower()
                    if any(k in enum_name for k in ("status", "state", "type", "role")):
                        members = cls.get("enum_members", [])
                        members_lower = [m.lower() for m in members]
                        if any(m in members_lower for m in
                               ("active", "inactive", "suspended", "banned")):
                            _add("Account Status Management",
                                 f"enum: {cname} in {rel}")
                        if any(m in members_lower for m in
                               ("pending", "processing", "completed",
                                "cancelled", "refunded")):
                            _add("Order/Workflow Management",
                                 f"enum: {cname} in {rel}")
                        if any(m in members_lower for m in
                               ("admin", "moderator", "user", "editor")):
                            _add("Role-Based Access Control",
                                 f"enum: {cname} in {rel}")

    # --- 6. Decorator-based features ---
    for rel, info in files_info.items():
        sigs = info.get("signatures", {})
        for dec in sigs.get("decorator_patterns", []):
            dec_lower = dec.lower() if isinstance(dec, str) else ""
            if any(k in dec_lower for k in ("login_required", "auth",
                                             "permission", "jwt_required",
                                             "protected")):
                _add("Authentication & Authorization",
                     f"decorator: {dec}")
            if any(k in dec_lower for k in ("rate_limit", "throttle",
                                             "ratelimit")):
                _add("Rate Limiting", f"decorator: {dec}")
            if any(k in dec_lower for k in ("cache", "cached",
                                             "memoize")):
                _add("Caching", f"decorator: {dec}")
            if any(k in dec_lower for k in ("celery", "task",
                                             "shared_task", "job")):
                _add("Background Tasks", f"decorator: {dec}")
            if any(k in dec_lower for k in ("schedule", "periodic",
                                             "crontab")):
                _add("Task Scheduling", f"decorator: {dec}")
            if any(k in dec_lower for k in ("validate", "validator")):
                _add("Input Validation", f"decorator: {dec}")
            if any(k in dec_lower for k in ("middleware",)):
                _add("Middleware Layer", f"decorator: {dec}")

    # --- 7. README-based feature hints ---
    readme = index.get("readme", "")
    if readme:
        readme_lower = readme.lower()
        _README_FEATURES = {
            "authentication": "User Authentication",
            "authorization": "Authorization",
            "payment": "Payment Processing",
            "notification": "Notification System",
            "real-time": "Realtime Features",
            "real time": "Realtime Features",
            "websocket": "WebSocket Support",
            "file upload": "File Upload",
            "search": "Search",
            "analytics": "Analytics",
            "dashboard": "Dashboard",
            "api": "REST API",
            "graphql": "GraphQL API",
            "microservice": "Microservice Architecture",
            "email": "Email System",
            "chat": "Chat System",
            "e-commerce": "E-commerce",
            "ecommerce": "E-commerce",
            "marketplace": "Marketplace",
            "subscription": "Subscription Management",
            "billing": "Billing System",
            "multi-tenant": "Multi-tenancy",
        }
        for keyword, feature in _README_FEATURES.items():
            if keyword in readme_lower:
                _add(feature, "readme mention")

    # --- Build final feature list sorted by evidence count ---
    features = []
    for name, evidences in evidence_map.items():
        count = len(evidences)
        if count >= 3:
            confidence = "high"
        elif count >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        features.append({
            "name": name,
            "confidence": confidence,
            "evidence_count": count,
            "evidence": sorted(evidences)[:5],  # cap evidence list
        })

    # Sort: high confidence first, then by evidence count
    _CONF_ORDER = {"high": 0, "medium": 1, "low": 2}
    features.sort(key=lambda f: (_CONF_ORDER[f["confidence"]],
                                  -f["evidence_count"]))

    return features[:40]  # cap at 40 features


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
    """Update only the changed/new files in the index, remove deleted ones.
    Also refreshes project-level extractions that depend on file content."""
    files_info = index.get("files", {})

    for rel in removed:
        files_info.pop(rel, None)

    for rel in changed:
        filepath = root / rel
        if filepath.is_file():
            files_info[rel] = _analyze_file(root, filepath)

    index["files"] = files_info
    index["scanned_at"] = datetime.now().isoformat()

    # Re-run project-level extractions since files changed
    # These are cheap and ensure the cache stays accurate
    readme = _extract_readme(root)
    if readme:
        index["readme"] = readme

    env_keys = _extract_env_keys(root)
    if env_keys:
        index["env_keys"] = env_keys
    elif "env_keys" in index:
        del index["env_keys"]

    manifest = _extract_manifest_metadata(root)
    if manifest:
        index["manifest"] = manifest

    infra = _extract_infra_info(root)
    if infra:
        index["infra"] = infra
    elif "infra" in index:
        del index["infra"]

    dep_graph = _build_dependency_graph(root, files_info)
    if dep_graph:
        index["dependency_graph"] = dep_graph
    elif "dependency_graph" in index:
        del index["dependency_graph"]

    test_map = _extract_test_map(root, files_info)
    if test_map:
        index["test_map"] = test_map
    elif "test_map" in index:
        del index["test_map"]

    entry_points = _detect_entry_points(root, files_info)
    if entry_points:
        index["entry_points"] = entry_points
    elif "entry_points" in index:
        del index["entry_points"]

    tech_stack = _detect_full_tech_stack(root, files_info, index)
    if tech_stack:
        index["tech_stack"] = tech_stack
    elif "tech_stack" in index:
        del index["tech_stack"]

    discovered_features = _discover_features(root, files_info, index)
    if discovered_features:
        index["discovered_features"] = discovered_features
    elif "discovered_features" in index:
        del index["discovered_features"]

    return index


# ---------------------------------------------------------------------------
# Summary builders (tiered)
# ---------------------------------------------------------------------------

def _build_summary(root: Path, index: dict) -> dict:
    """Level 2 summary: comprehensive structured project metadata.
    Includes classes (with model/enum flags), functions (with docs), routes (with paths),
    constants, decorator patterns, DB models, env config, infra, dependency graph,
    test coverage map, entry points, and README summary."""
    files = index.get("files", {})
    file_paths = [root / rel for rel in files]

    all_classes = []
    all_functions = []
    all_routes = []
    all_imports = set()
    all_models = []
    all_enums = []
    all_constants = []
    all_decorators = set()
    all_middleware = []
    file_types = {}

    for rel, info in files.items():
        ftype = info.get("type", "other")
        file_types[ftype] = file_types.get(ftype, 0) + 1

        sigs = info.get("signatures", {})

        for cls in sigs.get("classes", []):
            if isinstance(cls, dict):
                cls_entry = f"{rel}::{cls['name']}"
                # Add doc hint if available
                if cls.get("doc"):
                    cls_entry += f" ({cls['doc'][:60]})"
                all_classes.append(cls_entry)

                # Track models separately
                if cls.get("is_model"):
                    model_info = {"file": rel, "name": cls["name"]}
                    if cls.get("model_fields"):
                        model_info["fields"] = [f.get("name", "") for f in cls["model_fields"]]
                    all_models.append(model_info)

                # Track enums separately
                if cls.get("is_enum"):
                    enum_info = {"file": rel, "name": cls["name"]}
                    if cls.get("enum_members"):
                        enum_info["members"] = cls["enum_members"]
                    all_enums.append(enum_info)
            else:
                all_classes.append(f"{rel}::{cls}")

        for fn in sigs.get("functions", []):
            if isinstance(fn, dict):
                name = fn.get("name", fn.get("sig", "unknown"))
                fn_entry = f"{rel}::{name}"
                if fn.get("doc"):
                    fn_entry += f" ({fn['doc'][:50]})"
                all_functions.append(fn_entry)
            else:
                all_functions.append(f"{rel}::{fn}")

        for route in sigs.get("routes", []):
            if isinstance(route, dict):
                route_str = f"{rel}::{route.get('function', route.get('sig', ''))}"
                if route.get("path"):
                    route_str += f" [{route.get('decorator', '')}{route['path']}]"
                elif route.get("decorator"):
                    route_str += f" [{route['decorator']}]"
                all_routes.append(route_str)

        for imp in sigs.get("imports", []):
            if isinstance(imp, str):
                all_imports.add(imp)

        for c in sigs.get("constants", []):
            if isinstance(c, dict):
                all_constants.append(f"{rel}::{c.get('name', '')}")

        for d in sigs.get("decorator_patterns", []):
            if isinstance(d, str):
                all_decorators.add(d)

        for mw in sigs.get("middleware", []):
            if isinstance(mw, dict):
                all_middleware.append(mw.get("sig", "")[:80])

    summary = {
        "language": _detect_language(file_paths),
        "framework": _detect_framework(file_paths),
        "file_count": len(files),
        "has_tests": any(info.get("type") == "test" for info in files.values()),
        "top_folders": _top_folders(root, file_paths),
        "file_types": file_types,
        "classes": all_classes[:40],
        "functions": all_functions[:60],
        "routes": all_routes[:40],
        "key_imports": sorted(all_imports)[:50],
    }

    # New enriched fields
    if all_models:
        summary["models"] = all_models[:20]
    if all_enums:
        summary["enums"] = all_enums[:15]
    if all_constants:
        summary["constants"] = all_constants[:30]
    if all_decorators:
        summary["decorator_patterns"] = sorted(all_decorators)[:20]
    if all_middleware:
        summary["middleware"] = all_middleware[:10]

    # Project-level data from index
    if index.get("readme"):
        # Extract first meaningful paragraph as project description
        readme_lines = index["readme"].split("\n")
        desc_lines = []
        for line in readme_lines:
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "!", "[", "---", "===")):
                desc_lines.append(stripped)
            if len(desc_lines) >= 3:
                break
        if desc_lines:
            summary["readme_summary"] = " ".join(desc_lines)[:300]

    if index.get("env_keys"):
        summary["env_keys"] = index["env_keys"]
    if index.get("manifest"):
        summary["manifest"] = index["manifest"]
    if index.get("infra"):
        summary["infra"] = index["infra"]
    if index.get("dependency_graph"):
        summary["dependency_graph"] = index["dependency_graph"]
    if index.get("test_map"):
        summary["test_map"] = index["test_map"]
    if index.get("entry_points"):
        summary["entry_points"] = index["entry_points"]
    if index.get("tech_stack"):
        summary["tech_stack"] = index["tech_stack"]
    if index.get("discovered_features"):
        summary["discovered_features"] = index["discovered_features"]

    return summary


def _build_tier1_summary(summary: dict, index: dict) -> str:
    """Level 1 summary: dense natural-language (~400-800 tokens).
    This is what goes into the system prompt every turn.
    Contains everything the LLM needs to answer developer questions
    from cache — without reading files again."""
    parts = []

    # --- Project identity ---
    manifest = summary.get("manifest", {})
    if manifest.get("name"):
        parts.append(f"Project: {manifest['name']} v{manifest.get('version', '?')}")
        if manifest.get("description"):
            parts.append(f"Description: {manifest['description']}")

    if summary.get("readme_summary"):
        parts.append(f"README: {summary['readme_summary']}")

    parts.append(f"Language: {summary['language']}, Framework: {summary['framework']}")
    parts.append(f"Files: {summary['file_count']}, Tests: {'yes' if summary['has_tests'] else 'no'}")

    # --- Full tech stack (the foundation for all planning) ---
    ts = summary.get("tech_stack", {})
    if ts:
        # Readable category labels
        _CAT_LABELS = {
            "database": "Database", "database_cache": "Cache/Store",
            "orm": "ORM", "migration": "Migrations", "auth": "Auth",
            "queue": "Message Queue", "api_style": "API Style",
            "api_docs": "API Docs", "testing": "Testing",
            "template": "Templates", "build": "Build Tools",
            "linter": "Linting/Formatting", "cloud": "Cloud",
            "infra_tool": "Infra Tools", "ci_cd": "CI/CD",
            "container": "Containers", "monitoring": "Monitoring",
            "storage": "Storage", "payment": "Payments",
            "email": "Email", "realtime": "Realtime",
            "css": "CSS/UI", "ai": "AI/ML",
            "validation": "Validation", "scheduler": "Scheduling",
            "http_client": "HTTP Client", "package_manager": "Package Manager",
            "python_version": "Python", "runtime": "Runtime",
            "language_extra": "Language", "config": "Config",
        }
        stack_lines = []
        for cat, items in ts.items():
            label = _CAT_LABELS.get(cat, cat.replace("_", " ").title())
            stack_lines.append(f"{label}: {', '.join(items)}")
        if stack_lines:
            parts.append("--- FULL TECH STACK ---")
            parts.extend(stack_lines)
            parts.append("--- END TECH STACK ---")

    ft = summary.get("file_types", {})
    if ft:
        parts.append(f"File breakdown: {', '.join(f'{k}={v}' for k, v in ft.items())}")

    folders = summary.get("top_folders", [])
    if folders:
        parts.append(f"Top folders: {', '.join(folders)}")

    # --- Entry points ---
    entry_points = summary.get("entry_points", [])
    if entry_points:
        eps = [f"{e['file']} ({e.get('reason', '')})" for e in entry_points[:5]]
        parts.append(f"Entry points: {', '.join(eps)}")

    # --- Code structure ---
    classes = summary.get("classes", [])
    if classes:
        parts.append(f"Key classes ({len(classes)}): {', '.join(classes[:15])}")

    # DB models with fields
    models = summary.get("models", [])
    if models:
        model_strs = []
        for m in models[:10]:
            fields = m.get("fields", [])
            if fields:
                model_strs.append(f"{m['name']}({', '.join(fields[:8])})")
            else:
                model_strs.append(m["name"])
        parts.append(f"DB Models: {', '.join(model_strs)}")

    # Enums
    enums = summary.get("enums", [])
    if enums:
        enum_strs = []
        for e in enums[:8]:
            members = e.get("members", [])
            if members:
                enum_strs.append(f"{e['name']}[{', '.join(members[:5])}]")
            else:
                enum_strs.append(e["name"])
        parts.append(f"Enums: {', '.join(enum_strs)}")

    routes = summary.get("routes", [])
    if routes:
        parts.append(f"Routes/endpoints ({len(routes)}): {', '.join(routes[:12])}")

    fns = summary.get("functions", [])
    if fns:
        parts.append(f"Top-level functions ({len(fns)}): {', '.join(fns[:15])}")

    # Constants
    constants = summary.get("constants", [])
    if constants:
        parts.append(f"Constants: {', '.join(constants[:15])}")

    # Decorator patterns (auth, middleware, etc.)
    decs = summary.get("decorator_patterns", [])
    if decs:
        parts.append(f"Decorator patterns: {', '.join(decs[:12])}")

    # Middleware
    mw = summary.get("middleware", [])
    if mw:
        parts.append(f"Middleware: {', '.join(mw[:8])}")

    imps = summary.get("key_imports", [])
    if imps:
        parts.append(f"Key dependencies: {', '.join(imps[:25])}")

    # --- Dependencies (from manifest) ---
    if manifest.get("dependencies"):
        parts.append(f"Installed packages: {', '.join(manifest['dependencies'][:25])}")
    if manifest.get("scripts"):
        scripts = manifest["scripts"]
        parts.append(f"Scripts: {', '.join(f'{k}={v}' for k, v in list(scripts.items())[:8])}")

    # --- Config & env ---
    env_keys = summary.get("env_keys", [])
    if env_keys:
        parts.append(f"Env variables: {', '.join(env_keys[:20])}")

    # --- Infra ---
    infra = summary.get("infra", {})
    if infra:
        infra_parts = []
        for df in infra.get("dockerfiles", []):
            infra_parts.append(f"Docker({df.get('base_image', '?')})")
        if infra.get("compose_services"):
            infra_parts.append(f"Compose services: {', '.join(infra['compose_services'][:8])}")
        if infra.get("github_actions"):
            names = [w.get("name", w.get("file", "?")) for w in infra["github_actions"][:5]]
            infra_parts.append(f"CI/CD: {', '.join(names)}")
        if infra.get("makefile_targets"):
            infra_parts.append(f"Make targets: {', '.join(infra['makefile_targets'][:8])}")
        if infra_parts:
            parts.append(f"Infrastructure: {' | '.join(infra_parts)}")

    # --- Dependency graph (compact: show most-depended-on files) ---
    dep_graph = summary.get("dependency_graph", {})
    if dep_graph:
        # Count how many files depend on each file
        dep_count: dict[str, int] = {}
        for deps in dep_graph.values():
            for d in deps:
                dep_count[d] = dep_count.get(d, 0) + 1
        top_deps = sorted(dep_count.items(), key=lambda x: -x[1])[:8]
        if top_deps:
            dep_strs = [f"{f}({c} dependents)" for f, c in top_deps]
            parts.append(f"Core modules: {', '.join(dep_strs)}")

    # --- Test coverage ---
    test_map = summary.get("test_map", {})
    if test_map:
        test_count = sum(len(v.get("tests", [])) for v in test_map.values())
        tested_modules = [v.get("tests_module", "?") for v in test_map.values() if v.get("tests_module")]
        parts.append(f"Test coverage: {len(test_map)} test files, {test_count} tests")
        if tested_modules:
            parts.append(f"Tested modules: {', '.join(tested_modules[:10])}")

    # --- Discovered features ---
    features = summary.get("discovered_features", [])
    if features:
        parts.append("--- DISCOVERED APPLICATION FEATURES ---")
        high = [f for f in features if f["confidence"] == "high"]
        med = [f for f in features if f["confidence"] == "medium"]
        low = [f for f in features if f["confidence"] == "low"]
        if high:
            parts.append(f"Confirmed features: {', '.join(f['name'] for f in high)}")
        if med:
            parts.append(f"Likely features: {', '.join(f['name'] for f in med)}")
        if low:
            parts.append(f"Possible features: {', '.join(f['name'] for f in low[:10])}")
        parts.append("--- END DISCOVERED FEATURES ---")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Language / framework detection (basic helpers, kept for backward compat)
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
# Comprehensive tech stack detection
# ---------------------------------------------------------------------------
# Maps: signal keyword (lowercase) → (category, detected_name)
# Scans ALL imports from all files + manifest dependencies + file patterns

_TECH_STACK_SIGNALS: dict[str, tuple[str, str]] = {
    # --- Databases ---
    "psycopg2": ("database", "PostgreSQL"),
    "psycopg": ("database", "PostgreSQL"),
    "asyncpg": ("database", "PostgreSQL"),
    "pg": ("database", "PostgreSQL"),
    "pymongo": ("database", "MongoDB"),
    "mongoengine": ("database", "MongoDB"),
    "motor": ("database", "MongoDB"),
    "mongoose": ("database", "MongoDB"),
    "pymysql": ("database", "MySQL"),
    "mysql": ("database", "MySQL"),
    "mysql2": ("database", "MySQL"),
    "sqlite3": ("database", "SQLite"),
    "aiosqlite": ("database", "SQLite"),
    "better-sqlite3": ("database", "SQLite"),
    "redis": ("database_cache", "Redis"),
    "aioredis": ("database_cache", "Redis"),
    "ioredis": ("database_cache", "Redis"),
    "memcached": ("database_cache", "Memcached"),
    "cassandra-driver": ("database", "Cassandra"),
    "neo4j": ("database", "Neo4j"),
    "elasticsearch": ("database", "Elasticsearch"),
    "opensearch": ("database", "OpenSearch"),
    "dynamodb": ("database", "DynamoDB"),
    "firestore": ("database", "Firestore"),
    "cockroachdb": ("database", "CockroachDB"),
    "clickhouse": ("database", "ClickHouse"),

    # --- ORMs / Query builders ---
    "sqlalchemy": ("orm", "SQLAlchemy"),
    "sqlmodel": ("orm", "SQLModel"),
    "tortoise": ("orm", "Tortoise ORM"),
    "peewee": ("orm", "Peewee"),
    "django.db": ("orm", "Django ORM"),
    "prisma": ("orm", "Prisma"),
    "sequelize": ("orm", "Sequelize"),
    "typeorm": ("orm", "TypeORM"),
    "drizzle-orm": ("orm", "Drizzle ORM"),
    "knex": ("orm", "Knex.js"),
    "objection": ("orm", "Objection.js"),
    "gorm": ("orm", "GORM"),
    "diesel": ("orm", "Diesel"),
    "sea-orm": ("orm", "SeaORM"),
    "databases": ("orm", "Databases (async)"),
    "alembic": ("migration", "Alembic"),
    "django.db.migrations": ("migration", "Django Migrations"),

    # --- Auth ---
    "pyjwt": ("auth", "JWT (PyJWT)"),
    "python-jose": ("auth", "JWT (python-jose)"),
    "jsonwebtoken": ("auth", "JWT"),
    "passport": ("auth", "Passport.js"),
    "passport-jwt": ("auth", "Passport JWT"),
    "passport-local": ("auth", "Passport Local"),
    "authlib": ("auth", "Authlib"),
    "oauthlib": ("auth", "OAuth"),
    "social-auth": ("auth", "Python Social Auth"),
    "django-allauth": ("auth", "django-allauth"),
    "flask-login": ("auth", "Flask-Login"),
    "fastapi-users": ("auth", "FastAPI Users"),
    "bcrypt": ("auth", "bcrypt"),
    "passlib": ("auth", "Passlib"),
    "argon2": ("auth", "Argon2"),
    "next-auth": ("auth", "NextAuth.js"),
    "clerk": ("auth", "Clerk"),
    "supabase": ("auth", "Supabase Auth"),
    "firebase-admin": ("auth", "Firebase Auth"),
    "keycloak": ("auth", "Keycloak"),
    "auth0": ("auth", "Auth0"),
    "oauth2client": ("auth", "OAuth2"),
    "django.contrib.auth": ("auth", "Django Auth"),

    # --- Message queues / Task queues ---
    "celery": ("queue", "Celery"),
    "rq": ("queue", "RQ (Redis Queue)"),
    "dramatiq": ("queue", "Dramatiq"),
    "huey": ("queue", "Huey"),
    "bull": ("queue", "Bull"),
    "bullmq": ("queue", "BullMQ"),
    "amqplib": ("queue", "RabbitMQ"),
    "pika": ("queue", "RabbitMQ"),
    "amqp": ("queue", "RabbitMQ"),
    "kafka": ("queue", "Kafka"),
    "confluent-kafka": ("queue", "Kafka"),
    "kafkajs": ("queue", "Kafka"),
    "nats": ("queue", "NATS"),
    "zeromq": ("queue", "ZeroMQ"),
    "sqs": ("queue", "AWS SQS"),

    # --- API style ---
    "graphene": ("api_style", "GraphQL (Graphene)"),
    "strawberry": ("api_style", "GraphQL (Strawberry)"),
    "ariadne": ("api_style", "GraphQL (Ariadne)"),
    "apollo-server": ("api_style", "GraphQL (Apollo)"),
    "type-graphql": ("api_style", "GraphQL (TypeGraphQL)"),
    "graphql": ("api_style", "GraphQL"),
    "grpcio": ("api_style", "gRPC"),
    "grpc": ("api_style", "gRPC"),
    "protobuf": ("api_style", "Protocol Buffers"),
    "trpc": ("api_style", "tRPC"),
    "swagger": ("api_docs", "Swagger/OpenAPI"),
    "openapi": ("api_docs", "OpenAPI"),
    "drf-spectacular": ("api_docs", "drf-spectacular"),
    "flasgger": ("api_docs", "Flasgger"),

    # --- Testing ---
    "pytest": ("testing", "pytest"),
    "unittest": ("testing", "unittest"),
    "jest": ("testing", "Jest"),
    "mocha": ("testing", "Mocha"),
    "vitest": ("testing", "Vitest"),
    "cypress": ("testing", "Cypress"),
    "playwright": ("testing", "Playwright"),
    "selenium": ("testing", "Selenium"),
    "httpx": ("testing", "HTTPX"),
    "requests": ("http_client", "Requests"),
    "supertest": ("testing", "Supertest"),
    "testing-library": ("testing", "Testing Library"),
    "factory-boy": ("testing", "Factory Boy"),
    "faker": ("testing", "Faker"),
    "hypothesis": ("testing", "Hypothesis"),

    # --- Template engines ---
    "jinja2": ("template", "Jinja2"),
    "mako": ("template", "Mako"),
    "ejs": ("template", "EJS"),
    "handlebars": ("template", "Handlebars"),
    "pug": ("template", "Pug"),
    "nunjucks": ("template", "Nunjucks"),

    # --- Build / bundler tools ---
    "webpack": ("build", "Webpack"),
    "vite": ("build", "Vite"),
    "esbuild": ("build", "esbuild"),
    "rollup": ("build", "Rollup"),
    "parcel": ("build", "Parcel"),
    "turbopack": ("build", "Turbopack"),
    "swc": ("build", "SWC"),
    "babel": ("build", "Babel"),
    "tsup": ("build", "tsup"),

    # --- Linting / formatting ---
    "eslint": ("linter", "ESLint"),
    "prettier": ("linter", "Prettier"),
    "black": ("linter", "Black"),
    "ruff": ("linter", "Ruff"),
    "flake8": ("linter", "Flake8"),
    "pylint": ("linter", "Pylint"),
    "mypy": ("linter", "mypy"),
    "pyright": ("linter", "Pyright"),
    "biome": ("linter", "Biome"),
    "stylelint": ("linter", "Stylelint"),
    "isort": ("linter", "isort"),

    # --- Cloud / deployment ---
    "boto3": ("cloud", "AWS"),
    "botocore": ("cloud", "AWS"),
    "aws-sdk": ("cloud", "AWS"),
    "google-cloud": ("cloud", "Google Cloud"),
    "azure": ("cloud", "Azure"),
    "vercel": ("cloud", "Vercel"),
    "heroku": ("cloud", "Heroku"),
    "fly.io": ("cloud", "Fly.io"),
    "railway": ("cloud", "Railway"),
    "render": ("cloud", "Render"),
    "terraform": ("infra_tool", "Terraform"),
    "pulumi": ("infra_tool", "Pulumi"),
    "ansible": ("infra_tool", "Ansible"),
    "kubernetes": ("infra_tool", "Kubernetes"),
    "helm": ("infra_tool", "Helm"),

    # --- Monitoring / observability ---
    "sentry-sdk": ("monitoring", "Sentry"),
    "sentry": ("monitoring", "Sentry"),
    "prometheus": ("monitoring", "Prometheus"),
    "datadog": ("monitoring", "Datadog"),
    "newrelic": ("monitoring", "New Relic"),
    "opentelemetry": ("monitoring", "OpenTelemetry"),
    "loguru": ("monitoring", "Loguru"),
    "structlog": ("monitoring", "structlog"),
    "winston": ("monitoring", "Winston"),
    "pino": ("monitoring", "Pino"),

    # --- Storage ---
    "boto3.s3": ("storage", "AWS S3"),
    "minio": ("storage", "MinIO"),
    "cloudinary": ("storage", "Cloudinary"),
    "multer": ("storage", "Multer (file upload)"),
    "python-multipart": ("storage", "Multipart uploads"),

    # --- Payments ---
    "stripe": ("payment", "Stripe"),
    "paypal": ("payment", "PayPal"),
    "razorpay": ("payment", "Razorpay"),
    "braintree": ("payment", "Braintree"),
    "paddle": ("payment", "Paddle"),

    # --- Email ---
    "sendgrid": ("email", "SendGrid"),
    "mailgun": ("email", "Mailgun"),
    "nodemailer": ("email", "Nodemailer"),
    "resend": ("email", "Resend"),
    "postmark": ("email", "Postmark"),

    # --- Realtime ---
    "websockets": ("realtime", "WebSockets"),
    "socket.io": ("realtime", "Socket.IO"),
    "channels": ("realtime", "Django Channels"),
    "pusher": ("realtime", "Pusher"),
    "ably": ("realtime", "Ably"),
    "sse-starlette": ("realtime", "Server-Sent Events"),

    # --- Frontend libs (detected in deps) ---
    "tailwindcss": ("css", "Tailwind CSS"),
    "bootstrap": ("css", "Bootstrap"),
    "chakra-ui": ("css", "Chakra UI"),
    "material-ui": ("css", "Material UI"),
    "shadcn": ("css", "shadcn/ui"),
    "radix-ui": ("css", "Radix UI"),
    "styled-components": ("css", "Styled Components"),
    "emotion": ("css", "Emotion"),

    # --- AI / ML ---
    "openai": ("ai", "OpenAI"),
    "langchain": ("ai", "LangChain"),
    "litellm": ("ai", "LiteLLM"),
    "transformers": ("ai", "HuggingFace Transformers"),
    "torch": ("ai", "PyTorch"),
    "tensorflow": ("ai", "TensorFlow"),
    "scikit-learn": ("ai", "scikit-learn"),
    "numpy": ("ai", "NumPy"),
    "pandas": ("ai", "Pandas"),

    # --- Validation / serialization ---
    "pydantic": ("validation", "Pydantic"),
    "marshmallow": ("validation", "Marshmallow"),
    "cerberus": ("validation", "Cerberus"),
    "zod": ("validation", "Zod"),
    "joi": ("validation", "Joi"),
    "class-validator": ("validation", "class-validator"),
    "yup": ("validation", "Yup"),

    # --- Scheduling ---
    "apscheduler": ("scheduler", "APScheduler"),
    "django-celery-beat": ("scheduler", "Celery Beat"),
    "cron": ("scheduler", "Cron"),
    "node-cron": ("scheduler", "node-cron"),
    "agenda": ("scheduler", "Agenda"),
}

# File-pattern signals (file existence → tech detection)
_FILE_TECH_SIGNALS: dict[str, tuple[str, str]] = {
    "alembic.ini": ("migration", "Alembic"),
    "alembic": ("migration", "Alembic"),
    ".eslintrc": ("linter", "ESLint"),
    ".eslintrc.js": ("linter", "ESLint"),
    ".eslintrc.json": ("linter", "ESLint"),
    "eslint.config.js": ("linter", "ESLint"),
    ".prettierrc": ("linter", "Prettier"),
    "prettier.config.js": ("linter", "Prettier"),
    "pyproject.toml": ("_check_tools", ""),  # handled separately
    "tailwind.config.js": ("css", "Tailwind CSS"),
    "tailwind.config.ts": ("css", "Tailwind CSS"),
    "postcss.config.js": ("build", "PostCSS"),
    "webpack.config.js": ("build", "Webpack"),
    "vite.config.js": ("build", "Vite"),
    "vite.config.ts": ("build", "Vite"),
    "tsconfig.json": ("language_extra", "TypeScript"),
    "babel.config.js": ("build", "Babel"),
    ".babelrc": ("build", "Babel"),
    "jest.config.js": ("testing", "Jest"),
    "jest.config.ts": ("testing", "Jest"),
    "vitest.config.ts": ("testing", "Vitest"),
    "cypress.config.js": ("testing", "Cypress"),
    "playwright.config.ts": ("testing", "Playwright"),
    ".github": ("ci_cd", "GitHub Actions"),
    ".gitlab-ci.yml": ("ci_cd", "GitLab CI"),
    "Jenkinsfile": ("ci_cd", "Jenkins"),
    "bitbucket-pipelines.yml": ("ci_cd", "Bitbucket Pipelines"),
    ".circleci": ("ci_cd", "CircleCI"),
    ".travis.yml": ("ci_cd", "Travis CI"),
    "Dockerfile": ("container", "Docker"),
    "docker-compose.yml": ("container", "Docker Compose"),
    "docker-compose.yaml": ("container", "Docker Compose"),
    "compose.yml": ("container", "Docker Compose"),
    "compose.yaml": ("container", "Docker Compose"),
    "kubernetes": ("infra_tool", "Kubernetes"),
    "k8s": ("infra_tool", "Kubernetes"),
    "terraform": ("infra_tool", "Terraform"),
    "serverless.yml": ("cloud", "Serverless Framework"),
    "vercel.json": ("cloud", "Vercel"),
    "netlify.toml": ("cloud", "Netlify"),
    "fly.toml": ("cloud", "Fly.io"),
    "render.yaml": ("cloud", "Render"),
    "Procfile": ("cloud", "Heroku"),
    "setup.cfg": ("build", "setuptools"),
    "Makefile": ("build", "Make"),
    "tox.ini": ("testing", "tox"),
    ".env": ("config", ".env file present"),
    ".env.example": ("config", ".env.example present"),
    "sentry.properties": ("monitoring", "Sentry"),
    "newrelic.ini": ("monitoring", "New Relic"),
    "prometheus.yml": ("monitoring", "Prometheus"),
    ".sentryclirc": ("monitoring", "Sentry"),
    "swagger.json": ("api_docs", "Swagger"),
    "openapi.yaml": ("api_docs", "OpenAPI"),
    "openapi.json": ("api_docs", "OpenAPI"),
}


def _detect_full_tech_stack(root: Path, files_info: dict, index: dict) -> dict:
    """Comprehensive tech stack detection by analyzing:
    1. All imports from every code file
    2. Manifest dependencies (pyproject.toml, package.json, requirements.txt, etc.)
    3. File/directory existence patterns
    4. Decorator patterns and class base classes
    Returns a dict organized by category: {category: [detected_items]}"""

    # Collect ALL signals into a single lowercase set
    all_signals: set[str] = set()

    # 1. Imports from all scanned code files
    for rel, info in files_info.items():
        sigs = info.get("signatures", {})
        for imp in sigs.get("imports", []):
            if isinstance(imp, str):
                all_signals.add(imp.lower())
                # Also add sub-parts: "django.contrib.auth" → "django", "django.contrib.auth"
                parts = imp.split(".")
                for i in range(len(parts)):
                    all_signals.add(".".join(parts[:i+1]).lower())
        for imp_d in sigs.get("import_details", []):
            mod = imp_d.get("module", "")
            if mod:
                all_signals.add(mod.lower())
                parts = mod.split(".")
                for i in range(len(parts)):
                    all_signals.add(".".join(parts[:i+1]).lower())
            for name in imp_d.get("names", []):
                all_signals.add(name.lower())

    # 2. Manifest dependencies
    manifest = index.get("manifest", {})
    for dep in manifest.get("dependencies", []):
        all_signals.add(dep.lower())
    for dep in manifest.get("dev_dependencies", []):
        all_signals.add(dep.lower())

    # 3. Match signals against the tech stack database
    stack: dict[str, list[str]] = {}
    seen_values: set[str] = set()  # avoid duplicates like "Redis" in both db and cache

    for signal, (category, name) in _TECH_STACK_SIGNALS.items():
        if signal in all_signals and name not in seen_values:
            stack.setdefault(category, []).append(name)
            seen_values.add(name)

    # 4. File/directory pattern detection
    try:
        top_items = {item.name for item in root.iterdir()}
    except Exception:
        top_items = set()

    for pattern, (category, name) in _FILE_TECH_SIGNALS.items():
        if category == "_check_tools":
            continue  # handled via manifest
        if pattern in top_items and name not in seen_values:
            stack.setdefault(category, []).append(name)
            seen_values.add(name)

    # Also check for config files deeper (e.g., .github/workflows)
    gh_dir = root / ".github" / "workflows"
    if gh_dir.is_dir() and "GitHub Actions" not in seen_values:
        stack.setdefault("ci_cd", []).append("GitHub Actions")
        seen_values.add("GitHub Actions")

    # 5. pyproject.toml tool sections (ruff, black, mypy, pytest, etc.)
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and tomllib is not None:
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            tools = data.get("tool", {})
            _TOOL_MAP = {
                "ruff": ("linter", "Ruff"),
                "black": ("linter", "Black"),
                "isort": ("linter", "isort"),
                "mypy": ("linter", "mypy"),
                "pylint": ("linter", "Pylint"),
                "pytest": ("testing", "pytest"),
                "coverage": ("testing", "Coverage.py"),
                "pyright": ("linter", "Pyright"),
                "flake8": ("linter", "Flake8"),
            }
            for tool_name, (cat, display) in _TOOL_MAP.items():
                if tool_name in tools and display not in seen_values:
                    stack.setdefault(cat, []).append(display)
                    seen_values.add(display)
        except Exception:
            pass

    # 6. Detect package manager
    pkg_managers = []
    if (root / "poetry.lock").exists():
        pkg_managers.append("Poetry")
    elif (root / "Pipfile.lock").exists():
        pkg_managers.append("Pipenv")
    elif (root / "requirements.txt").exists():
        pkg_managers.append("pip")
    if (root / "pnpm-lock.yaml").exists():
        pkg_managers.append("pnpm")
    elif (root / "yarn.lock").exists():
        pkg_managers.append("Yarn")
    elif (root / "package-lock.json").exists():
        pkg_managers.append("npm")
    if (root / "bun.lockb").exists():
        pkg_managers.append("Bun")
    if pkg_managers:
        stack["package_manager"] = pkg_managers

    # 7. Detect Python version from pyproject / runtime files
    if pyproject.is_file() and tomllib is not None:
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            py_ver = data.get("project", {}).get("requires-python", "")
            if py_ver:
                stack["python_version"] = [py_ver]
        except Exception:
            pass
    runtime = root / "runtime.txt"
    if runtime.is_file():
        try:
            ver = runtime.read_text(encoding="utf-8").strip()
            if ver:
                stack.setdefault("runtime", []).append(ver)
        except Exception:
            pass

    # 8. Node engine from package.json
    pkg_json = root / "package.json"
    if pkg_json.is_file():
        try:
            data = json.loads(pkg_json.read_text(encoding="utf-8"))
            engines = data.get("engines", {})
            if engines.get("node"):
                stack.setdefault("runtime", []).append(f"Node {engines['node']}")
        except Exception:
            pass

    return stack


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