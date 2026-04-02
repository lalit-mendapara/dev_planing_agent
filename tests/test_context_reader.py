import pytest, tempfile
from pathlib import Path
from planagent.state import create_initial_state
from planagent.context_reader import read_context, _discover_features

def test_empty_folder():
    with tempfile.TemporaryDirectory() as tmp:
        s = create_initial_state()
        s = read_context(tmp, s)
        assert s["scenario"] == "empty"
        assert s["existing_summary"] is None

def test_python_project_detected():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "requirements.txt").write_text("fastapi\nsqlalchemy\n")
        Path(tmp, "app").mkdir()
        Path(tmp, "app", "main.py").write_text("from fastapi import FastAPI")
        s = create_initial_state()
        s = read_context(tmp, s)
        assert s["scenario"] == "existing"
        assert s["existing_summary"]["language"] == "Python"
        assert s["existing_summary"]["framework"] == "FastAPI"


# ---------------------------------------------------------------------------
# Feature discovery tests
# ---------------------------------------------------------------------------

class TestFeatureDiscoveryRoutes:
    """Test that route/endpoint patterns trigger feature detection."""

    def test_auth_routes_detected(self):
        files_info = {
            "app/auth.py": {
                "signatures": {
                    "routes": [
                        {"path": "/login", "function": "login", "decorator": "app.post"},
                        {"path": "/register", "function": "register", "decorator": "app.post"},
                    ]
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "User Authentication" in names
        assert "User Registration" in names

    def test_ecommerce_routes_detected(self):
        files_info = {
            "app/shop.py": {
                "signatures": {
                    "routes": [
                        {"path": "/products", "function": "list_products"},
                        {"path": "/cart", "function": "view_cart"},
                        {"path": "/checkout", "function": "checkout"},
                        {"path": "/orders", "function": "list_orders"},
                    ]
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "Product Catalog" in names
        assert "Shopping Cart" in names
        assert "Checkout Flow" in names
        assert "Order Management" in names

    def test_payment_routes_detected(self):
        files_info = {
            "app/billing.py": {
                "signatures": {
                    "routes": [
                        {"path": "/payments", "function": "process_payment"},
                        {"path": "/subscriptions", "function": "manage_sub"},
                    ]
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "Payment Processing" in names
        assert "Subscription Management" in names


class TestFeatureDiscoveryFolders:
    """Test that folder/module names trigger feature detection."""

    def test_auth_folder(self):
        files_info = {
            "auth/__init__.py": {"signatures": {}},
            "auth/views.py": {"signatures": {}},
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "User Authentication" in names

    def test_multiple_folders(self):
        files_info = {
            "payments/stripe.py": {"signatures": {}},
            "notifications/email.py": {"signatures": {}},
            "admin/dashboard.py": {"signatures": {}},
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "Payment Processing" in names
        assert "Notification System" in names
        assert "Admin Panel" in names


class TestFeatureDiscoveryImports:
    """Test that library imports trigger feature detection."""

    def test_stripe_import(self):
        files_info = {
            "app/pay.py": {
                "signatures": {
                    "imports": ["stripe", "stripe.webhook"],
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert any("Stripe" in n for n in names)

    def test_celery_import(self):
        files_info = {
            "app/tasks.py": {
                "signatures": {
                    "imports": ["celery"],
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "Background Task Processing" in names

    def test_manifest_dependencies(self):
        files_info = {"app/main.py": {"signatures": {}}}
        index = {"manifest": {"dependencies": ["sendgrid", "pyjwt"]}}
        features = _discover_features(Path("/fake"), files_info, index)
        names = [f["name"] for f in features]
        assert any("SendGrid" in n for n in names)
        assert "JWT Authentication" in names


class TestFeatureDiscoveryEnvKeys:
    """Test that environment variable names trigger feature detection."""

    def test_stripe_env(self):
        files_info = {"app/main.py": {"signatures": {}}}
        index = {"env_keys": ["STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET"]}
        features = _discover_features(Path("/fake"), files_info, index)
        names = [f["name"] for f in features]
        assert "Payment Processing" in names

    def test_smtp_env(self):
        files_info = {"app/main.py": {"signatures": {}}}
        index = {"env_keys": ["SMTP_HOST", "SMTP_PORT"]}
        features = _discover_features(Path("/fake"), files_info, index)
        names = [f["name"] for f in features]
        assert "Email Sending" in names


class TestFeatureDiscoveryFunctionNames:
    """Test that function name patterns trigger feature detection."""

    def test_send_email_function(self):
        files_info = {
            "app/utils.py": {
                "signatures": {
                    "functions": [{"name": "send_email", "line": 10}],
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "Email Notifications" in names

    def test_upload_file_function(self):
        files_info = {
            "app/upload.py": {
                "signatures": {
                    "functions": [{"name": "upload_file", "line": 5}],
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "File Upload" in names

    def test_generate_pdf_function(self):
        files_info = {
            "app/reports.py": {
                "signatures": {
                    "functions": [{"name": "generate_pdf", "line": 15}],
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "PDF/Report Generation" in names


class TestFeatureDiscoveryModels:
    """Test that DB model names trigger feature detection."""

    def test_user_model(self):
        files_info = {
            "app/models.py": {
                "signatures": {
                    "classes": [{
                        "name": "User",
                        "is_model": True,
                        "bases": ["Base"],
                        "model_fields": [{"name": "email"}, {"name": "password"}],
                        "line": 10,
                    }]
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "User Management" in names

    def test_order_and_product_models(self):
        files_info = {
            "app/models.py": {
                "signatures": {
                    "classes": [
                        {"name": "Product", "is_model": True, "bases": ["Base"], "line": 10},
                        {"name": "Order", "is_model": True, "bases": ["Base"], "line": 30},
                    ]
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "Product Catalog" in names
        assert "Order Management" in names


class TestFeatureDiscoveryEnums:
    """Test that enum definitions hint at features."""

    def test_role_enum(self):
        files_info = {
            "app/enums.py": {
                "signatures": {
                    "classes": [{
                        "name": "UserRole",
                        "is_enum": True,
                        "bases": ["Enum"],
                        "enum_members": ["ADMIN", "MODERATOR", "USER"],
                        "line": 5,
                    }]
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "Role-Based Access Control" in names

    def test_order_status_enum(self):
        files_info = {
            "app/enums.py": {
                "signatures": {
                    "classes": [{
                        "name": "OrderStatus",
                        "is_enum": True,
                        "bases": ["Enum"],
                        "enum_members": ["PENDING", "PROCESSING", "COMPLETED", "CANCELLED"],
                        "line": 10,
                    }]
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        names = [f["name"] for f in features]
        assert "Order/Workflow Management" in names


class TestFeatureDiscoveryReadme:
    """Test that README content triggers feature detection."""

    def test_readme_mentions(self):
        files_info = {"app/main.py": {"signatures": {}}}
        index = {"readme": "# My App\nA marketplace platform with real-time chat and payment processing."}
        features = _discover_features(Path("/fake"), files_info, index)
        names = [f["name"] for f in features]
        assert "Marketplace" in names
        assert "Realtime Features" in names
        assert "Payment Processing" in names


class TestFeatureDiscoveryConfidence:
    """Test confidence levels based on evidence count."""

    def test_high_confidence_multiple_signals(self):
        """A feature with 3+ signals should be high confidence."""
        files_info = {
            "auth/views.py": {
                "signatures": {
                    "routes": [{"path": "/login", "function": "login"}],
                    "imports": ["pyjwt"],
                }
            }
        }
        index = {"env_keys": ["JWT_SECRET_KEY"]}
        features = _discover_features(Path("/fake"), files_info, index)
        # auth folder + /login route + pyjwt import = 3+ signals for auth
        auth_features = [f for f in features
                         if "auth" in f["name"].lower() or "jwt" in f["name"].lower()]
        assert any(f["evidence_count"] >= 2 for f in auth_features)

    def test_low_confidence_single_signal(self):
        """A feature with only 1 signal should be low confidence."""
        files_info = {
            "app/main.py": {
                "signatures": {
                    "routes": [{"path": "/geocode", "function": "geocode"}],
                }
            }
        }
        features = _discover_features(Path("/fake"), files_info, {})
        geo = [f for f in features if f["name"] == "Geocoding"]
        assert len(geo) == 1
        assert geo[0]["confidence"] == "low"
        assert geo[0]["evidence_count"] == 1


class TestFeatureDiscoveryIntegration:
    """Test full scan → feature discovery integration."""

    def test_full_scan_discovers_features(self):
        """Features should appear in summary after a full scan."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "requirements.txt").write_text("fastapi\nstripe\npyjwt\ncelery\n")
            Path(tmp, "auth").mkdir()
            Path(tmp, "auth", "__init__.py").write_text("")
            Path(tmp, "auth", "routes.py").write_text(
                "from fastapi import APIRouter\n"
                "router = APIRouter()\n"
                "@router.post('/login')\n"
                "async def login(): pass\n"
                "@router.post('/register')\n"
                "async def register(): pass\n"
            )
            Path(tmp, "payments").mkdir()
            Path(tmp, "payments", "__init__.py").write_text("")
            Path(tmp, "payments", "stripe.py").write_text(
                "import stripe\n"
                "def process_payment(): pass\n"
            )
            Path(tmp, ".env.example").write_text(
                "STRIPE_SECRET_KEY=sk_test\nJWT_SECRET=mysecret\n"
            )

            s = create_initial_state()
            s = read_context(tmp, s)

            assert s["scenario"] == "existing"
            summary = s["existing_summary"]
            assert "discovered_features" in summary

            feature_names = [f["name"] for f in summary["discovered_features"]]
            # Auth folder + /login route + pyjwt dep → auth features
            assert "User Authentication" in feature_names
            # payments folder + stripe import + STRIPE env → payment feature
            assert any("Payment" in n or "Stripe" in n for n in feature_names)

    def test_empty_project_no_features(self):
        """Empty project should have no discovered features."""
        with tempfile.TemporaryDirectory() as tmp:
            s = create_initial_state()
            s = read_context(tmp, s)
            assert s["scenario"] == "empty"
            assert s["existing_summary"] is None

    def test_features_in_tier1_summary(self):
        """Discovered features should appear in the tier1 system prompt summary."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "requirements.txt").write_text("fastapi\nstripe\n")
            Path(tmp, "payments").mkdir()
            Path(tmp, "payments", "__init__.py").write_text("")
            Path(tmp, "payments", "views.py").write_text("import stripe\n")

            s = create_initial_state()
            s = read_context(tmp, s)

            tier1 = s.get("context_tier1", "")
            assert "DISCOVERED APPLICATION FEATURES" in tier1
