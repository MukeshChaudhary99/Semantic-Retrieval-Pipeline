# Apply TLS strict-verify relaxation early, before any httpx/urllib3/huggingface
# client is constructed (model downloads happen inside background tasks).
from .core.config import get_settings as _get_settings
from .core.ssl_compat import patch_ssl_strict_verify as _patch_ssl

_patch_ssl(_get_settings().ssl_strict_verify)
