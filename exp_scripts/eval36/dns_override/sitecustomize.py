"""Resolve named hosts to fixed addresses for this process (put this directory on PYTHONPATH).

.36 cannot reach its DNS server (136.166.10.50, port 53 times out from .36 only), but it reaches the
Azure OpenAI private endpoint over HTTPS. EVAL36_HOSTS="host=ip[,host=ip]" maps those hosts to the
address .29 resolves them to; TLS still verifies the certificate against the host name, so the
endpoint, model and requests are exactly the ones every other host uses. Hosts not listed resolve
as usual.
"""
import os
import socket

_HOSTS = dict(item.split("=", 1) for item in os.environ.get("EVAL36_HOSTS", "").split(",") if "=" in item)
if _HOSTS:
    _getaddrinfo = socket.getaddrinfo

    def _pinned_getaddrinfo(host, *args, **kwargs):
        if isinstance(host, (bytes, bytearray)):
            host = host.decode()
        return _getaddrinfo(_HOSTS.get(host.lower() if isinstance(host, str) else host, host), *args, **kwargs)

    socket.getaddrinfo = _pinned_getaddrinfo
