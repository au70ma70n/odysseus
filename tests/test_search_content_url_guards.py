import ipaddress

import pytest

from services.search import content as service_content


@pytest.mark.parametrize("module", [service_content])
@pytest.mark.parametrize("url", [
    "http://printer.local/",
    "http://nas.lan/",
    "http://admin.internal/",
    "http://service.intranet/",
    "http://[::ffff:169.254.169.254]/latest/meta-data/",
    "http://224.0.0.1/",
    "http://[ff02::1]/",
    "http://[::]/",
])
def test_search_content_url_guard_blocks_internal_names_and_address_classes(module, url, monkeypatch):
    monkeypatch.delenv("ODYSSEUS_ALLOW_PRIVATE_WEB_FETCH", raising=False)
    assert module._public_http_url(url) is False


@pytest.mark.parametrize("module", [service_content])
def test_search_content_url_guard_allows_internal_when_opt_in_on(monkeypatch, module):
    monkeypatch.setenv("ODYSSEUS_ALLOW_PRIVATE_WEB_FETCH", "1")
    assert module._public_http_url("http://nas.lan/") is True
    assert module._public_http_url("https://server.lan:444/api/docs/") is True


@pytest.mark.parametrize("module", [service_content])
def test_search_content_url_guard_blocks_dns_to_multicast(monkeypatch, module):
    monkeypatch.delenv("ODYSSEUS_ALLOW_PRIVATE_WEB_FETCH", raising=False)
    monkeypatch.setattr(
        module,
        "_resolve_hostname_ips",
        lambda host: [ipaddress.ip_address("224.0.0.1")],
    )

    assert module._public_http_url("https://example.test/page") is False


@pytest.mark.parametrize("module", [service_content])
def test_search_content_url_guard_still_allows_public_ip(module):
    assert module._public_http_url("https://93.184.216.34/") is True
