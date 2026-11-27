"""The shared Blink session renews once at a time and stores every rotation."""

from __future__ import annotations

import asyncio

from blinkpy.auth import UnauthorizedError
from blinkpy.helpers.constants import OAUTH_TOKEN_URL
import pytest
from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMocker

from custom_components.blink_instant_events.providers.blink.api import BlinkAuth, async_push_sign_in, sign_in_request
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .conftest import AUTH

HOMESCREEN = "/api/v3/accounts/1/homescreen"


async def test_concurrent_callers_share_one_renewal(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """Two requests needing a token renew once; the rotated token is saved."""
    aioclient_mock.post(OAUTH_TOKEN_URL, json={"access_token": "new", "refresh_token": "r2", "expires_in": 3600})
    saved: list[dict] = []
    auth = BlinkAuth(async_get_clientsession(hass), {**AUTH, "expiration_date": 0}, saved.append)
    assert await asyncio.gather(auth.async_token(), auth.async_token()) == ["new", "new"]
    assert aioclient_mock.call_count == 1
    assert aioclient_mock.mock_calls[0][2]["client_id"] == "ios"
    assert saved[-1]["refresh_token"] == "r2"
    assert "password" not in saved[-1]


async def test_rejected_refresh_means_sign_in_again(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """Blink refusing the refresh token is an auth failure, not a network error."""
    aioclient_mock.post(OAUTH_TOKEN_URL, status=401)
    auth = BlinkAuth(async_get_clientsession(hass), {**AUTH, "expiration_date": 0}, lambda _: None)
    with pytest.raises(UnauthorizedError):
        await auth.async_token()


async def test_turned_away_token_renews_and_retries(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """A 401 on a token that looked valid (clock behind after a wake) renews once and retries."""
    from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMockResponse

    url = f"https://rest-u062.immedia-semi.com{HOMESCREEN}"

    async def homescreen(method: str, url: object, data: object) -> AiohttpClientMockResponse:
        sent = aioclient_mock.mock_calls[-1][3]["Authorization"]
        return AiohttpClientMockResponse(method, url, status=200 if sent == "Bearer new" else 401, json={"ok": 1})

    aioclient_mock.get(url, side_effect=homescreen)
    aioclient_mock.post(OAUTH_TOKEN_URL, json={"access_token": "new", "refresh_token": "r2", "expires_in": 3600})
    saved: list[dict] = []
    auth = BlinkAuth(async_get_clientsession(hass), {**AUTH, "expiration_date": 9e9}, saved.append)
    assert await auth.query(url=url, headers=auth.header, reqtype="get") == {"ok": 1}
    assert [call[1].path for call in aioclient_mock.mock_calls] == [HOMESCREEN, "/oauth/token", HOMESCREEN]
    assert saved[-1]["refresh_token"] == "r2"


async def test_refused_renewal_is_logged_and_not_retried(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, caplog: pytest.LogCaptureFixture
) -> None:
    """Blink's reason is logged; a refused renewal is not tried a second time for the same request."""
    url = f"https://rest-u062.immedia-semi.com{HOMESCREEN}"
    aioclient_mock.get(url, status=401)
    aioclient_mock.post(
        OAUTH_TOKEN_URL,
        status=400,
        json={"error": "invalid_grant", "error_description": "expired"},
        headers={"Content-Type": "application/json"},
    )
    auth = BlinkAuth(async_get_clientsession(hass), {**AUTH, "expiration_date": 0}, lambda _: None)
    with pytest.raises(UnauthorizedError):
        await auth.query(url=url, headers=auth.header, reqtype="get")
    assert aioclient_mock.call_count == 1  # the renewal only; no request with a dead token, no second renewal
    assert "refresh_token grant for the ios client: 400 invalid_grant expired" in caplog.text


async def test_cancelled_caller_still_saves_the_rotation(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """An app hanging up mid-renewal must not lose the refresh token Blink already rotated."""
    from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMockResponse

    async def slow(method: str, url: object, data: object) -> AiohttpClientMockResponse:
        await asyncio.sleep(0.05)
        return AiohttpClientMockResponse(
            method, url, json={"access_token": "new", "refresh_token": "r2", "expires_in": 3600}
        )

    aioclient_mock.post(OAUTH_TOKEN_URL, side_effect=slow)
    saved: list[dict] = []
    auth = BlinkAuth(async_get_clientsession(hass), {**AUTH, "expiration_date": 0}, saved.append)
    caller = asyncio.ensure_future(auth.async_token())
    await asyncio.sleep(0.01)
    caller.cancel()
    await asyncio.sleep(0.1)
    assert saved[-1]["refresh_token"] == "r2"
    assert await auth.async_token() == "new"
    assert aioclient_mock.call_count == 1


async def test_push_sign_in(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """The code is traded with the hardware id it was issued to; account, region and client follow."""
    aioclient_mock.post(OAUTH_TOKEN_URL, json={"access_token": "a", "refresh_token": "r", "expires_in": 3600})
    aioclient_mock.get(
        "https://rest-prod.immedia-semi.com/api/v1/users/tier_info", json={"tier": "u062", "account_id": 99}
    )
    aioclient_mock.get(
        "https://rest-u062.immedia-semi.com/api/v3/accounts/99/homescreen", json={}, headers={"Client-Id": "456"}
    )
    _, pending = sign_in_request()
    push = await async_push_sign_in(async_get_clientsession(hass), "code", pending)
    form = aioclient_mock.mock_calls[0][2]
    assert form["hardware_id"] == pending["hardware_id"] and form["client_id"] == "android"
    assert form["code_verifier"] == pending["verifier"]
    assert push | {"expiration_date": 0} == {
        "token": "a",
        "refresh_token": "r",
        "expiration_date": 0,
        "hardware_id": pending["hardware_id"],
        "region_id": "u062",
        "account_id": 99,
        "client_id": "456",
    }


async def test_push_call_renews_and_retries_on_401(hass: HomeAssistant, aioclient_mock: AiohttpClientMocker) -> None:
    """The push session renews once on a 401 instead of asking for the push sign-in again."""
    from pytest_homeassistant_custom_component.test_util.aiohttp import AiohttpClientMockResponse

    from custom_components.blink_instant_events.providers.blink.api import BlinkAccount, PushAuth

    from .conftest import PUSH_AUTH

    async def update(method: str, url: object, data: object) -> AiohttpClientMockResponse:
        sent = aioclient_mock.mock_calls[-1][3]["Authorization"]
        return AiohttpClientMockResponse(method, url, status=200 if sent == "Bearer new" else 401, json={})

    aioclient_mock.post("https://rest-u001.immedia-semi.com/clients/7/update", side_effect=update)
    aioclient_mock.post(OAUTH_TOKEN_URL, json={"access_token": "new", "refresh_token": "r2", "expires_in": 3600})
    saved: list[dict] = []
    push = PushAuth(async_get_clientsession(hass), {**PUSH_AUTH, "client_id": 7}, saved.append)
    await BlinkAccount(async_get_clientsession(hass), push).async_associate("fcm-token")
    assert aioclient_mock.call_count == 3
    assert saved[-1]["refresh_token"] == "r2"
