"""Shared, fail-closed Copilot session policy for schema-only typed plans."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import math
from collections.abc import Awaitable, Callable
from importlib import import_module
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, TypeVar

from featcopilot.utils.logger import get_logger

if TYPE_CHECKING:
    from copilot import CopilotClient
    from copilot.generated.session_events import AssistantMessageData, SessionEvent
    from copilot.session import CopilotSession

logger = get_logger(__name__)
_T = TypeVar("_T")
PLAN_CLEANUP_TIMEOUT = 5.0


class PlanIsolationError(RuntimeError):
    """The SDK cannot establish the reviewed isolation or existing-login policy."""


def plan_timeout_error(error: BaseException, timeout: float, phase: str) -> BaseException:
    """Give blank timeout errors fixed context, retaining explicit provider errors."""
    if isinstance(error, (TimeoutError, asyncio.TimeoutError, concurrent.futures.TimeoutError)) and (
        not error.args or error.args == ("",)
    ):
        contextual = type(error)(f"Copilot plan {phase} timed out after {timeout:g} seconds")
        contextual.__cause__ = error
        return contextual
    return error


class IsolatedCopilotTransport:
    """Check local SDK support without starting a client or probing credentials."""

    def __init__(self) -> None:
        copilot = import_module("copilot")

        required = {
            "on_permission_request",
            "model",
            "streaming",
            "available_tools",
            "tools",
            "working_directory",
            "enable_config_discovery",
            "mcp_servers",
            "skill_directories",
            "custom_agents",
            "infinite_sessions",
        }
        if not required.issubset(inspect.signature(copilot.CopilotClient.create_session).parameters):
            raise PlanIsolationError("Installed Copilot SDK lacks reviewed session-isolation parameters")
        constructor_parameters = inspect.signature(copilot.CopilotClient).parameters
        modern_parameters = {"connection", "working_directory", "log_level", "use_logged_in_user"}
        self._uses_runtime_connection = modern_parameters.issubset(constructor_parameters)
        legacy_parameters = {"config", "auto_start"}
        if not self._uses_runtime_connection and not legacy_parameters.issubset(constructor_parameters):
            raise PlanIsolationError("Installed Copilot SDK lacks reviewed client-isolation parameters")
        try:
            # These symbols differ between SDK 0.3 and 1.0; resolve only the
            # explicitly detected contract, never fall back after a failure.
            if self._uses_runtime_connection:
                rpc = import_module("copilot.generated.rpc")

                self._transport_factory = copilot.RuntimeConnection.for_stdio
                self._permission_denied = rpc.PermissionDecisionReject()
                required_transport_parameters = {"args"}
            else:
                sdk_client = import_module("copilot.client")
                sdk_session = import_module("copilot.session")

                self._transport_factory = sdk_client.SubprocessConfig
                self._permission_denied = sdk_session.PermissionRequestResult(kind="reject")
                required_transport_parameters = {"cwd", "cli_args", "log_level", "use_logged_in_user"}
            transport_parameters = inspect.signature(self._transport_factory).parameters
        except (AttributeError, ImportError, TypeError, ValueError) as exc:
            raise PlanIsolationError("Installed Copilot SDK lacks reviewed transport-isolation parameters") from exc
        if not required_transport_parameters.issubset(transport_parameters):
            raise PlanIsolationError("Installed Copilot SDK lacks reviewed transport-isolation parameters")
        self._client_type: Callable[..., CopilotClient] = copilot.CopilotClient

    async def request(self, model: str, timeout: float, send: Callable[[CopilotSession], Awaitable[_T]]) -> _T:
        """
        Run startup, authentication, session creation and send under one deadline.

        Parameters
        ----------
        model : str
            Explicit model identity, never substituted by the adapter.
        timeout : float
            Total request deadline in seconds, excluding shutdown grace.
        send : callable
            One application request using the isolated session.

        Returns
        ------
        object
            The callback result, after disconnect and stop finish.

        Notes
        -----
        Each shutdown step gets min(timeout, PLAN_CLEANUP_TIMEOUT) seconds of
        cooperative cancellation grace. Blocking or cancellation-resistant SDK
        code cannot be forcibly terminated by this in-process adapter.
        Workspace-removal OSErrors are logged without replacing the request
        outcome; an OS-busy workspace may remain for later cleanup.
        """
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Plan request timeout must be finite and positive")
        # Discovery=False does not disable AGENTS.md. An empty cwd plus the
        # reviewed CLI flags also exclude custom instructions and built-in MCPs.
        directory = TemporaryDirectory(prefix="featcopilot-plan-")
        try:
            workspace = directory.name
            cli_args = [
                "--no-custom-instructions",
                "--disable-builtin-mcps",
                "--no-ask-user",
                "--no-auto-update",
            ]
            if self._uses_runtime_connection:
                client = self._client_type(
                    connection=self._transport_factory(args=cli_args),
                    working_directory=workspace,
                    use_logged_in_user=True,
                    log_level="error",
                )
            else:
                config = self._transport_factory(
                    cwd=workspace,
                    use_logged_in_user=True,
                    log_level="error",
                    cli_args=cli_args,
                )
                client = self._client_type(config, auto_start=False)
            session: CopilotSession | None = None
            request_phase = "startup"

            async def generate() -> _T:
                nonlocal session, request_phase
                await client.start()
                request_phase = "authentication"
                auth_status = await client.get_auth_status()
                if auth_status.isAuthenticated is not True:
                    raise PlanIsolationError(
                        "Copilot has no authenticated logged-in user; authenticate outside typed-plan generation"
                    )
                request_phase = "session creation"
                session = await client.create_session(
                    on_permission_request=lambda _request, _invocation: self._permission_denied,
                    model=model,
                    streaming=False,
                    available_tools=[],
                    tools=[],
                    working_directory=workspace,
                    enable_config_discovery=False,
                    mcp_servers={},
                    skill_directories=[],
                    custom_agents=[],
                    infinite_sessions={"enabled": False},
                )
                request_phase = "send"
                return await send(session)

            primary_error: BaseException | None = None
            try:
                return await asyncio.wait_for(generate(), timeout=timeout)
            except BaseException as exc:
                primary_error = plan_timeout_error(exc, timeout, f"request during {request_phase}")
                if primary_error is exc:
                    raise
                raise primary_error from exc
            finally:
                cleanup = [("stop", client.stop)]
                if session is not None:
                    cleanup.insert(0, ("disconnect", session.disconnect))
                cleanup_error: BaseException | None = None
                for phase, close in cleanup:
                    try:
                        await asyncio.wait_for(close(), timeout=min(timeout, PLAN_CLEANUP_TIMEOUT))
                    except BaseException as exc:
                        logger.warning("Copilot plan %s cleanup failed (%s)", phase, type(exc).__name__)
                        if cleanup_error is None:
                            cleanup_error = plan_timeout_error(
                                exc,
                                min(timeout, PLAN_CLEANUP_TIMEOUT),
                                f"{phase} cleanup (request timeout {timeout:g} seconds)",
                            )
                if primary_error is None and cleanup_error is not None:
                    raise cleanup_error
        finally:
            try:
                directory.cleanup()
            except OSError as exc:
                logger.warning(
                    "Copilot plan workspace cleanup failed (%s, errno=%s); isolated workspace may remain at %s",
                    type(exc).__name__,
                    exc.errno if type(exc.errno) is int else None,
                    directory.name,
                )


def plan_message(event: SessionEvent | None) -> AssistantMessageData:
    """Validate a text-only typed-plan response, rejecting unexpected tool requests."""
    from copilot.generated.session_events import AssistantMessageData

    if event is None or not isinstance(event.data, AssistantMessageData) or not event.data.content.strip():
        raise ValueError("Copilot returned an empty or non-text plan response")
    if event.data.tool_requests:
        raise ValueError("Copilot returned tool requests despite the tool-disabled plan contract")
    return event.data
