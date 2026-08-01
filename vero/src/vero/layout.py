"""The single source of truth for a compiled Harbor task's layout.

Every container path, service name, and port in a compiled task is defined here
once and referenced everywhere else — the compiler, the Jinja templates, the
runtime configs, and the tests. Before this module the same strings were copied
between Python constants and template literals, so changing one silently failed
to propagate to the other.

The values are effectively frozen: they are a contract with things outside this
package, including every benchmark's ``read_only_paths`` and the compiled task
directories that are checked in. Rename an attribute freely; changing what it
points at is a breaking change.

Two conventions worth keeping straight, because both say "agent":

- The *target* is the program under optimization. It may not be an agent at all
  (a solver, an index build), which is why its paths are named ``target_*``.
- The *agent* is the optimizer, which really is an agent. ``agent_volume`` is
  its context directory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskLayout:
    """Container paths and service identities inside a compiled Harbor task.

    Attributes:
        target_repo: The editable target the optimizer may change.
        trusted_repo: Immutable copy of the target, the non-regression floor.
        seed_repo: The starting point the target repo is seeded from.
        vero: Local vero source, when built from a source checkout.
        cases: Per-partition case files, root-only.
        task_source: Local Harbor task definitions, for a harbor backend.
        harness: The scoring program, for a command backend.
        overlay: Host files baked into the optimizer's workspace.
        serve_config: The trusted deployment config, root-only.
        seed_script: Script that seeds the target repo on first boot.
        inference_config: The gateway's scope config.
        agent_volume: The optimizer's context directory, written by the sidecar.
        admin_volume: Trusted state: records, ledger, candidates.
        token_dir: Directory holding the admin token.
        inference_dir: Gateway state: usage ledger and request log.
        sidecar_host: Compose service name of the trusted evaluation sidecar.
        sidecar_port: Port the sidecar listens on.
        gateway_host: Compose service name of the inference gateway.
        gateway_port: Port the gateway listens on.
        producer_api_key_env: Container-side variable holding the optimizer's
            scoped gateway token.
        producer_base_url_env: Container-side variable holding the optimizer's
            gateway scope URL.
    """

    target_repo: str = "/work/agent"
    trusted_repo: str = "/opt/agent-baseline"
    seed_repo: str = "/opt/agent-seed"
    vero: str = "/opt/vero"
    cases: str = "/opt/cases"
    task_source: str = "/opt/task-source"
    harness: str = "/opt/harness"
    overlay: str = "/opt/overlay"
    serve_config: str = "/opt/serve.json"
    seed_script: str = "/opt/seed.sh"
    inference_config: str = "/opt/inference.json"
    agent_volume: str = "/state/agent-context"
    admin_volume: str = "/state/admin"
    token_dir: str = "/state/token"
    inference_dir: str = "/state/inference"
    sidecar_host: str = "eval-sidecar"
    sidecar_port: int = 8000
    gateway_host: str = "inference-gateway"
    gateway_port: int = 8001
    eval_url_env: str = "VERO_EVAL_URL"
    gateway_upstream_api_key_env: str = "VERO_INFERENCE_UPSTREAM_API_KEY"
    gateway_upstream_base_url_env: str = "VERO_INFERENCE_UPSTREAM_BASE_URL"
    optimizer_attribution: str = "optimizer"
    # Container-side names an OpenAI-surface client reads. The compose file sets
    # both for the optimizer -- key to the producer token, base URL to its
    # gateway scope -- and the compiler excludes them from the blanking loop so
    # it does not emit the same YAML key twice. The two must stay in step: a name
    # excluded from blanking but never set would keep whatever the host passed
    # in, so routed_credential_envs below is the single list both sides use.
    producer_api_key_env: str = "OPENAI_API_KEY"
    producer_base_url_env: str = "OPENAI_BASE_URL"
    # The gateway's proxy route, with the FastAPI parameter names it binds. Both
    # the route the gateway serves and every URL built for it derive from this one
    # string, so a caller cannot construct a path the gateway will not match.
    scope_route_base: str = "/scopes/{scope_name}/{attribution}/v1"

    # Derived paths. Defined here rather than spelled out at each use site, so a
    # base path and its children cannot drift apart.

    @property
    def session_dir(self) -> str:
        return f"{self.admin_volume}/session"

    @property
    def session_rescue_archive(self) -> str:
        """Pre-finalization session snapshot, taken before artifact collection.

        Deliberately a sibling of ``session_dir`` rather than a child, so the
        archive the verifier later builds from ``session_dir`` cannot contain a
        copy of this one.
        """
        return f"{self.admin_volume}/session-rescue.tar.gz"

    @property
    def case_resources_dir(self) -> str:
        return f"{self.admin_volume}/case-resources"

    @property
    def token_path(self) -> str:
        return f"{self.token_dir}/admin.token"

    @property
    def inference_state(self) -> str:
        return f"{self.inference_dir}/usage.json"

    @property
    def inference_request_log_dir(self) -> str:
        return f"{self.inference_dir}/requests"

    @property
    def target_git(self) -> str:
        return f"{self.target_repo}/.git"

    @property
    def target_git_exclude(self) -> str:
        return f"{self.target_git}/info/exclude"

    @property
    def target_evals(self) -> str:
        return f"{self.target_repo}/.evals"

    @property
    def routed_credential_envs(self) -> tuple[str, ...]:
        """Names the compose file sets explicitly, so must not also blank."""
        return (self.producer_api_key_env, self.producer_base_url_env)

    @property
    def sidecar_url(self) -> str:
        return f"http://{self.sidecar_host}:{self.sidecar_port}"

    @property
    def gateway_url(self) -> str:
        return f"http://{self.gateway_host}:{self.gateway_port}"

    @property
    def scope_route(self) -> str:
        """The route the gateway registers, including the proxied endpoint."""
        return f"{self.scope_route_base}/{{endpoint:path}}"

    def scope_path(self, scope: str, attribution: str) -> str:
        """Path for one scope, for callers that hold their own base URL.

        The trusted backend reads its gateway URL from the deployment config
        rather than assuming this layout's, so it needs the path alone.
        """
        return self.scope_route_base.format(scope_name=scope, attribution=attribution)

    def scope_url(self, scope: str, attribution: str) -> str:
        """Base URL an OpenAI-compatible client should use for one scope.

        The attribution segment is a free-form label for per-caller accounting,
        not a security boundary: the token decides what the scope may do.
        """
        return f"{self.gateway_url}{self.scope_path(scope, attribution)}"


LAYOUT = TaskLayout()
"""The one instance. There is no reason to construct another."""
