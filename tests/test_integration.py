"""End-to-end integration tests.

Every cloud call is mocked, so the whole suite runs offline and incurs no
cloud spend. The mocks stand in for the *boundaries* (EC2, SSH) only - the
catalog, blueprint parser and orchestrator under test are the real ones.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import yaml

from lightos.backends.base import CloudProvider, ProvisionedNode
from lightos.catalog import CatalogError, CostCatalog, estimate_cost
from lightos.execution import DEFAULT_EXCLUDES, _is_excluded
from lightos.orchestrator import Orchestrator
from lightos.task import Task, TaskError

SAMPLE_YAML = """
name: deep-learning-integration-test
workdir: .
resources:
  accelerators: A10G
  gpu_count: 1
envs:
  LIGHTOS_RUN_MODE: integration
setup: |
  echo "Initializing dependencies..."
  pip install torch --quiet
run: |
  python3 -c "print('Integration verification successful!')"
"""


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------
class MockAWSProvider(CloudProvider):
    """Stands in for EC2 so validation never spends money."""

    name = "aws"

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.launched: List[str] = []
        self.terminated: List[str] = []

    def launch_instance(self, instance_type: str, job_id: str,
                        **kwargs: Any) -> str:
        print(f"[MOCK AWS] Provisioning {instance_type} node...")
        self.launched.append(instance_type)
        return "i-mockinstance1234567890"

    def wait_until_ready(self, instance_id: str,
                         timeout: int = 600) -> ProvisionedNode:
        return ProvisionedNode(instance_id=instance_id, host_ip="203.0.113.10",
                               username="ubuntu", cloud=self.name)

    def terminate_instance(self, instance_id: str) -> None:
        print(f"[MOCK AWS] Terminating test instance: {instance_id}")
        self.terminated.append(instance_id)


class RecordingExecutor:
    """Stands in for the SSH layer, recording what the pipeline asked for."""

    def __init__(self, exit_code: int = 0, fail_on_run: bool = False):
        self.exit_code = exit_code
        self.fail_on_run = fail_on_run
        self.connected = False
        self.disconnected = False
        self.synced_from: Any = None
        self.synced_to: Any = None
        self.script: Any = None
        self.env: Any = None
        self.run_directory: Any = None

    def connect(self) -> None:
        self.connected = True

    def sync_workspace(self, local_dir: str, remote_destination: str,
                       **kwargs: Any) -> int:
        self.synced_from = local_dir
        self.synced_to = remote_destination
        return 7

    def execute_and_stream_logs(self, bash_script: str, run_directory: str,
                                env: Dict[str, Any] = None) -> int:
        if self.fail_on_run:
            raise RuntimeError("ssh channel died mid-job")
        self.script = bash_script
        self.run_directory = run_directory
        self.env = env
        return self.exit_code

    def disconnect(self) -> None:
        self.disconnected = True


def build_orchestrator(executor: RecordingExecutor,
                       provider: MockAWSProvider,
                       catalog: CostCatalog = None) -> Orchestrator:
    return Orchestrator(
        catalog=catalog or CostCatalog(),
        provider_factory=lambda cloud, kwargs: provider,
        executor_factory=lambda node, task: executor)


# ----------------------------------------------------------------------
class TestCostCatalog(unittest.TestCase):
    """Stage 2 in isolation: the matchmaker."""

    def setUp(self) -> None:
        self.catalog = CostCatalog()

    def test_picks_cheapest_matching_accelerator(self) -> None:
        offer = self.catalog.find_cheapest_match("A10G", 1)
        self.assertEqual(offer["cloud"], "aws")
        self.assertEqual(offer["instance_type"], "g5.xlarge")

    def test_matching_is_case_insensitive(self) -> None:
        self.assertEqual(self.catalog.find_cheapest_match("a10g", 1),
                         self.catalog.find_cheapest_match("A10G", 1))

    def test_honours_minimum_gpu_count(self) -> None:
        offer = self.catalog.find_cheapest_match("A100", 8)
        self.assertGreaterEqual(offer["gpus"], 8)
        # The 1-GPU k8s node is cheaper but cannot satisfy 8x.
        self.assertEqual(offer["instance_type"], "a2-highgpu-8g")

    def test_results_are_sorted_ascending_by_cost(self) -> None:
        costs = [o["hourly_cost"]
                 for o in self.catalog.find_matches("A100", 1)]
        self.assertEqual(costs, sorted(costs))

    def test_cloud_constraint_excludes_cheaper_clouds(self) -> None:
        offer = self.catalog.find_cheapest_match("A100", 1, clouds=["gcp"])
        self.assertEqual(offer["cloud"], "gcp")

    def test_region_constraint_is_applied(self) -> None:
        matches = self.catalog.find_matches("A10G", 1, region="eu-west-1")
        self.assertEqual(matches, [])

    def test_cpu_only_request_selects_zero_gpu_capacity(self) -> None:
        offer = self.catalog.find_cheapest_match(None, 0)
        self.assertEqual(offer["gpus"], 0)

    def test_unsatisfiable_request_raises(self) -> None:
        with self.assertRaises(CatalogError) as ctx:
            self.catalog.find_cheapest_match("H100", 1)
        self.assertIn("H100", str(ctx.exception))

    def test_cost_projection(self) -> None:
        offer = self.catalog.find_cheapest_match("A10G", 1)
        self.assertAlmostEqual(estimate_cost(offer, 8), 8.048, places=4)

    def test_snapshot_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "catalog.json")
            self.catalog.to_json(path)
            reloaded = CostCatalog(catalog_path=path)
            self.assertEqual(len(reloaded), len(self.catalog))
            self.assertEqual(reloaded.find_cheapest_match("A10G", 1),
                             self.catalog.find_cheapest_match("A10G", 1))

    def test_malformed_catalog_is_rejected(self) -> None:
        with self.assertRaises(CatalogError):
            CostCatalog(entries=[{"cloud": "aws"}])


class TestTaskBlueprint(unittest.TestCase):
    """Stage 1 in isolation: the YAML contract."""

    def test_parses_resources_and_payload(self) -> None:
        task = Task.from_dict(yaml.safe_load(SAMPLE_YAML))
        self.assertEqual(task.name, "deep-learning-integration-test")
        self.assertEqual(task.accelerator, "A10G")
        self.assertEqual(task.gpu_count, 1)
        self.assertIn("pip install torch", task.payload())
        self.assertIn("Integration verification successful!", task.payload())

    def test_accelerator_shorthand_sets_count(self) -> None:
        task = Task.from_dict({"name": "t", "run": "echo hi",
                               "resources": {"accelerators": "A100:8"}})
        self.assertEqual(task.accelerator, "A100")
        self.assertEqual(task.gpu_count, 8)

    def test_missing_run_payload_is_rejected(self) -> None:
        with self.assertRaises(TaskError):
            Task.from_dict({"name": "t", "resources": {}})

    def test_missing_name_is_rejected(self) -> None:
        with self.assertRaises(TaskError):
            Task.from_dict({"run": "echo hi"})

    def test_job_ids_are_unique_and_dns_safe(self) -> None:
        task = Task.from_dict(yaml.safe_load(SAMPLE_YAML))
        first, second = task.job_id(), task.job_id()
        self.assertNotEqual(first, second)
        self.assertTrue(all(c.isalnum() or c == "-" for c in first))

    def test_workdir_resolves_relative_to_the_blueprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "task.yaml")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(SAMPLE_YAML)
            task = Task.from_yaml(path)
            self.assertEqual(os.path.realpath(task.workdir),
                             os.path.realpath(tmp))

    def test_shipped_blueprint_is_valid(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        task = Task.from_yaml(os.path.join(root, "task_config.yaml"))
        self.assertEqual(task.accelerator, "A10G")
        self.assertTrue(task.autostop)


class TestBackendRegistry(unittest.TestCase):
    """Backend resolution and the region/zone mismatch between clouds."""

    def test_known_clouds_resolve(self) -> None:
        from lightos.backends import available_providers

        self.assertEqual(set(available_providers()), {"aws", "gcp", "k8s"})

    def test_unknown_cloud_is_rejected(self) -> None:
        from lightos.backends import ProvisionError, get_provider

        with self.assertRaises(ProvisionError):
            get_provider("azure")

    def test_gcp_derives_a_zone_from_a_catalog_region(self) -> None:
        """The catalog stores regions; the Compute API needs zones."""
        from lightos.backends.gcp import GCPProvider

        self.assertEqual(GCPProvider(region="us-central1").zone,
                         "us-central1-a")
        # An explicit zone is passed through untouched.
        self.assertEqual(GCPProvider(zone="europe-west4-b").zone,
                         "europe-west4-b")
        self.assertEqual(GCPProvider(region="us-central1").region,
                         "us-central1")

    def test_orchestrator_passes_the_offer_region_to_the_backend(self) -> None:
        captured: List[Dict[str, Any]] = []
        task = Task.from_dict({"name": "t", "run": "echo hi",
                               "resources": {"accelerators": "L4"}})
        orchestrator = Orchestrator(
            catalog=CostCatalog(),
            provider_factory=lambda cloud, kwargs: (
                captured.append(kwargs) or MockAWSProvider()),
            executor_factory=lambda node, task_: RecordingExecutor())
        orchestrator.launch(task)
        # The cheapest L4 is gcp/us-central1.
        self.assertEqual(captured, [{"region": "us-central1"}])

    def test_kubernetes_offers_get_no_region_kwarg(self) -> None:
        captured: List[Dict[str, Any]] = []
        task = Task.from_dict({"name": "t", "run": "echo hi",
                               "resources": {"accelerators": "A100",
                                             "gpu_count": 1}})
        orchestrator = Orchestrator(
            catalog=CostCatalog(),
            provider_factory=lambda cloud, kwargs: (
                captured.append((cloud, kwargs)) or MockAWSProvider()),
            executor_factory=lambda node, task_: RecordingExecutor())
        orchestrator.launch(task)
        # The free k8s node wins; it takes a namespace, not a region.
        self.assertEqual(captured, [("k8s", {})])


class TestEnvExpansion(unittest.TestCase):
    """Secrets are referenced by name, never committed."""

    def test_env_reference_resolves_from_the_shell(self) -> None:
        with patch.dict(os.environ, {"LIGHTOS_TEST_SECRET": "s3cret"}):
            task = Task.from_dict({
                "name": "t", "run": "echo hi",
                "envs": {"TOKEN": "${LIGHTOS_TEST_SECRET}"}})
        self.assertEqual(task.envs["TOKEN"], "s3cret")

    def test_unset_env_reference_is_rejected_up_front(self) -> None:
        os.environ.pop("LIGHTOS_DEFINITELY_UNSET", None)
        with self.assertRaises(TaskError) as ctx:
            Task.from_dict({"name": "t", "run": "echo hi",
                            "envs": {"TOKEN": "${LIGHTOS_DEFINITELY_UNSET}"}})
        self.assertIn("LIGHTOS_DEFINITELY_UNSET", str(ctx.exception))

    def test_literal_values_pass_through(self) -> None:
        task = Task.from_dict({"name": "t", "run": "echo hi",
                               "envs": {"MODE": "integration", "RETRIES": 3}})
        self.assertEqual(task.envs, {"MODE": "integration", "RETRIES": 3})

    def test_production_blueprint_parses(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with patch.dict(os.environ, {"LIGHTOS_ACME_EMAIL": "ops@example.com",
                                     "LIGHTOS_API_TOKEN": "token"}):
            task = Task.from_yaml(os.path.join(root, "deploy",
                                               "production.yaml"))
        self.assertIsNone(task.accelerator)
        self.assertEqual(task.gpu_count, 0)
        self.assertEqual(task.remote_workdir, "/opt/lightos")
        # The production host must survive the deploy job.
        self.assertFalse(task.autostop)
        self.assertEqual(task.envs["LIGHTOS_DOMAIN"], "agentic.lightos.sh")
        self.assertEqual(os.path.realpath(task.workdir),
                         os.path.realpath(root))

    def test_production_blueprint_resolves_to_cpu_capacity(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with patch.dict(os.environ, {"LIGHTOS_ACME_EMAIL": "ops@example.com",
                                     "LIGHTOS_API_TOKEN": "token"}):
            task = Task.from_yaml(os.path.join(root, "deploy",
                                               "production.yaml"))
        offer = CostCatalog().find_cheapest_match(
            task.accelerator, task.gpu_count, task.clouds, task.region)
        self.assertEqual(offer["gpus"], 0)
        self.assertEqual(offer["cloud"], "aws")


class TestAPIServer(unittest.TestCase):
    """The HTTP surface behind agentic.lightos.sh."""

    @staticmethod
    def _request(server, method: str, path: str, body: bytes = None,
                 headers: Dict[str, str] = None):
        import http.client

        host, port = server.server_address[:2]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            payload = response.read()
            return response.status, json.loads(payload or b"{}")
        finally:
            conn.close()

    def _serve(self, api_token: str = None):
        """Start the real server on an ephemeral port for the test's lifetime."""
        import threading

        from lightos.server import build_server

        provider = MockAWSProvider()
        executor = RecordingExecutor()
        orchestrator = build_orchestrator(executor, provider)
        server = build_server(host="127.0.0.1", port=0,
                              catalog=orchestrator.catalog,
                              orchestrator=orchestrator,
                              api_token=api_token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # Cleanups run LIFO, so register them in reverse of the order we
        # want: stop accepting, join the loop, then release the socket.
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        return server, provider, executor

    def test_healthz(self) -> None:
        server, _, _ = self._serve()
        status, body = self._request(server, "GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_index_advertises_routes(self) -> None:
        server, _, _ = self._serve()
        status, body = self._request(server, "GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(body["service"], "lightos")
        self.assertIn("aws", body["clouds"])
        self.assertFalse(body["job_submission_enabled"])

    def test_catalog_query(self) -> None:
        server, _, _ = self._serve()
        status, body = self._request(
            server, "GET", "/v1/catalog?accelerator=A10G&gpus=1&hours=2")
        self.assertEqual(status, 200)
        self.assertEqual(body["cheapest"]["instance_type"], "g5.xlarge")
        self.assertAlmostEqual(body["matches"][0]["projected_cost"], 2.012,
                               places=3)

    def test_catalog_rejects_non_numeric_gpus(self) -> None:
        server, _, _ = self._serve()
        status, body = self._request(server, "GET",
                                     "/v1/catalog?accelerator=A10G&gpus=lots")
        self.assertEqual(status, 400)
        self.assertIn("must be numbers", body["error"])

    def test_estimate_prices_a_blueprint_without_provisioning(self) -> None:
        server, provider, _ = self._serve()
        status, body = self._request(server, "POST", "/v1/estimate",
                                     body=SAMPLE_YAML.encode("utf-8"))
        self.assertEqual(status, 200)
        self.assertEqual(body["selected"]["instance_type"], "g5.xlarge")
        self.assertFalse(body["provisioned"])
        self.assertEqual(provider.launched, [])

    def test_estimate_accepts_json_bodies_too(self) -> None:
        server, _, _ = self._serve()
        payload = json.dumps({"name": "json-task", "run": "echo hi",
                              "resources": {"accelerators": "L4"}})
        status, body = self._request(server, "POST", "/v1/estimate",
                                     body=payload.encode("utf-8"))
        self.assertEqual(status, 200)
        self.assertEqual(body["selected"]["accelerator"], "L4")

    def test_estimate_rejects_invalid_blueprint(self) -> None:
        server, _, _ = self._serve()
        status, _ = self._request(server, "POST", "/v1/estimate",
                                  body=b"name: broken")
        self.assertEqual(status, 422)

    def test_estimate_reports_unsatisfiable_hardware(self) -> None:
        server, _, _ = self._serve()
        payload = json.dumps({"name": "t", "run": "echo hi",
                              "resources": {"accelerators": "H100"}})
        status, body = self._request(server, "POST", "/v1/estimate",
                                     body=payload.encode("utf-8"))
        self.assertEqual(status, 409)
        self.assertIn("H100", body["error"])

    def test_job_submission_is_disabled_without_a_token(self) -> None:
        server, provider, _ = self._serve(api_token=None)
        status, body = self._request(server, "POST", "/v1/jobs",
                                     body=SAMPLE_YAML.encode("utf-8"))
        self.assertEqual(status, 403)
        self.assertIn("disabled", body["error"])
        self.assertEqual(provider.launched, [])

    def test_job_submission_requires_the_right_token(self) -> None:
        server, provider, _ = self._serve(api_token="correct-horse")
        status, _ = self._request(
            server, "POST", "/v1/jobs", body=SAMPLE_YAML.encode("utf-8"),
            headers={"Authorization": "Bearer wrong-token"})
        self.assertEqual(status, 401)
        self.assertEqual(provider.launched, [])

    def test_job_submission_runs_the_pipeline_with_a_valid_token(self) -> None:
        server, provider, executor = self._serve(api_token="correct-horse")
        status, body = self._request(
            server, "POST", "/v1/jobs", body=SAMPLE_YAML.encode("utf-8"),
            headers={"Authorization": "Bearer correct-horse"})
        self.assertEqual(status, 200)
        self.assertTrue(body["succeeded"])
        self.assertEqual(provider.launched, ["g5.xlarge"])
        self.assertEqual(provider.terminated, ["i-mockinstance1234567890"])
        self.assertEqual(executor.script.strip().splitlines()[0],
                         'echo "Initializing dependencies..."')

    def test_oversized_body_is_refused(self) -> None:
        server, _, _ = self._serve()
        huge = b"name: t\nrun: echo hi\n# " + b"x" * (300 * 1024)
        status, _ = self._request(server, "POST", "/v1/estimate", body=huge)
        self.assertEqual(status, 413)

    def test_unknown_route_404s(self) -> None:
        server, _, _ = self._serve()
        status, _ = self._request(server, "GET", "/v1/nope")
        self.assertEqual(status, 404)


class TestWorkspaceSyncFiltering(unittest.TestCase):
    """Credentials and caches must never leave the machine."""

    def test_secrets_and_caches_are_excluded(self) -> None:
        for path in (".git/config", "__pycache__/catalog.cpython-311.pyc",
                     ".env", "keys/id_rsa", "deploy.pem",
                     "node_modules/left-pad/index.js", ".venv/bin/python"):
            self.assertTrue(_is_excluded(path, DEFAULT_EXCLUDES),
                            f"{path} should not be synced")

    def test_project_sources_are_included(self) -> None:
        for path in ("main.py", "lightos/catalog.py", "task_config.yaml",
                     "tests/test_integration.py"):
            self.assertFalse(_is_excluded(path, DEFAULT_EXCLUDES),
                             f"{path} should be synced")


class TestSkyOrchestratorIntegration(unittest.TestCase):
    """Verifies data flows from YAML parser to catalog matchmaker to executor."""

    def setUp(self) -> None:
        self.config = yaml.safe_load(SAMPLE_YAML)
        self.catalog = CostCatalog()

    def test_end_to_end_orchestration_flow(self) -> None:
        print("\n=======================================================")
        print("STARTING SYSTEM END-TO-END INTEGRATION TEST")
        print("=======================================================")

        # ---- STAGE 1: blueprint parsing ----
        print("\n[Stage 1] Parsing user blueprint...")
        task = Task.from_dict(self.config)
        self.assertEqual(task.name, "deep-learning-integration-test")
        print(f"  Task '{task.name}' ingested.")

        # ---- STAGE 2: cost-catalog optimization ----
        print("\n[Stage 2] Matching specs against the cloud catalog...")
        options = self.catalog.find_matches(task.accelerator, task.gpu_count)
        selected_node = options[0]
        self.assertEqual(selected_node["cloud"], "aws")
        self.assertEqual(selected_node["instance_type"], "g5.xlarge")
        print(f"  Optimizer selected {selected_node['cloud'].upper()} "
              f"{selected_node['instance_type']} "
              f"(${selected_node['hourly_cost']}/hr)")

        # ---- STAGES 3-5: provision, sync, execute, tear down ----
        print("\n[Stage 3-5] Running the pipeline against mock cloud...")
        provider = MockAWSProvider()
        executor = RecordingExecutor()
        result = build_orchestrator(executor, provider, self.catalog).launch(task)

        # Provisioning reached the driver with the optimizer's choice.
        self.assertEqual(provider.launched, ["g5.xlarge"])
        self.assertEqual(result.instance_id, "i-mockinstance1234567890")

        # Sync and execution received the exact blueprint values.
        self.assertTrue(executor.connected)
        self.assertEqual(executor.synced_from, task.workdir)
        self.assertEqual(executor.synced_to, "/tmp/lightos_workdir")
        self.assertEqual(executor.script, task.payload())
        self.assertEqual(executor.env, {"LIGHTOS_RUN_MODE": "integration"})
        self.assertEqual(result.files_synced, 7)

        # Teardown ran and the job is reported as successful.
        self.assertTrue(executor.disconnected)
        self.assertEqual(provider.terminated, ["i-mockinstance1234567890"])
        self.assertTrue(result.terminated)
        self.assertTrue(result.succeeded)
        self.assertEqual(result.exit_code, 0)

        print("\n=======================================================")
        print("INTEGRATION TEST PASSED")
        print("=======================================================\n")

    def test_failed_payload_is_reported_and_still_torn_down(self) -> None:
        task = Task.from_dict(self.config)
        provider = MockAWSProvider()
        executor = RecordingExecutor(exit_code=1)
        result = build_orchestrator(executor, provider, self.catalog).launch(task)

        self.assertFalse(result.succeeded)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(provider.terminated, ["i-mockinstance1234567890"])

    def test_crash_mid_job_still_releases_the_cluster(self) -> None:
        """The expensive failure mode: an exception leaving a GPU node alive."""
        task = Task.from_dict(self.config)
        provider = MockAWSProvider()
        executor = RecordingExecutor(fail_on_run=True)
        result = build_orchestrator(executor, provider, self.catalog).launch(task)

        self.assertFalse(result.succeeded)
        self.assertIn("ssh channel died", result.error)
        self.assertEqual(provider.terminated, ["i-mockinstance1234567890"])
        self.assertTrue(result.terminated)
        self.assertTrue(executor.disconnected)

    def test_no_autostop_leaves_the_cluster_running(self) -> None:
        task = Task.from_dict(self.config)
        task.autostop = False
        provider = MockAWSProvider()
        result = build_orchestrator(RecordingExecutor(), provider,
                                    self.catalog).launch(task)

        self.assertEqual(provider.terminated, [])
        self.assertFalse(result.terminated)
        self.assertTrue(result.succeeded)

    def test_dry_run_provisions_nothing(self) -> None:
        task = Task.from_dict(self.config)
        provider = MockAWSProvider()
        result = build_orchestrator(RecordingExecutor(), provider,
                                    self.catalog).launch(task, dry_run=True)

        self.assertEqual(provider.launched, [])
        self.assertIsNone(result.exit_code)
        self.assertEqual(result.offer["instance_type"], "g5.xlarge")

    def test_cloud_constraint_routes_to_a_different_backend(self) -> None:
        task = Task.from_dict(self.config)
        task.accelerator = "A100"
        task.gpu_count = 1
        task.cloud = "gcp"
        requested: List[str] = []

        orchestrator = Orchestrator(
            catalog=self.catalog,
            provider_factory=lambda cloud, kwargs: (
                requested.append(cloud) or MockAWSProvider()),
            executor_factory=lambda node, task_: RecordingExecutor())
        orchestrator.launch(task)
        self.assertEqual(requested, ["gcp"])

    def test_unsatisfiable_hardware_fails_before_provisioning(self) -> None:
        task = Task.from_dict(self.config)
        task.accelerator = "H100"
        provider = MockAWSProvider()
        with self.assertRaises(CatalogError):
            build_orchestrator(RecordingExecutor(), provider,
                              self.catalog).launch(task)
        self.assertEqual(provider.launched, [])

    def test_result_summary_reports_cost_and_teardown(self) -> None:
        task = Task.from_dict(self.config)
        result = build_orchestrator(RecordingExecutor(), MockAWSProvider(),
                                    self.catalog).launch(task)
        summary = result.summary()
        self.assertIn("SUCCEEDED", summary)
        self.assertIn("g5.xlarge", summary)
        self.assertIn("Teardown   : done", summary)


class TestRemoteExecutorContract(unittest.TestCase):
    """The SSH executor's wiring, with paramiko itself mocked out."""

    @staticmethod
    def _mock_channel(chunks: List[bytes] = None):
        """A channel that yields `chunks` then reports a clean exit."""
        pending = list(chunks or [])
        channel = MagicMock()
        channel.recv_ready.side_effect = lambda: bool(pending)
        channel.recv.side_effect = lambda n: pending.pop(0)
        channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.side_effect = lambda: not pending
        channel.recv_exit_status.return_value = 0
        return channel

    def _run(self, channel, **executor_kwargs):
        from lightos.execution import RemoteJobExecutor

        sink = io.StringIO()
        executor = RemoteJobExecutor(host_ip="203.0.113.10", stdout=sink,
                                     **executor_kwargs)
        transport = MagicMock()
        transport.open_session.return_value = channel
        client = MagicMock()
        client.get_transport.return_value = transport
        with patch.object(RemoteJobExecutor, "ssh_client", client):
            exit_code = executor.execute_and_stream_logs(
                "python3 train.py", run_directory="/tmp/lightos_workdir",
                env={"TOKEN": "abc"})
        return exit_code, channel, sink.getvalue()

    def test_payload_is_wrapped_with_workdir_and_env(self) -> None:
        exit_code, channel, _ = self._run(self._mock_channel(),
                                          use_select=False)
        self.assertEqual(exit_code, 0)
        channel.get_pty.assert_called_once()
        command = channel.exec_command.call_args[0][0]
        self.assertIn("cd /tmp/lightos_workdir", command)
        self.assertIn("export TOKEN='abc'", command)
        self.assertIn("python3 train.py", command)
        self.assertIn("pipefail", command)

    def test_output_is_streamed_on_the_polling_path(self) -> None:
        """The path taken on Windows, where select() rejects non-sockets."""
        channel = self._mock_channel([b"epoch 1\n", b"epoch 2\n"])
        exit_code, _, output = self._run(channel, use_select=False)
        self.assertEqual(exit_code, 0)
        self.assertIn("epoch 1", output)
        self.assertIn("epoch 2", output)

    def test_output_is_streamed_on_the_select_path(self) -> None:
        """The path taken on POSIX, which the Linux CI runners exercise."""
        channel = self._mock_channel([b"epoch 1\n", b"epoch 2\n"])
        # Stand in for the real socket select() would need.
        with patch("select.select", return_value=([channel], [], [])):
            exit_code, _, output = self._run(channel, use_select=True)
        self.assertEqual(exit_code, 0)
        self.assertIn("epoch 1", output)
        self.assertIn("epoch 2", output)

    def test_select_is_used_only_on_posix(self) -> None:
        """Guards against the branch being chosen by accident."""
        from lightos.execution import RemoteJobExecutor

        with patch.object(os, "name", "posix"):
            self.assertTrue(RemoteJobExecutor(host_ip="h").use_select)
        with patch.object(os, "name", "nt"):
            self.assertFalse(RemoteJobExecutor(host_ip="h").use_select)

    def test_stderr_is_streamed_too(self) -> None:
        channel = self._mock_channel()
        errors = [b"traceback\n"]
        channel.recv_stderr_ready.side_effect = lambda: bool(errors)
        channel.recv_stderr.side_effect = lambda n: errors.pop(0)
        channel.exit_status_ready.side_effect = lambda: not errors
        exit_code, _, output = self._run(channel, use_select=False)
        self.assertEqual(exit_code, 0)
        self.assertIn("traceback", output)

    def test_missing_ssh_key_is_reported_clearly(self) -> None:
        from lightos.execution import ExecutionError, RemoteJobExecutor

        executor = RemoteJobExecutor(host_ip="203.0.113.10",
                                     ssh_key_path="/nonexistent/key",
                                     stdout=io.StringIO())
        with self.assertRaises(ExecutionError) as ctx:
            executor.connect()
        self.assertIn("SSH key not found", str(ctx.exception))


class TestCLI(unittest.TestCase):
    """The user-facing surface."""

    def test_catalog_command_lists_matches(self) -> None:
        from lightos.cli import main as cli_main

        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = cli_main(["catalog", "-a", "A10G", "-n", "1"])
        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("g5.xlarge", output)
        self.assertIn("Cheapest: AWS g5.xlarge", output)

    def test_catalog_command_reports_no_match(self) -> None:
        from lightos.cli import main as cli_main

        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = cli_main(["catalog", "-a", "H100"])
        self.assertEqual(code, 1)
        self.assertIn("No capacity matches", buffer.getvalue())

    def test_validate_command_accepts_the_shipped_blueprint(self) -> None:
        from lightos.cli import main as cli_main

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = cli_main(["validate",
                             os.path.join(root, "task_config.yaml")])
        self.assertEqual(code, 0)
        self.assertIn("Blueprint is valid", buffer.getvalue())

    def test_launch_dry_run_provisions_nothing(self) -> None:
        from lightos.cli import main as cli_main

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            code = cli_main(["launch", os.path.join(root, "task_config.yaml"),
                             "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("dry-run", buffer.getvalue())

    def test_bad_blueprint_path_exits_cleanly(self) -> None:
        from lightos.cli import main as cli_main

        buffer = io.StringIO()
        with patch("sys.stderr", buffer):
            code = cli_main(["validate", "/nonexistent/task.yaml"])
        self.assertEqual(code, 2)
        self.assertIn("not found", buffer.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
