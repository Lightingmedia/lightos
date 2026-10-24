# LightOS

A cost-optimized multi-cloud job orchestrator. You describe the hardware you
need in YAML; LightOS finds the cheapest cluster that satisfies it across AWS,
GCP and Kubernetes, provisions it, syncs your code, streams the logs back, and
tears the cluster down when the job exits.

```
┌─────────────────────────────────────────────────────────────┐
│                      EXECUTION LOOP                         │
├─────────────────────────────────────────────────────────────┤
│ 1. READ:     parse the task blueprint (YAML)                │
│ 2. EVALUATE: match the request against the cost catalog     │
│ 3. ACT:      provision, sync, execute, tear down            │
└─────────────────────────────────────────────────────────────┘
```

## Install

```bash
pip install -r requirements.txt
```

`boto3` is included for the AWS backend. Add `google-cloud-compute` or
`kubernetes` if you provision against GCP or K8s.

## Quick start

```bash
python main.py catalog -a A100 -n 8     # what capacity exists, and what it costs
python main.py validate task_config.yaml # check a blueprint
python main.py --dry-run                 # pick a cluster, provision nothing
python main.py                           # run task_config.yaml for real
```

`python main.py` with no arguments launches `task_config.yaml`. Everything
else is a subcommand: `launch`, `catalog`, `validate`, `serve`.

## Blueprints

```yaml
name: train-llama
workdir: .                  # synced to the node; relative to this file
resources:
  accelerators: A100        # or A100:8 as shorthand for gpu_count: 8
  gpu_count: 8
  cloud: gcp                # optional hard constraint
  region: us-central1       # optional hard constraint
envs:
  HF_TOKEN: ${HF_TOKEN}     # resolved from your shell, never committed
setup: |
  pip install torch
run: |
  python train.py
autostop: true              # terminate as soon as `run` exits
```

An `envs` value of the form `${VAR}` is read from the launching shell and
fails loudly at parse time if the variable is unset, rather than exporting an
empty string on the node.

## Architecture

| Module | Responsibility |
| --- | --- |
| [`lightos/catalog.py`](lightos/catalog.py) | Price database and the matchmaker that picks the cheapest satisfying offer |
| [`lightos/task.py`](lightos/task.py) | Blueprint parsing, validation and env resolution |
| [`lightos/backends/`](lightos/backends/base.py) | `CloudProvider` interface plus AWS, GCP and Kubernetes drivers |
| [`lightos/execution.py`](lightos/execution.py) | SSH/SFTP sync and live log streaming; pod-exec equivalent for K8s |
| [`lightos/orchestrator.py`](lightos/orchestrator.py) | The pipeline that wires the above together, teardown included |
| [`lightos/cli.py`](lightos/cli.py) | Command-line interface |
| [`lightos/server.py`](lightos/server.py) | HTTP API (what `agentic.lightos.sh` serves) |

Three design points worth knowing:

* **Teardown is in a `finally`.** An orphaned GPU node is the one failure mode
  that keeps costing money after the process exits, so the cluster is released
  even when the payload crashes or the SSH channel dies. If termination itself
  fails, the summary says `Teardown: NOT DONE` and names the instance.
* **Backends are imported lazily.** Running an AWS job does not require the
  GCP or Kubernetes client libraries to be installed.
* **The sync excludes secrets and caches.** `.git`, `__pycache__`, `.venv`,
  `.env`, `*.pem` and `id_rsa` never leave the machine
  ([`DEFAULT_EXCLUDES`](lightos/execution.py)).

`rsync` is used for workspace sync when it is on `PATH` (much faster on
incremental code changes) with an SFTP fallback, so the path also works from
Windows. Log streaming uses `select` on POSIX and a short polling loop
elsewhere, because `select` only accepts real sockets on Windows.

## Tests

```bash
python -m unittest discover tests
```

53 tests, no cloud credentials and no spend: the EC2 and SSH boundaries are
mocked, while the catalog, parser, orchestrator and HTTP API under test are
the real ones. The API tests start the actual server on an ephemeral port and
drive it over HTTP.

The suite was mutation-checked — reversing the catalog's cost sort fails 8
tests, and removing the teardown call fails 3 — so it does detect the defects
it claims to cover.

## HTTP API

```bash
python main.py serve --port 8080
```

| Route | Auth | Purpose |
| --- | --- | --- |
| `GET /healthz` | none | liveness probe |
| `GET /v1/catalog?accelerator=A100&gpus=8&hours=3` | none | query capacity and projected cost |
| `POST /v1/estimate` | none | price a blueprint, provision nothing |
| `POST /v1/jobs` | bearer token | launch a blueprint |

`POST /v1/jobs` runs caller-supplied shell on provisioned hardware — it is
arbitrary code execution by design. It returns `403` until `LIGHTOS_API_TOKEN`
is set, compares the token in constant time, and the bundled Caddy config
blocks it at the edge as well. Expose it only deliberately, and generate the
token with `python -c "from lightos.server import generate_token; print(generate_token())"`.

## Deploying to production

The deploy is itself a LightOS job: [`deploy/production.yaml`](deploy/production.yaml)
provisions the cheapest CPU node, syncs this repo to `/opt/lightos`, installs
Docker, and brings up the API behind Caddy, which obtains the TLS certificate
over ACME. No key material is stored in the repo.

```bash
# 1. Credentials and configuration
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
export LIGHTOS_AWS_AMI=ami-...            # Ubuntu 22.04 in your region
export LIGHTOS_AWS_KEY_NAME=my-keypair    # matching ~/.ssh/id_rsa
export LIGHTOS_ACME_EMAIL=ops@example.com
export LIGHTOS_API_TOKEN=$(python -c "from lightos.server import generate_token; print(generate_token())")

# 2. Price it before spending anything
python main.py deploy/production.yaml --dry-run

# 3. Deploy
python main.py deploy/production.yaml
```

Two things must be true before step 3, and neither is something the
orchestrator can do for you:

1. **Networking.** The node's security group must allow inbound 80 and 443
   (ACME validation needs both) and 22 for the sync.
2. **DNS.** `agentic.lightos.sh` must have an A record pointing at the node's
   public address. Certificate issuance fails until it resolves. Run step 3,
   note the address in the provisioning output, create the record, then re-run
   — or allocate an Elastic IP first and point DNS at it ahead of time.

`autostop` is `false` in that blueprint, so the production host outlives the
deploy job. To update a running deployment, re-run the same command: the
compose build is incremental and Caddy keeps its certificate in a named
volume.

### Running the stack directly

If you already have a host, skip the orchestrator:

```bash
export LIGHTOS_ACME_EMAIL=ops@example.com
export LIGHTOS_DOMAIN=agentic.lightos.sh
docker compose -f deploy/docker-compose.yml up -d --build
```

The API container runs unprivileged, read-only, with `no-new-privileges`, and
is not published to the host — only Caddy reaches it over the internal
network.
