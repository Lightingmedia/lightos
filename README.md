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
| [`lightos/server.py`](lightos/server.py) | HTTP API the frontend calls |

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

### Calling it from a browser frontend

LightOS is the backend. A frontend served from a different hostname is a
different *origin*, and a browser blocks cross-origin reads unless the server
opts in, so list the frontend's origin:

```bash
export LIGHTOS_ALLOWED_ORIGINS="https://agentic.lightos.sh,http://localhost:5173"
```

Comma-separated; include the scheme and no trailing slash (one is stripped if
you add it). Unset means no cross-origin access at all — the request succeeds
server-side but the browser discards the response, which looks like a CORS
error in the console rather than a server failure. `GET /` reports the
configured origins, and so does the startup log, so a frontend that cannot
reach the API is quick to diagnose.

`*` is accepted and echoes the caller's origin. Only use it while every
exposed endpoint is read-only.

**Keep the API token out of the frontend.** Anything a browser holds is
public. The frontend should call `GET /v1/catalog` and `POST /v1/estimate`,
which need no auth; `POST /v1/jobs` should be driven from a server you
control, not from the page.

## Deploying to production

The deploy is itself a LightOS job: [`deploy/production.yaml`](deploy/production.yaml)
provisions the cheapest CPU node, syncs this repo to `/opt/lightos`, installs
Docker, and brings up the API behind Caddy, which obtains the TLS certificate
over ACME. No key material is stored in the repo.

This deploys the **backend only**. The frontend is a separate app that calls
this API, so the API gets its own hostname — `api.agentic.lightos.sh` in the
blueprint. Pointing `LIGHTOS_DOMAIN` at the frontend's hostname would make
Caddy claim that name's certificate and take the frontend offline, so CI
fails the build if the two are ever the same.

```bash
# 1. Credentials and configuration
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
export LIGHTOS_AWS_AMI=ami-...            # Ubuntu 22.04 in your region
export LIGHTOS_AWS_KEY_NAME=my-keypair    # registered in EC2
export LIGHTOS_SSH_KEY=~/.ssh/my-keypair  # the matching private key
export LIGHTOS_ACME_EMAIL=ops@example.com
export LIGHTOS_API_TOKEN=$(python -c "from lightos.server import generate_token; print(generate_token())")

# 2. Check the blueprint, including that the SSH key resolves
python main.py validate deploy/production.yaml

# 3. Price it before spending anything
python main.py deploy/production.yaml --dry-run

# 4. Deploy
python main.py deploy/production.yaml
```

Three things must be true before step 4, and none is something the
orchestrator can do for you:

1. **An SSH keypair that matches.** `LIGHTOS_AWS_KEY_NAME` is the keypair
   registered in EC2; `LIGHTOS_SSH_KEY` is the private half on your machine.
   The default is `~/.ssh/id_rsa`, which is only a convention — if your key
   is named anything else, set `LIGHTOS_SSH_KEY` or `ssh_key_path:` in the
   blueprint. `validate` exits non-zero and names the keys you do have when
   this is wrong, so you find out before a node is provisioned and billed.
2. **Networking.** The node's security group must allow inbound 80 and 443
   (ACME validation needs both) and 22 for the sync.
3. **DNS.** The API hostname needs an A record pointing at the node's public
   address, created wherever the zone is hosted — your registrar's panel, not
   AWS, unless the zone is on Route 53. Certificate issuance fails until it
   resolves. Deploy, note the address in the provisioning output, create the
   record, then re-run — or allocate an Elastic IP first and point DNS at it
   ahead of time. If the record sits behind a proxy that terminates TLS
   (Cloudflare's orange cloud), set it to DNS-only for this hostname or
   Caddy's certificate will never be the one served.

`autostop` is `false` in that blueprint, so the production host outlives the
deploy job. To update a running deployment, re-run the same command: the
compose build is incremental and Caddy keeps its certificate in a named
volume.

### Running the stack directly

If you already have a host, skip the orchestrator:

```bash
export LIGHTOS_ACME_EMAIL=ops@example.com
export LIGHTOS_DOMAIN=api.agentic.lightos.sh          # the API's own host
export LIGHTOS_ALLOWED_ORIGINS=https://agentic.lightos.sh   # the frontend
docker compose -f deploy/docker-compose.yml up -d --build
```

The API container runs unprivileged, read-only, with `no-new-privileges`, and
is not published to the host — only Caddy reaches it over the internal
network.
