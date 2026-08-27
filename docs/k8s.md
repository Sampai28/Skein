# Running on k3d

Manifests only — none of this has been applied.

## Sequence

```bash
k3d cluster create --config k8s/k3d-config.yaml

docker build -f docker/Dockerfile -t skein:local .
k3d image import skein:local -c skein

kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml

kubectl rollout status deployment/skein
kubectl port-forward svc/skein 8000:8000
```

`k3d image import` rather than a registry push: k3d nodes cannot see the host
Docker daemon's images, so a locally built image has to be loaded into the
cluster explicitly. Skipping it gives `ErrImageNeverPull`, because
`imagePullPolicy: IfNotPresent` with no registry means the node has nowhere to
fetch from. The k3d config also creates a local registry on port 5001 if you
would rather push than import — faster on repeat builds, since only changed
layers move.

Ollama is not in these manifests. Either run it outside the cluster and point
`SKEIN_OLLAMA_URL` at the host, or set `SKEIN_STUB_LLM=true` in the ConfigMap.

## Shutdown timing

The numbers in `deployment.yaml` are chosen together and break if changed
independently.

```
preStop sleep             5s
SKEIN_DRAIN_TIMEOUT_S    25s
SKEIN_CANCEL_TIMEOUT_S    5s
                        ----
                         35s   < terminationGracePeriodSeconds (40)
```

On pod deletion, endpoint removal and the `preStop` hook happen **concurrently**,
then SIGTERM is sent, then SIGKILL at the end of the grace period. The preStop
sleep exists because endpoint removal is eventually consistent across kube-proxy
on every node — without it, requests keep arriving for a second or two after the
app has already started refusing them.

If the grace period were shorter than the sum, SIGKILL would land partway
through cancellation and skip the cleanup the whole runtime is built around.
Raising `SKEIN_DRAIN_TIMEOUT_S` means raising `terminationGracePeriodSeconds`
too.

Watching it work:

```bash
kubectl delete pod -l app.kubernetes.io/name=skein --grace-period=40
kubectl logs -l app.kubernetes.io/name=skein -f --previous
```

Expect the readiness probe to start failing immediately (draining), then the
drain log line, then the process exiting before the grace period elapses.

## Probes

`livenessProbe` hits `/healthz` and deliberately does not check Ollama. A
liveness probe that fails when a dependency is down restarts a healthy pod, and
a restart loop caused by someone else's outage is worse than degraded service.

`readinessProbe` hits `/readyz`, which reports false while draining and while
the ingress queue is full. `failureThreshold: 1` so a draining pod leaves the
endpoint list on the first check rather than fifteen seconds later.

`startupProbe` gives 60s for the app to come up before liveness starts, so a
slow first start is not mistaken for a hang.

## Scaling

The HPA has two metrics. CPU is a weak signal for an I/O-bound async service —
a pod saturated on concurrent LLM calls can sit at 15% CPU — so it is there as a
floor. The metric that reflects real load is `skein_queue_depth`, which requires
a Prometheus adapter to expose as a custom metric. Without one, that entry is
ignored and CPU alone drives scaling; nothing breaks, it just scales late.

Scale-down is deliberately slow (300s stabilisation, one pod per minute): each
removed pod drains for up to 35 seconds, and flapping into a traffic dip costs
more availability than an idle replica costs money.

## The affinity caveat

`Service` uses `sessionAffinity: ClientIP`. The run store is in memory, so a run
submitted to one replica is invisible to the others — a `GET /runs/{id}` that
lands elsewhere returns 404, and the WebSocket stream has nothing to stream.

This is a workaround, not a design. A shared run store would remove the need for
it, and until then the honest limitation is that Skein scales horizontally for
throughput but not for availability of any individual run: if the pod owning a
run dies, that run is gone.

## Teardown

```bash
kubectl delete -f k8s/hpa.yaml -f k8s/service.yaml -f k8s/deployment.yaml -f k8s/configmap.yaml
k3d cluster delete skein
```
