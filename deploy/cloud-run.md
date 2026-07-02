# Deploy & Scale Path (designed-for, not operated)

> These are **config + docs** (the 🔵 tier). Nothing here runs in production today; it documents how the system is built to deploy and scale.

## Cloud Run (documented deploy target)

The app is a single container (see `Dockerfile`) exposing the FastAPI factory `employee_agent.api.app:create_app`.

```bash
# Store the LLM key in Secret Manager, not env files.
gcloud secrets create gemini-api-key --replication-policy=automatic
printf '%s' "$GEMINI_API_KEY" | gcloud secrets versions add gemini-api-key --data-file=-

gcloud run deploy employee-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated=false \
  --set-env-vars PROVIDER=gemini,ENABLE_FAILOVER=false,API_KEYS=<rotate-me> \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

**Provider swap → Vertex AI:** because every LLM/embedding call goes through the `providers/` abstraction, moving from the AI Studio Gemini API to **Vertex AI** is a provider change (new `Provider` implementation + `PROVIDER=vertex`), with **no changes** to the graph, API, or UI.

**State:** the SQLite checkpointer and Chroma index live on the container's writable path. For Cloud Run (ephemeral disk), mount a persistent volume (or move the checkpointer to Cloud SQL / a managed vector store) so job state survives instance recycling.

## Scale path (if usage 10×'s)

- **Horizontal scaling:** the container is stateless per request (state is in the checkpointer), so scale out on Cloud Run or **GKE**.
- **Global routing:** a multi-cluster **GKE Inference Gateway** routes requests to model workloads across regions.
- **API management (Apigee):** enforce **quota** + **Spike-Arrest** policies to cap unrestricted resource consumption (and the LLM bill) before requests reach the app.
- **WAF (Cloud Armor):** L3/L4 DDoS protection in front of the load balancer.

## App-level guards already in place

- **Runaway-loop guard:** the verifier's retry loop is bounded by `MAX_RETRIES`; it always terminates at the human gate rather than looping forever.
- **Rate limiting:** slowapi per-key limits on `POST /jobs`.
- **Tenant isolation (BOLA):** each job is owned by its creating API key; cross-owner access returns 404.
- **SSRF-via-MCP mitigation:** the Analyst may only call tools in the role's `tool_allowlist`.
