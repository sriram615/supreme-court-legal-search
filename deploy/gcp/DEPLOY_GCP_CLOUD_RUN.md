# Deploying to GCP Cloud Run -- the fast path

This is genuinely simpler than the AWS route: no IAM role, no container
registry to create by hand, no separate "Function URL" step, and -- the
biggest win given today -- **no local Docker needed**. `gcloud run deploy
--source .` sends your code to Cloud Build, which builds the image in the
cloud and pushes it, then Cloud Run deploys it and hands you a public HTTPS
URL, all from one command.

## 1. Install the gcloud CLI

```bash
brew install --cask google-cloud-sdk
```

If that hits any brew tap/trust weirdness again, use the official installer
instead (no brew involved): https://cloud.google.com/sdk/docs/install -- it's
a plain `.pkg`/tarball for macOS.

## 2. Log in and set a project

```bash
gcloud init
```

This opens a browser login, then asks you to pick or create a GCP project.
Create a new one if you don't already have one (e.g. `legal-search-demo`).

**You'll be asked to enable billing on the project.** Cloud Run's Always
Free tier is real and ongoing, but GCP still requires a valid card on file
before it'll let you deploy anything, same as AWS did. Normal usage for a
portfolio demo will not generate a charge, but set a budget alert anyway
(GCP Console -> Billing -> Budgets & alerts -- takes 2 minutes).

## 3. Run the deploy

From the repo root:
```bash
chmod +x deploy/gcp/deploy.sh
./deploy/gcp/deploy.sh
```

This enables the required APIs (one-time), stages the Dockerfile, builds in
the cloud, and deploys with `--min-instances 0` (scale-to-zero, free when
idle) and `--max-instances 3` (a cheap safety cap against a runaway bill).
It prints your public URL at the end.

## 4. Attach the Groq key as a secret

The script prints these two commands at the end -- run them with your
*current, non-exposed* Groq key (not the one that got pasted into this
conversation earlier):
```bash
echo -n 'your-new-groq-key' | gcloud secrets create groq-api-key --data-file=-
gcloud run services update legal-search-engine --region us-central1 --set-secrets=GROQ_API_KEY=groq-api-key:latest
```

## 5. Verify before it goes on the resume

```bash
curl <the-URL-from-step-3>
```
should return the frontend HTML. Then open it in a browser, run a real
query through the console, and check `/api/v1/agentic-analyze` specifically
since that's the feature this whole deployment is meant to prove is real.
Time the first request after a few idle minutes vs. a second one right
after -- that gap is the cold-start cost of `--min-instances 0`; know what
it is before someone else hits the link cold.

## 6. If something fails partway

`gcloud run deploy` output tells you which stage failed (Cloud Build vs.
the deploy itself). The most likely failure for this specific app is
running out of memory during the build or at cold start if `LAWdata_Corpus_2024/`
is larger than expected -- if so, bump `MEMORY` in `deploy.sh` (e.g. to
`4Gi`) and rerun; redeploying is the same one command, not a rebuild-from-
scratch ordeal like the AWS path.

## What's not verified

Same caveat as the AWS files: this was written against Cloud Run's and
Cloud Build's documented behavior and the script's syntax was checked
(`bash -n`), but it hasn't been run end-to-end -- this session has no GCP
account. The first real run is where you'll find out if memory needs
adjusting or the corpus data needs a different handling approach (e.g. if
it's too large to bake into the image cleanly, in which case ask me and
I'll write the Cloud Storage-mount version instead -- but try baking it in
first, it's simpler and this size corpus should fit fine).
