# mpc-demo

Notebook showing an MPC with rolling horizon control of an energy storage system.

## Instructions

Install dependencies:

```bash
uv sync
```

Run the notebook:

```bash
poe run
```

## Run in a container

Build the image:

```bash
docker build -t mpc-demo .
```

Run it locally:

```bash
docker run --rm -p 8080:8080 -e PORT=8080 mpc-demo
```

Open `http://localhost:8080`.

## Deploy to Google Cloud Run

```bash
PROJECT_ID=<your-gcp-project-id>
REGION=<your-gcp-region>
gcloud builds submit --tag gcr.io/$PROJECT_ID/mpc-demo
gcloud run deploy mpc-demo \
  --image gcr.io/$PROJECT_ID/mpc-demo \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated
```
