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

Set your project and region in `.env`:

```
PROJECT_ID=your-gcp-project-id
REGION=europe-west1
```

Then run:

```bash
poe deploy
```
