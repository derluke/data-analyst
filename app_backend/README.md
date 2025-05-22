# Talk to My Data: FastAPI App app_backend

## Tech Stack

- FastAPI


## Development

Start the backend from the project root:

```bash
make run-local-dev-backend 
```

To start the frontend development server, cd into the `../frontend_react` directory first:

```bash
npm i
npm run dev
```

Make sure to open the frontend url (likely on vite default port 5173) *not* the backend url.


## Building

To build the application for production:

```bash
npm run build
```

The build output will be placed in the `../app_backend/static` directory, which is then used by the Python backend to serve the application. When using the React frontend through the `FRONTEND_TYPE="react"` environment variable, the application will look for the built files in this location.

You can run it locally using the static files by starting it from project root with:

```bash
make run-local-static-backend 
```

## Codespace

To run the app in a codespace, first open a terminal and install the necessary dependencies:
```bash
pip install -r requirements.txt
```

Then start the app with:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --proxy-headers --log-level "warning"
```

Navigate to `Session environment` and find the link to the app in the exposed ports section. 


## Testing

To run tests:

```
uv run pytest --cov --cov-report term --cov-report html
```


To test through a DataRobot like proxy:

Install traefik by downloading from: https://github.com/traefik/traefik/releases


Configure it like with `traefik.toml` like:

```toml
[entryPoints]
  [entryPoints.http]
    address = ":9999"

[providers]
  [providers.file]
    filename = "routes.toml"
```

Create a `routes.toml` file like:

```toml
[http]
  [http.middlewares]
    [http.middlewares.add-foo.addPrefix]
      prefix = "/app_backend"
    
    [http.middlewares.add-prefix-header.headers]
      customRequestHeaders = { X-Forwarded-Prefix = "/app_backend" }

  [http.routers]
    [http.routers.app-http]
      entryPoints = ["http"]
      service = "app"
      rule = "PathPrefix(`/app_backend`)"
      middlewares = ["add-prefix-header"]

  [http.services]
    [http.services.app]
      [http.services.app.loadBalancer]
        [[http.services.app.loadBalancer.servers]]
          url = "http://127.0.0.1:8080"
```


And run locally with:

`traefik --configFile=traefik.toml`

With the fastapi running now accessing:

http://localhost:9999/app_backend

will take you to a proxy compatible installation
