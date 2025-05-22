# Data Analyst React Frontend

This application provides a modern React-based frontend for the **Talk to My Data** application. It allows users to interact with data, perform analyses, and chat with the system to gain insights from their datasets.

## Features

- Interactive chat interface for data analysis
- Data visualization with interactive plots
- Dataset management and cleansing
- Support for multiple data sources (CSV, Data Registry, Snowflake, Google Cloud)
- Code execution and insights generation

## Tech Stack

- React 18 with TypeScript
- Vite for fast development and building
- Tailwind CSS for styling
- Jest for testing
- React Query for API state management

## Development

Start the backend from the project root:

```bash
make run-local-dev-backend 
```

To start the development server, cd into the `frontend_react` directory first:

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

## Testing

To run the test suite:

```bash
npm run test
```

## Project Structure

- `src/api-state`: API client and hooks for data fetching
- `src/components/ui`: shadcn components
- `src/components/ui-custom`: shadcn based generic components
- `src/pages`: Main application pages
- `src/state`: Application state management
- `src/assets`: Static assets like images and icons
