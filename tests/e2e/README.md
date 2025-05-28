# End-to-End (E2E) Tests

This document provides instructions on how to run the end-to-end (E2E) tests for the project.

## Prerequisites

Ensure you have the following installed:
- Docker
- Python 3.11
- Chrome browser and ChromeDriver


2. **Set environment variables:**
  Create a `.env` file in the `tests/e2e` directory and add the following environment variables:

    ```sh
    APP_USERNAME=your_username
    APP_PASSWORD=your_password
    APP_ID=your_app_id
    DR_HOST=https://.......
    APP_URL=your_app_url  # Optional
    RUN_VISUAL=False  # Set to "True" to run tests with a visible browser
    CHROME_BINARY_LOCATION=/usr/bin/chromium-browser
    CHROMEDRIVER_BINARY_LOCATION=/usr/bin/chromedriver
    FRONTEND_TYPE=react # Set the app type streamlit or react
    ```

3. **Build the Docker image:**
  ```sh
  cd tests/e2e
  docker build -t e2e-tests .
  ```

## Running the Tests

1. **Run the tests using Docker:**
  ```sh
  docker run -it --rm --env-file ./tests/e2e/.env -v "$(pwd)/tests:/tests" e2e-tests python3 -m pytest -vvvs tests/e2e
  ```

2. **Run the tests locally:**
  ```sh
  pytest tests/e2e
  ```

## Notes

- The tests use Selenium WebDriver for browser automation.
- Ensure that the Chrome browser and ChromeDriver versions are compatible.
- The tests are configured to run in headless mode by default. Set `RUN_VISUAL` to `True` to see the browser during test execution.

For more details, refer to the individual test files and configurations in the `tests/e2e` directory.
