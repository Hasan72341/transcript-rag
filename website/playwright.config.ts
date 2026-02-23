import { defineConfig } from "@playwright/test";

export default defineConfig({
    testDir: "./e2e",
    use: { baseURL: "http://127.0.0.1:3100", headless: true },
    webServer: {
        command: "yarn dev --hostname 127.0.0.1 --port 3100",
        url: "http://127.0.0.1:3100",
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
    },
});
