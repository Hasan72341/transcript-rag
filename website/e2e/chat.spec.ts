import { expect, test } from "@playwright/test";

const response = {
    bot_answer: {
        answer: "The customer canceled because the fee was too high.",
        time: 1.5,
        node_ids: ["node-1"],
        unique_text: [{id: "node-1", text: "A cancellation complaint."}],
        leave_text: [{id: "transcript-1", text: "Customer: Please cancel."}],
    },
    evidence: [{
        transcript_id: "transcript-1",
        turn_numbers: [0],
        relevant_turns: [{turn_index: 0, speaker: "Customer", text: "Please cancel."}],
        full_turns: [
            {turn_index: 0, speaker: "Customer", text: "Please cancel."},
            {turn_index: 1, speaker: "Agent", text: "Your booking\nis canceled."},
        ],
    }],
};

test("shows backend errors, retries without duplicate messages, and opens full evidence", async ({page}) => {
    let requests = 0;
    await page.route("**/api/chat", async (route) => {
        requests += 1;
        await route.fulfill({status: requests === 1 ? 503 : 200, json: requests === 1 ? {error: "Missing RAG artifacts"} : response});
    });
    await page.goto("/");
    await page.getByRole("textbox").fill("Why did the customer cancel?");
    await page.getByRole("button", {name: "Send message"}).click();
    await expect(page.getByRole("main").getByRole("alert")).toContainText("Missing RAG artifacts");
    await page.getByRole("button", {name: "Retry question"}).click();
    await expect(page.getByText(response.bot_answer.answer, {exact: true})).toBeVisible();
    await expect(page.getByRole("main").getByText("Why did the customer cancel?", {exact: true})).toHaveCount(1);
    await page.getByRole("button", {name: /View Source Evidence/}).click();
    await expect(page.getByText("Please cancel.", {exact:true})).toBeVisible();
    await page.getByRole("button", {name: "Show Full"}).click();
    await expect(page.getByText("Your booking is canceled.", {exact:true})).toBeVisible();
});

test("disables repeated submissions while the answer is pending", async ({page}) => {
    let release!: () => void;
    const pending = new Promise<void>((resolve) => { release = resolve; });
    await page.route("**/api/chat", async (route) => {
        await pending;
        await route.fulfill({json: response});
    });
    await page.goto("/");
    await page.getByRole("textbox").fill("Why cancel?");
    await page.getByRole("button", {name: "Send message"}).click();
    await expect(page.getByRole("status")).toContainText("Searching conversations");
    await expect(page.getByRole("button", {name: "Send message"})).toBeDisabled();
    await expect(page.getByRole("textbox")).toBeDisabled();
    release();
    await expect(page.getByText(response.bot_answer.answer, {exact:true})).toBeVisible();
    await expect(page.getByRole("textbox")).toBeEnabled();
});

test("keeps a pending answer in its original chat after switching chats", async ({page}) => {
    let release!: () => void;
    const pending = new Promise<void>((resolve) => { release = resolve; });
    await page.route("**/api/chat", async (route) => {
        await pending;
        await route.fulfill({json: response});
    });
    await page.goto("/");
    await page.getByRole("textbox").fill("Original question");
    await page.getByRole("button", {name: "Send message"}).click();
    await expect(page.getByRole("status")).toBeVisible();
    await page.getByRole("button", {name: "Toggle chat sidebar"}).click();
    await page.getByRole("button", {name: "New chat"}).click();
    release();
    await expect(page.getByRole("textbox")).toBeEnabled();
    await expect(page.getByText(response.bot_answer.answer, {exact:true})).toHaveCount(0);
    await page.locator("aside").getByText("Original question", {exact:true}).click();
    await expect(page.getByText(response.bot_answer.answer, {exact:true})).toBeVisible();
});
