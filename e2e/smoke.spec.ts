import { expect, test } from '@playwright/test';

test('loads the game and solves level 1', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('./');
  await expect(page.locator('h1')).toHaveText('Map');

  await page.locator('.cm-content').click();
  await page.keyboard.press('ControlOrMeta+a');
  await page.keyboard.insertText(
    '__global__ void map(const float* a, float* out) {\n  int i = threadIdx.x;\n  out[i] = a[i] + 10.0f;\n}\n',
  );
  await page.getByRole('button', { name: /^Run/ }).click();

  await expect(page.locator('#problem')).toContainText('Solved');
  await expect(page.locator('#score .status')).toHaveText('solved');
  await expect(page.locator('#outputs')).toContainText('all 32 values match');
  await expect(page.locator('#step-info')).toContainText('step 4 of 4');

  // Stepping moves through the trace and highlights the source line.
  await page.getByRole('button', { name: 'Step back' }).click();
  await expect(page.locator('#step-info')).toContainText('store out');
  await expect(page.locator('.cm-step-line')).toHaveText('  out[i] = a[i] + 10.0f;');
  expect(errors).toEqual([]);
});
