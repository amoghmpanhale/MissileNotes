jest.mock('expo-file-system/legacy', () => ({
  EncodingType: { Base64: 'base64' },
  readAsStringAsync: jest.fn(async (uri: string) => `b64(${uri})`),
}));

jest.mock('expo-print', () => ({
  printToFileAsync: jest.fn(async () => ({ uri: 'file:///tmp/fake.pdf' })),
}));

import { buildPdfHtml, docToPdf } from './pdf';
import * as FileSystem from 'expo-file-system/legacy';
import * as Print from 'expo-print';
import type { Doc } from '@/src/store/documents';

describe('buildPdfHtml', () => {
  it('embeds one img per page, base64-encoded', () => {
    const html = buildPdfHtml(['aaaa', 'bbbb', 'cccc']);

    expect((html.match(/<img /g) ?? []).length).toBe(3);
    expect(html).toContain('data:image/jpeg;base64,aaaa');
    expect(html).toContain('data:image/jpeg;base64,bbbb');
    expect(html).toContain('data:image/jpeg;base64,cccc');
  });

  it('applies a page-break class per page div, consistently across the doc', () => {
    const html = buildPdfHtml(['aaaa', 'bbbb', 'cccc']);

    const pageDivs = html.match(/<div class="page">/g) ?? [];
    expect(pageDivs.length).toBe(3);
    // The break behavior (all-but-last break) is expressed once via CSS
    // (`.page:last-child`), not per-div markup, so every div is identical.
  });

  it('sets @page margin to 0', () => {
    const html = buildPdfHtml(['aaaa']);
    expect(html).toMatch(/@page\s*{\s*margin:\s*0/);
  });

  it('handles a single page', () => {
    const html = buildPdfHtml(['solo']);
    expect((html.match(/<img /g) ?? []).length).toBe(1);
    expect(html).toContain('data:image/jpeg;base64,solo');
  });

  it('handles zero pages without throwing', () => {
    const html = buildPdfHtml([]);
    expect((html.match(/<img /g) ?? []).length).toBe(0);
  });
});

describe('docToPdf', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  const baseDoc: Doc = {
    id: 'doc1',
    title: 'Test Doc',
    createdAt: new Date().toISOString(),
    pages: ['file:///docs/doc1/p0.jpg', 'file:///docs/doc1/p1.jpg'],
  };

  it('reads each page as base64 and returns the printed PDF uri', async () => {
    const uri = await docToPdf(baseDoc);

    expect(FileSystem.readAsStringAsync).toHaveBeenCalledTimes(2);
    expect(FileSystem.readAsStringAsync).toHaveBeenCalledWith(
      'file:///docs/doc1/p0.jpg',
      { encoding: 'base64' }
    );
    expect(Print.printToFileAsync).toHaveBeenCalledWith({
      html: expect.stringContaining('b64(file:///docs/doc1/p0.jpg)'),
    });
    expect(uri).toBe('file:///tmp/fake.pdf');
  });

  it('throws a clear error for a zero-page doc', async () => {
    await expect(docToPdf({ ...baseDoc, pages: [] })).rejects.toThrow(/no pages/i);
    expect(Print.printToFileAsync).not.toHaveBeenCalled();
  });
});
