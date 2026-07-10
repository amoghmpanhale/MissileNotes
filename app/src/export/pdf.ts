// Renders a Doc's pages into a single PDF via expo-print's HTML-to-PDF path.
// Each page becomes one full-bleed <img>, separated by a page break so the
// PDF has exactly one page per source image.
import * as FileSystem from 'expo-file-system/legacy';
import * as Print from 'expo-print';

import type { Doc } from '@/src/store/documents';

// Pure and unit-tested directly: given already-base64-encoded page images,
// build the HTML document expo-print will rasterize. No file/network I/O
// here so it's cheap to assert against in tests.
export function buildPdfHtml(pageBase64: string[]): string {
  const pages = pageBase64
    .map(
      (b64) => `<div class="page">
      <img src="data:image/jpeg;base64,${b64}" />
    </div>`
    )
    .join('\n');

  return `<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      @page { margin: 0; }
      html, body { margin: 0; padding: 0; }
      .page { page-break-after: always; }
      .page:last-child { page-break-after: auto; }
      .page img { width: 100%; display: block; }
    </style>
  </head>
  <body>
    ${pages}
  </body>
</html>`;
}

export async function docToPdf(doc: Doc): Promise<string> {
  if (doc.pages.length === 0) {
    throw new Error('Cannot export a PDF for a document with no pages.');
  }

  const pageBase64 = await Promise.all(
    doc.pages.map((uri) =>
      FileSystem.readAsStringAsync(uri, { encoding: FileSystem.EncodingType.Base64 })
    )
  );

  const html = buildPdfHtml(pageBase64);
  const { uri } = await Print.printToFileAsync({ html });
  return uri;
}
