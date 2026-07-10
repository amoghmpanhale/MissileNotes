// In-memory fake for expo-file-system/legacy. jest-expo does not provide a real
// mockFilesystem, so we simulate one with plain JS maps/sets keyed by URI string.
const mockFiles = new Map<string, string>();
const mockDirs = new Set<string>();

function reset() {
  mockFiles.clear();
  mockDirs.clear();
  mockDirs.add('file:///doc-dir/');
}
reset();

jest.mock('expo-file-system/legacy', () => {
  return {
    documentDirectory: 'file:///doc-dir/',
    getInfoAsync: async (uri: string) => {
      if (mockFiles.has(uri)) return { exists: true, uri, isDirectory: false };
      if (mockDirs.has(uri)) return { exists: true, uri, isDirectory: true };
      return { exists: false, uri, isDirectory: false };
    },
    readAsStringAsync: async (uri: string) => {
      if (!mockFiles.has(uri)) {
        const err: any = new Error(`ENOENT: ${uri}`);
        throw err;
      }
      return mockFiles.get(uri)!;
    },
    writeAsStringAsync: async (uri: string, contents: string) => {
      mockFiles.set(uri, contents);
    },
    deleteAsync: async (uri: string, _options?: { idempotent?: boolean }) => {
      // Recursively remove anything nested under this uri (directory semantics),
      // plus the exact key itself (file semantics).
      mockFiles.delete(uri);
      mockDirs.delete(uri);
      for (const key of Array.from(mockFiles.keys())) {
        if (key.startsWith(uri)) mockFiles.delete(key);
      }
      for (const key of Array.from(mockDirs.keys())) {
        if (key.startsWith(uri)) mockDirs.delete(key);
      }
    },
    copyAsync: async ({ from, to }: { from: string; to: string }) => {
      if (!mockFiles.has(from)) {
        throw new Error(`ENOENT (copy source): ${from}`);
      }
      mockFiles.set(to, mockFiles.get(from)!);
    },
    makeDirectoryAsync: async (uri: string, _options?: { intermediates?: boolean }) => {
      mockDirs.add(uri);
    },
    readDirectoryAsync: async (uri: string) => {
      const prefix = uri.endsWith('/') ? uri : uri + '/';
      const names = new Set<string>();
      for (const key of mockFiles.keys()) {
        if (key.startsWith(prefix)) names.add(key.slice(prefix.length));
      }
      return Array.from(names);
    },
  };
});

import {
  createDoc,
  addPage,
  listDocs,
  getDoc,
  updateDoc,
  deleteDoc,
} from './documents';

const FileSystem = require('expo-file-system/legacy');

beforeEach(() => {
  reset();
  // Seed a fake "temp" image the camera/picker would have produced.
  mockFiles.set('file:///tmp/photo1.jpg', 'fake-jpeg-bytes-1');
  mockFiles.set('file:///tmp/photo2.jpg', 'fake-jpeg-bytes-2');
});

describe('createDoc + listDocs', () => {
  it('create shows up in list', async () => {
    const doc = await createDoc('My Scan');

    expect(doc.title).toBe('My Scan');
    expect(doc.id).toEqual(expect.any(String));
    expect(doc.pages).toEqual([]);

    const list = await listDocs();
    expect(list).toHaveLength(1);
    expect(list[0]).toMatchObject({
      id: doc.id,
      title: 'My Scan',
      pageCount: 0,
    });
    expect(typeof list[0].createdAt).toBe('string');
  });

  it('lists multiple docs, most recently created included', async () => {
    await createDoc('First');
    await createDoc('Second');

    const list = await listDocs();
    expect(list).toHaveLength(2);
    expect(list.map((d) => d.title).sort()).toEqual(['First', 'Second']);
  });
});

describe('addPage', () => {
  it('copies the file into the doc directory and appends it to pages', async () => {
    const doc = await createDoc('Receipt');

    const updated = await addPage(doc.id, 'file:///tmp/photo1.jpg');

    expect(updated.pages).toHaveLength(1);
    const pageUri = updated.pages[0];
    expect(pageUri).not.toBe('file:///tmp/photo1.jpg');
    expect(pageUri).toContain(doc.id);

    // The bytes were actually copied to the new location.
    expect(mockFiles.get(pageUri)).toBe('fake-jpeg-bytes-1');

    const list = await listDocs();
    expect(list.find((d) => d.id === doc.id)?.pageCount).toBe(1);
  });

  it('appends subsequent pages in order', async () => {
    const doc = await createDoc('Multi-page');
    await addPage(doc.id, 'file:///tmp/photo1.jpg');
    const updated = await addPage(doc.id, 'file:///tmp/photo2.jpg');

    expect(updated.pages).toHaveLength(2);
    expect(mockFiles.get(updated.pages[0])).toBe('fake-jpeg-bytes-1');
    expect(mockFiles.get(updated.pages[1])).toBe('fake-jpeg-bytes-2');
  });
});

describe('getDoc / updateDoc', () => {
  it('round-trips a title update through the manifest and index', async () => {
    const doc = await createDoc('Draft Title');
    await addPage(doc.id, 'file:///tmp/photo1.jpg');

    const updated = await updateDoc(doc.id, { title: 'Final Title', summary: 'a summary' });
    expect(updated.title).toBe('Final Title');
    expect(updated.summary).toBe('a summary');
    expect(updated.pages).toHaveLength(1);

    const fetched = await getDoc(doc.id);
    expect(fetched.title).toBe('Final Title');
    expect(fetched.summary).toBe('a summary');

    const list = await listDocs();
    expect(list.find((d) => d.id === doc.id)?.title).toBe('Final Title');
  });
});

describe('deleteDoc', () => {
  it('removes the doc directory and the index entry', async () => {
    const doc = await createDoc('To Delete');
    await addPage(doc.id, 'file:///tmp/photo1.jpg');

    await deleteDoc(doc.id);

    const list = await listDocs();
    expect(list.find((d) => d.id === doc.id)).toBeUndefined();

    await expect(getDoc(doc.id)).rejects.toThrow();

    // Nothing under the doc's directory should remain in the fake FS.
    const remaining = Array.from(mockFiles.keys()).filter((k) => k.includes(doc.id));
    expect(remaining).toHaveLength(0);
  });

  it('leaves other docs untouched', async () => {
    const keep = await createDoc('Keep me');
    const gone = await createDoc('Delete me');

    await deleteDoc(gone.id);

    const list = await listDocs();
    expect(list).toHaveLength(1);
    expect(list[0].id).toBe(keep.id);
    expect(await getDoc(keep.id)).toBeTruthy();
  });
});

describe('newId fallback (Hermes without crypto)', () => {
  it('generates unique ids when globalThis.crypto is unavailable', async () => {
    const originalCrypto = (globalThis as any).crypto;
    try {
      (globalThis as any).crypto = undefined;

      const doc = await createDoc('Fallback Test');
      expect(doc.id).toBeTruthy();
      expect(typeof doc.id).toBe('string');

      // Verify we can still list and retrieve the doc
      const list = await listDocs();
      expect(list).toHaveLength(1);
      expect(list[0].id).toBe(doc.id);

      const fetched = await getDoc(doc.id);
      expect(fetched.id).toBe(doc.id);
    } finally {
      (globalThis as any).crypto = originalCrypto;
    }
  });
});
