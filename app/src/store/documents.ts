// Local-first document store. Every doc lives at
//   <documentDirectory>/docs/<uuid>/manifest.json
// with its page images alongside as p0.jpg, p1.jpg, ... A flat library index
// at <documentDirectory>/docs/index.json lets listDocs() avoid reading every
// manifest just to render the library screen.
import * as FileSystem from 'expo-file-system/legacy';

export type Doc = {
  id: string;
  title: string;
  createdAt: string;
  pages: string[]; // absolute file URIs
  summary?: string;
  driveFileId?: string;
};

type IndexEntry = { id: string; title: string; createdAt: string; pageCount: number };

const ROOT = FileSystem.documentDirectory + 'docs/';
const INDEX_PATH = ROOT + 'index.json';

function docDir(id: string): string {
  return `${ROOT}${id}/`;
}

function manifestPath(id: string): string {
  return `${docDir(id)}manifest.json`;
}

function newId(): string {
  return globalThis.crypto.randomUUID();
}

async function ensureRoot(): Promise<void> {
  const info = await FileSystem.getInfoAsync(ROOT);
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(ROOT, { intermediates: true });
  }
}

async function readIndex(): Promise<IndexEntry[]> {
  const info = await FileSystem.getInfoAsync(INDEX_PATH);
  if (!info.exists) return [];
  const raw = await FileSystem.readAsStringAsync(INDEX_PATH);
  return JSON.parse(raw) as IndexEntry[];
}

async function writeIndex(entries: IndexEntry[]): Promise<void> {
  await FileSystem.writeAsStringAsync(INDEX_PATH, JSON.stringify(entries));
}

// ponytail: naive read-modify-write on index.json — fine for a single-user,
// single-process app, but concurrent writers would race and clobber each
// other. Revisit (file locking, or a real DB) if this ever becomes multi-writer.
async function updateIndex(mutate: (entries: IndexEntry[]) => IndexEntry[]): Promise<void> {
  const entries = await readIndex();
  await writeIndex(mutate(entries));
}

async function writeManifest(doc: Doc): Promise<void> {
  await FileSystem.writeAsStringAsync(manifestPath(doc.id), JSON.stringify(doc));
}

export async function createDoc(title: string): Promise<Doc> {
  await ensureRoot();
  const id = newId();
  const doc: Doc = { id, title, createdAt: new Date().toISOString(), pages: [] };

  await FileSystem.makeDirectoryAsync(docDir(id), { intermediates: true });
  await writeManifest(doc);
  await updateIndex((entries) => [
    ...entries,
    { id: doc.id, title: doc.title, createdAt: doc.createdAt, pageCount: 0 },
  ]);

  return doc;
}

export async function addPage(docId: string, tempImageUri: string): Promise<Doc> {
  const doc = await getDoc(docId);
  const destUri = `${docDir(docId)}p${doc.pages.length}.jpg`;

  await FileSystem.copyAsync({ from: tempImageUri, to: destUri });

  const updated: Doc = { ...doc, pages: [...doc.pages, destUri] };
  await writeManifest(updated);
  await updateIndex((entries) =>
    entries.map((e) => (e.id === docId ? { ...e, pageCount: updated.pages.length } : e))
  );

  return updated;
}

export async function listDocs(): Promise<
  Array<{ id: string; title: string; createdAt: string; pageCount: number }>
> {
  return readIndex();
}

export async function getDoc(id: string): Promise<Doc> {
  const raw = await FileSystem.readAsStringAsync(manifestPath(id));
  return JSON.parse(raw) as Doc;
}

export async function updateDoc(id: string, patch: Partial<Doc>): Promise<Doc> {
  const existing = await getDoc(id);
  const updated: Doc = { ...existing, ...patch, id: existing.id };

  await writeManifest(updated);
  await updateIndex((entries) =>
    entries.map((e) =>
      e.id === id
        ? { ...e, title: updated.title, pageCount: updated.pages.length }
        : e
    )
  );

  return updated;
}

export async function deleteDoc(id: string): Promise<void> {
  await FileSystem.deleteAsync(docDir(id), { idempotent: true });
  await updateIndex((entries) => entries.filter((e) => e.id !== id));
}
