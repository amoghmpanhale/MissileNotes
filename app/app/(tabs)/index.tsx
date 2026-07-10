import { useCallback, useState } from 'react';
import { FlatList, Pressable, StyleSheet } from 'react-native';
import { router, useFocusEffect } from 'expo-router';
import { Text, View } from '@/components/Themed';
import { listDocs } from '@/src/store/documents';

type DocSummary = {
  id: string;
  title: string;
  createdAt: string;
  pageCount: number;
};

export default function LibraryScreen() {
  const [docs, setDocs] = useState<DocSummary[]>([]);

  useFocusEffect(
    useCallback(() => {
      listDocs().then(setDocs);
    }, [])
  );

  return (
    <View style={styles.container}>
      <FlatList
        data={docs}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <Text style={styles.emptyText}>No scans yet. Tap Scan to start.</Text>
          </View>
        }
        renderItem={({ item }) => (
          <Pressable
            style={styles.row}
            onPress={() => router.push(`/doc/${item.id}`)}
          >
            <Text style={styles.rowTitle}>{item.title}</Text>
            <Text style={styles.rowSubtitle}>
              {item.pageCount} {item.pageCount === 1 ? 'page' : 'pages'} ·{' '}
              {new Date(item.createdAt).toLocaleDateString()}
            </Text>
          </Pressable>
        )}
      />

      <Pressable style={styles.fab} onPress={() => router.push('/scan')}>
        <Text style={styles.fabText}>Scan</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  listContent: {
    paddingVertical: 8,
    flexGrow: 1,
  },
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 80,
  },
  emptyText: {
    fontSize: 15,
    opacity: 0.6,
  },
  row: {
    paddingHorizontal: 20,
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#ccc',
  },
  rowTitle: {
    fontSize: 17,
    fontWeight: '600',
  },
  rowSubtitle: {
    fontSize: 13,
    opacity: 0.6,
    marginTop: 4,
  },
  fab: {
    position: 'absolute',
    right: 20,
    bottom: 32,
    backgroundColor: '#2f95dc',
    paddingHorizontal: 22,
    paddingVertical: 14,
    borderRadius: 28,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
  },
  fabText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 16,
  },
});
