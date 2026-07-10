import { useCallback, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  FlatList,
  Image,
  Modal,
  NativeScrollEvent,
  NativeSyntheticEvent,
  Pressable,
  StyleSheet,
  TextInput,
} from 'react-native';
import { router, useFocusEffect, useLocalSearchParams } from 'expo-router';
import * as Sharing from 'expo-sharing';

import { Text, View } from '@/components/Themed';
import { deleteDoc, getDoc, updateDoc, type Doc } from '@/src/store/documents';
import { docToPdf } from '@/src/export/pdf';

export default function DocScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();

  const [doc, setDoc] = useState<Doc | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [loading, setLoading] = useState(true);
  const [pageIndex, setPageIndex] = useState(0);

  const [renameVisible, setRenameVisible] = useState(false);
  const [titleInput, setTitleInput] = useState('');
  const [exportVisible, setExportVisible] = useState(false);
  const [exporting, setExporting] = useState(false);

  useFocusEffect(
    useCallback(() => {
      let cancelled = false;
      if (!id) {
        setNotFound(true);
        setLoading(false);
        return;
      }
      setLoading(true);
      getDoc(id).then(
        (fetched) => {
          if (!cancelled) {
            setDoc(fetched);
            setNotFound(false);
            setLoading(false);
          }
        },
        () => {
          if (!cancelled) {
            setNotFound(true);
            setLoading(false);
          }
        }
      );
      return () => {
        cancelled = true;
      };
    }, [id])
  );

  function handlePageScroll(e: NativeSyntheticEvent<NativeScrollEvent>) {
    const width = e.nativeEvent.layoutMeasurement.width || 1;
    const index = Math.round(e.nativeEvent.contentOffset.x / width);
    setPageIndex(index);
  }

  function openRename() {
    if (!doc) return;
    setTitleInput(doc.title);
    setRenameVisible(true);
  }

  async function handleConfirmRename() {
    if (!doc) return;
    const nextTitle = titleInput.trim();
    if (nextTitle) {
      const updated = await updateDoc(doc.id, { title: nextTitle });
      setDoc(updated);
    }
    setRenameVisible(false);
  }

  function handleDelete() {
    if (!doc) return;
    Alert.alert('Delete document?', 'This cannot be undone.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          await deleteDoc(doc.id);
          router.back();
        },
      },
    ]);
  }

  async function handleExportPdf() {
    if (!doc || doc.pages.length === 0) return;
    setExportVisible(false);
    setExporting(true);
    try {
      const uri = await docToPdf(doc);
      await Sharing.shareAsync(uri, {
        UTI: 'com.adobe.pdf',
        mimeType: 'application/pdf',
      });
    } catch (err) {
      Alert.alert('Export failed', String(err));
    } finally {
      setExporting(false);
    }
  }

  async function handleExportImages() {
    if (!doc || doc.pages.length === 0) return;
    setExportVisible(false);
    setExporting(true);
    try {
      // expo-sharing shares one file at a time on iOS; there is no bundled
      // "share all pages as a folder" primitive without pulling in a zip
      // library, which the brief rules out. Share each page in turn instead.
      for (const uri of doc.pages) {
        await Sharing.shareAsync(uri, { mimeType: 'image/jpeg' });
      }
    } catch (err) {
      Alert.alert('Export failed', String(err));
    } finally {
      setExporting(false);
    }
  }

  if (loading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator />
      </View>
    );
  }

  if (notFound || !doc) {
    return (
      <View style={styles.centered}>
        <Text style={styles.notFoundText}>Document not found</Text>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Text style={styles.backButtonText}>Back</Text>
        </Pressable>
      </View>
    );
  }

  const hasPages = doc.pages.length > 0;
  const screenWidth = Dimensions.get('window').width;

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Pressable style={styles.headerButton} onPress={() => router.back()}>
          <Text style={styles.headerButtonText}>Back</Text>
        </Pressable>
        <Pressable style={styles.titleTouch} onPress={openRename}>
          <Text style={styles.title} numberOfLines={1}>
            {doc.title}
          </Text>
        </Pressable>
        <Pressable style={styles.headerButton} onPress={handleDelete}>
          <Text style={[styles.headerButtonText, styles.deleteText]}>Delete</Text>
        </Pressable>
      </View>

      {hasPages ? (
        <>
          <FlatList
            data={doc.pages}
            keyExtractor={(uri) => uri}
            horizontal
            pagingEnabled
            showsHorizontalScrollIndicator={false}
            onMomentumScrollEnd={handlePageScroll}
            renderItem={({ item }) => (
              <View style={[styles.page, { width: screenWidth }]}>
                <Image source={{ uri: item }} style={styles.pageImage} resizeMode="contain" />
              </View>
            )}
          />
          <Text style={styles.pageIndicator}>
            {pageIndex + 1} / {doc.pages.length}
          </Text>
        </>
      ) : (
        <View style={styles.centered}>
          <Text style={styles.notFoundText}>No pages in this document.</Text>
        </View>
      )}

      <View style={styles.footer}>
        <Pressable
          style={[styles.exportButton, !hasPages && styles.exportButtonDisabled]}
          onPress={() => setExportVisible(true)}
          disabled={!hasPages || exporting}
        >
          <Text style={styles.exportButtonText}>
            {exporting ? 'Exporting…' : 'Export'}
          </Text>
        </Pressable>
      </View>

      <Modal
        visible={renameVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setRenameVisible(false)}
      >
        <View style={styles.modalBackdrop}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Rename document</Text>
            <TextInput
              style={styles.modalInput}
              value={titleInput}
              onChangeText={setTitleInput}
              autoFocus
              placeholderTextColor="#999"
            />
            <View style={styles.modalActions}>
              <Pressable style={styles.modalButton} onPress={() => setRenameVisible(false)}>
                <Text style={styles.modalButtonText}>Cancel</Text>
              </Pressable>
              <Pressable
                style={[styles.modalButton, styles.modalButtonPrimary]}
                onPress={handleConfirmRename}
              >
                <Text style={[styles.modalButtonText, styles.modalButtonPrimaryText]}>
                  Save
                </Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>

      <Modal
        visible={exportVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setExportVisible(false)}
      >
        <View style={styles.modalBackdrop}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Export</Text>
            <Pressable style={styles.modalButtonFull} onPress={handleExportPdf}>
              <Text style={styles.modalButtonFullText}>Export PDF</Text>
            </Pressable>
            <Pressable style={styles.modalButtonFull} onPress={handleExportImages}>
              <Text style={styles.modalButtonFullText}>Export Images</Text>
            </Pressable>
            <Pressable style={styles.modalButton} onPress={() => setExportVisible(false)}>
              <Text style={styles.modalButtonText}>Cancel</Text>
            </Pressable>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  notFoundText: {
    fontSize: 16,
    opacity: 0.7,
  },
  backButton: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: '#2f95dc',
    borderRadius: 8,
  },
  backButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#ccc',
  },
  headerButton: {
    paddingHorizontal: 8,
    paddingVertical: 6,
  },
  headerButtonText: {
    fontSize: 15,
    color: '#2f95dc',
  },
  deleteText: {
    color: '#d9534f',
  },
  titleTouch: {
    flex: 1,
    marginHorizontal: 8,
  },
  title: {
    fontSize: 17,
    fontWeight: '600',
    textAlign: 'center',
  },
  page: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  pageImage: {
    width: '100%',
    height: '100%',
  },
  pageIndicator: {
    textAlign: 'center',
    paddingVertical: 8,
    fontSize: 13,
    opacity: 0.6,
  },
  footer: {
    padding: 16,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#ccc',
  },
  exportButton: {
    backgroundColor: '#2f95dc',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  exportButtonDisabled: {
    opacity: 0.4,
  },
  exportButtonText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 16,
  },
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalCard: {
    width: '85%',
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 20,
  },
  modalTitle: {
    fontSize: 17,
    fontWeight: '600',
    marginBottom: 12,
  },
  modalInput: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 16,
  },
  modalActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginTop: 16,
    gap: 12,
  },
  modalButton: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 8,
  },
  modalButtonPrimary: {
    backgroundColor: '#2f95dc',
  },
  modalButtonText: {
    fontSize: 15,
    fontWeight: '500',
  },
  modalButtonPrimaryText: {
    color: '#fff',
  },
  modalButtonFull: {
    paddingVertical: 12,
    borderRadius: 8,
    backgroundColor: '#2f95dc',
    alignItems: 'center',
    marginTop: 8,
  },
  modalButtonFullText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 15,
  },
});
