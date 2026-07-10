import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router, useNavigation } from 'expo-router';

import { processScan } from '@/src/pipeline/scan';
import { addPage, createDoc, deleteDoc, updateDoc } from '@/src/store/documents';

function defaultTitle(): string {
  return `Scan ${new Date().toLocaleDateString()}`;
}

export default function ScanScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const navigation = useNavigation();

  const [docId, setDocId] = useState<string | null>(null);
  const [pages, setPages] = useState<string[]>([]);
  const [capturing, setCapturing] = useState(false);
  // True while processScan() (marker detect -> segment -> warp) runs on a
  // just-captured photo. Distinct from `capturing` (shutter-to-photo) so the
  // UI can show "processing" during the (slower) CV/ML step specifically.
  const [processing, setProcessing] = useState(false);
  const [titleModalVisible, setTitleModalVisible] = useState(false);
  const [titleInput, setTitleInput] = useState('');

  // Synchronous gate for double-capture: `capturing` state only blocks the
  // Pressable after a re-render, so two taps in the same JS tick could both
  // pass a state-only check. This ref is set/reset immediately, no await gap.
  const capturingRef = useRef(false);

  // Set true only by the committed "Done/save" path. Any other exit
  // (hardware back, swipe-back, the Close button) leaves this false, and the
  // beforeRemove listener below discards the in-progress doc on the way out.
  const savedRef = useRef(false);

  // Ref to track current docId for the beforeRemove listener, avoiding
  // stale-closure bugs when navigation events fire between setDocId and
  // the effect re-running.
  const docIdRef = useRef<string | null>(null);

  useEffect(() => {
    const unsubscribe = navigation.addListener('beforeRemove', () => {
      if (!savedRef.current && docIdRef.current) {
        deleteDoc(docIdRef.current).catch((err) => {
          console.warn('Failed to discard in-progress scan', err);
        });
      }
    });
    return unsubscribe;
  }, [navigation]);

  async function handleCapture() {
    if (!cameraRef.current || capturingRef.current) return;
    capturingRef.current = true;
    setCapturing(true);
    try {
      const photo = await cameraRef.current.takePictureAsync();
      if (!photo?.uri) return;

      let id = docId;
      if (!id) {
        const doc = await createDoc(defaultTitle());
        id = doc.id;
        docIdRef.current = id;
        setDocId(id);
      }

      // processScan() never throws — on any CV/ML failure (no marker, no
      // quad, inference error) it resolves to the original photo.uri, so a
      // scan is never blocked on the pipeline.
      setProcessing(true);
      const scannedUri = await processScan(photo.uri).finally(() => setProcessing(false));

      const updated = await addPage(id, scannedUri);
      setPages(updated.pages);
    } finally {
      capturingRef.current = false;
      setCapturing(false);
    }
  }

  function handleDone() {
    if (!docId || pages.length === 0) {
      router.back();
      return;
    }
    setTitleInput(defaultTitle());
    setTitleModalVisible(true);
  }

  async function handleConfirmTitle() {
    // Mark this as the committed save path before anything async so the
    // beforeRemove listener (fired by the router.back() below) doesn't treat
    // this exit as an abandonment and delete the doc.
    savedRef.current = true;
    if (docId) {
      try {
        await updateDoc(docId, { title: titleInput.trim() || defaultTitle() });
      } catch (err) {
        console.warn('Failed to save scan title', err);
      }
    }
    setTitleModalVisible(false);
    router.back();
  }

  function handleCancel() {
    // No delete here: navigation's beforeRemove listener (registered above)
    // handles discarding the in-progress doc for every non-save exit,
    // including this button, hardware back, and swipe-back alike.
    router.back();
  }

  if (!permission) {
    return <View style={styles.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>
          MissileNotes needs camera access to scan pages.
        </Text>
        <Pressable style={styles.permissionButton} onPress={requestPermission}>
          <Text style={styles.permissionButtonText}>Grant camera access</Text>
        </Pressable>
        <Pressable style={styles.closeLink} onPress={() => router.back()}>
          <Text style={styles.closeLinkText}>Close</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing="back" />

      <Pressable style={styles.closeButton} onPress={handleCancel}>
        <Text style={styles.closeButtonText}>Close</Text>
      </Pressable>

      <Pressable style={styles.doneButton} onPress={handleDone}>
        <Text style={styles.doneButtonText}>Done ({pages.length})</Text>
      </Pressable>

      <View style={styles.bottomBar}>
        {pages.length > 0 && (
          <ScrollView
            horizontal
            style={styles.thumbStrip}
            contentContainerStyle={styles.thumbStripContent}
            showsHorizontalScrollIndicator={false}
          >
            {pages.map((uri) => (
              <Image key={uri} source={{ uri }} style={styles.thumb} />
            ))}
          </ScrollView>
        )}

        <Pressable
          style={[styles.shutterButton, capturing && styles.shutterButtonDisabled]}
          onPress={handleCapture}
          disabled={capturing}
        />
      </View>

      {processing && (
        <View style={styles.processingOverlay} pointerEvents="none">
          <ActivityIndicator size="large" color="#fff" />
          <Text style={styles.processingText}>Flattening scan…</Text>
        </View>
      )}

      <Modal
        visible={titleModalVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setTitleModalVisible(false)}
      >
        <View style={styles.modalBackdrop}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Name this document</Text>
            <TextInput
              style={styles.modalInput}
              value={titleInput}
              onChangeText={setTitleInput}
              autoFocus
              placeholder={defaultTitle()}
              placeholderTextColor="#999"
            />
            <View style={styles.modalActions}>
              <Pressable
                style={styles.modalButton}
                onPress={() => setTitleModalVisible(false)}
              >
                <Text style={styles.modalButtonText}>Cancel</Text>
              </Pressable>
              <Pressable
                style={[styles.modalButton, styles.modalButtonPrimary]}
                onPress={handleConfirmTitle}
              >
                <Text style={[styles.modalButtonText, styles.modalButtonPrimaryText]}>
                  Save
                </Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    alignItems: 'center',
    justifyContent: 'center',
  },
  permissionText: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    marginHorizontal: 32,
    marginBottom: 20,
  },
  permissionButton: {
    backgroundColor: '#fff',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
  },
  permissionButtonText: {
    fontWeight: '600',
  },
  closeLink: {
    marginTop: 16,
  },
  closeLinkText: {
    color: '#aaa',
  },
  closeButton: {
    position: 'absolute',
    top: 56,
    left: 16,
    paddingHorizontal: 14,
    paddingVertical: 8,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 20,
    zIndex: 1,
  },
  closeButtonText: {
    color: '#fff',
    fontSize: 15,
  },
  doneButton: {
    position: 'absolute',
    top: 56,
    right: 16,
    paddingHorizontal: 14,
    paddingVertical: 8,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 20,
    zIndex: 1,
  },
  doneButtonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
  bottomBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingBottom: 32,
    paddingTop: 12,
    backgroundColor: 'rgba(0,0,0,0.35)',
    alignItems: 'center',
  },
  thumbStrip: {
    maxHeight: 64,
    marginBottom: 16,
  },
  thumbStripContent: {
    paddingHorizontal: 12,
    gap: 8,
  },
  thumb: {
    width: 48,
    height: 48,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#fff',
  },
  shutterButton: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: '#fff',
    borderWidth: 4,
    borderColor: 'rgba(255,255,255,0.4)',
  },
  shutterButtonDisabled: {
    opacity: 0.5,
  },
  processingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  processingText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
    marginTop: 12,
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
});
