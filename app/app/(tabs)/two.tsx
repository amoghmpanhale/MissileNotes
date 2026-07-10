import { Pressable, StyleSheet } from 'react-native';
import { Link } from 'expo-router';
import { Text, View } from '@/components/Themed';

export default function SettingsScreen() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Settings</Text>
      {/* ponytail: temporary link to the Task 2.1 native-module spike screen.
          Remove this along with app/app/spike.tsx after Task 2.2. */}
      <Link href="/spike" asChild>
        <Pressable style={styles.spikeLink}>
          <Text style={styles.spikeLinkText}>Dev: native module spike</Text>
        </Pressable>
      </Link>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  spikeLink: {
    marginTop: 24,
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderWidth: StyleSheet.hairlineWidth,
    borderRadius: 8,
  },
  spikeLinkText: {
    fontSize: 14,
    opacity: 0.7,
  },
});
