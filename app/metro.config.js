// Learn more https://docs.expo.dev/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// react-native-fast-tflite: bundle .tflite model files as assets.
config.resolver.assetExts.push('tflite');

module.exports = config;
