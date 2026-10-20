import 'dart:math' as math;
import 'dart:typed_data';

/// A 2-D point in pixels.  The Dart port keeps geometry in plain doubles so it
/// can be read next to the numpy reference without OpenCV wrappers in the way.
class Pt {
  const Pt(this.x, this.y);

  final double x;
  final double y;

  Pt operator +(Pt other) => Pt(x + other.x, y + other.y);
  Pt operator -(Pt other) => Pt(x - other.x, y - other.y);
  Pt operator *(double scale) => Pt(x * scale, y * scale);

  double dot(Pt other) => x * other.x + y * other.y;
  double get norm => math.sqrt(x * x + y * y);

  static double distance(Pt a, Pt b) => (a - b).norm;
}

/// A model probability map at the model frame's resolution.
class ProbMap {
  ProbMap(this.width, this.height, this.data)
      : assert(data.length == width * height);

  final int width;
  final int height;
  final Float32List data;

  double at(int row, int col) => data[row * width + col];

  /// `numpy.rot90(prob, 1)`: one quarter turn counter-clockwise.
  ProbMap rot90Ccw() {
    final out = Float32List(width * height);
    for (var row = 0; row < width; row++) {
      for (var col = 0; col < height; col++) {
        out[row * height + col] = at(col, width - 1 - row);
      }
    }
    return ProbMap(height, width, out);
  }
}

/// `round()` in Python/numpy: ties go to the nearest even integer.
int roundHalfEven(double value) {
  final lower = value.floorToDouble();
  final fraction = value - lower;
  if (fraction < 0.5) return lower.toInt();
  if (fraction > 0.5) return lower.toInt() + 1;
  final asInt = lower.toInt();
  return asInt.isEven ? asInt : asInt + 1;
}
