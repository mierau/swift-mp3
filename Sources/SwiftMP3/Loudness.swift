// SwiftMP3
// Loudness.swift
//
// Integrated loudness (ITU-R BS.1770-4) and sample peak of the encoded audio, for the
// ReplayGain fields of the LAME tag.

import Foundation

/// Measures integrated loudness and sample peak incrementally.
struct LoudnessMeter {
  private let channels: Int
  /// Samples per 100 ms step; gating blocks are four steps long.
  private let stepLength: Int
  private var filters: [KWeighting]
  /// Weighted mean square of each completed step, summed over channels.
  private var steps: [Double] = []
  private var stepSum = 0.0
  private var stepFill = 0
  /// Channel of the next sample, since input can end mid-frame.
  private var channel = 0
  private(set) var peak: Float = 0

  init(sampleRate: Int, channels: Int) {
    self.channels = channels
    stepLength = max(1, Int((Double(sampleRate) / 10).rounded()))
    filters = Array(repeating: KWeighting(sampleRate: Double(sampleRate)), count: channels)
  }

  /// Adds interleaved samples.
  mutating func add(_ samples: ArraySlice<Float>) {
    if channels == 2 && channel == 0 {
      addStereo(samples)
      return
    }
    var ch = channel
    defer { channel = ch }
    for sample in samples {
      peak = max(peak, abs(sample))
      let y = filters[ch].process(Double(sample))
      stepSum += y * y
      ch += 1
      if ch == channels {
        ch = 0
        stepFill += 1
        if stepFill == stepLength {
          steps.append(stepSum / Double(stepLength))
          stepSum = 0
          stepFill = 0
        }
      }
    }
  }

  /// ``add(_:)`` for stereo starting on a left sample, filtering both channels of each
  /// frame together so their filters run in parallel.
  private mutating func addStereo(_ samples: ArraySlice<Float>) {
    var left = filters[0]
    var right = filters[1]
    var peak = self.peak
    var sum = stepSum
    var fill = stepFill
    let frames = samples.count / 2
    samples.withUnsafeBufferPointer { samples in
      for i in 0..<frames {
        let l = samples[2 * i]
        let r = samples[2 * i + 1]
        peak = max(peak, abs(l))
        peak = max(peak, abs(r))
        let yl = left.process(Double(l))
        let yr = right.process(Double(r))
        sum += yl * yl
        sum += yr * yr
        fill += 1
        if fill == stepLength {
          steps.append(sum / Double(stepLength))
          sum = 0
          fill = 0
        }
      }
    }
    filters[0] = left
    filters[1] = right
    self.peak = peak
    stepSum = sum
    stepFill = fill
    if samples.count % 2 == 1 {
      // Finish with the odd left sample, leaving the next call to start on the right.
      let sample = samples[samples.endIndex - 1]
      self.peak = max(self.peak, abs(sample))
      let y = filters[0].process(Double(sample))
      stepSum += y * y
      channel = 1
    }
  }

  /// Gated integrated loudness in LUFS, or `nil` for less than 400 ms of audio or silence.
  var integratedLoudness: Double? {
    guard steps.count >= 4 else { return nil }
    var blocks = [Double]()
    for i in 0...(steps.count - 4) {
      let sum: Double = steps[i] + steps[i + 1] + steps[i + 2] + steps[i + 3]
      blocks.append(sum / 4)
    }
    func loudness(_ meanSquare: Double) -> Double { -0.691 + 10 * log10(meanSquare) }
    let audible = blocks.filter { $0 > 0 && loudness($0) > -70 }
    guard !audible.isEmpty else { return nil }
    let relativeGate = loudness(audible.reduce(0, +) / Double(audible.count)) - 10
    let gated = audible.filter { loudness($0) > relativeGate }
    return loudness(gated.reduce(0, +) / Double(gated.count))
  }

  /// ReplayGain 2.0 track gain in dB: the change that brings the audio to -18 LUFS.
  var trackGain: Double? {
    integratedLoudness.map { -18 - $0 }
  }
}

/// The two-stage K-weighting filter of BS.1770, derived for any sample rate from the
/// analog prototype of the 48 kHz coefficients (as libebur128 does).
private struct KWeighting {
  private var shelf: Biquad
  private var highpass: Biquad

  init(sampleRate: Double) {
    // High shelf: about +4 dB above 1.5 kHz, modeling the head.
    var f0 = 1681.974450955533
    let gain = 3.999843853973347
    var q = 0.7071752369554196
    var k = tan(Double.pi * f0 / sampleRate)
    let vh = pow(10, gain / 20)
    let vb = pow(vh, 0.4996667741545416)
    var a0 = 1 + k / q + k * k
    shelf = Biquad(
      b0: (vh + vb * k / q + k * k) / a0,
      b1: 2 * (k * k - vh) / a0,
      b2: (vh - vb * k / q + k * k) / a0,
      a1: 2 * (k * k - 1) / a0,
      a2: (1 - k / q + k * k) / a0
    )
    // High pass at about 38 Hz (the RLB curve).
    f0 = 38.13547087602444
    q = 0.5003270373238773
    k = tan(Double.pi * f0 / sampleRate)
    a0 = 1 + k / q + k * k
    highpass = Biquad(b0: 1, b1: -2, b2: 1, a1: 2 * (k * k - 1) / a0, a2: (1 - k / q + k * k) / a0)
  }

  mutating func process(_ x: Double) -> Double {
    highpass.process(shelf.process(x))
  }
}

private struct Biquad {
  let b0, b1, b2, a1, a2: Double
  private var z1 = 0.0
  private var z2 = 0.0

  init(b0: Double, b1: Double, b2: Double, a1: Double, a2: Double) {
    self.b0 = b0
    self.b1 = b1
    self.b2 = b2
    self.a1 = a1
    self.a2 = a2
  }

  /// Transposed direct form II.
  @inline(__always)
  mutating func process(_ x: Double) -> Double {
    let y = b0 * x + z1
    z1 = b1 * x - a1 * y + z2
    z2 = b2 * x - a2 * y
    return y
  }
}
