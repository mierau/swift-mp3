// SwiftMP3
// Psychoacoustics.swift
//
// A compact psychoacoustic model that works on MDCT spectra. For each scalefactor band it
// estimates how much quantization noise can be hidden under the signal (simultaneous
// masking with a Bark-domain spreading function and a tonality-dependent offset), bounded
// below by the absolute threshold of hearing.

import Foundation

/// Allowed noise and perceptual entropy for one granule of one channel.
struct MaskingResult {
  /// Allowed noise energy per band. Long blocks use 22 bands; short blocks use
  /// 13 bands × 3 windows, indexed `band * 3 + window`.
  var allowedNoise: [Float]
  /// Signal energy per band, with the same layout as ``allowedNoise``.
  var energy: [Float]
  /// Perceptual entropy: an estimate of the bits needed to code the granule transparently.
  var perceptualEntropy: Float
}

/// Per-sample-rate tables for the psychoacoustic model.
struct PsychoacousticModel {
  let longBands: [Int]
  let shortBands: [Int]
  /// Absolute threshold of hearing per band (energy), long and short.
  private let longATH: [Float]
  private let shortATH: [Float]
  /// Spreading weights `[maskee * n + masker]`.
  private let longSpreading: [Float]
  private let shortSpreading: [Float]
  /// Bark value of each band's center frequency.
  private let longBark: [Float]
  private let shortBark: [Float]
  /// Multiplier applied to masking thresholds (from the quality setting).
  private let maskingScale: Float

  init(sampleRate: Int, sampleRateIndex: Int, quality: Int) {
    longBands = MPEG.longBands(sampleRateIndex: sampleRateIndex)
    shortBands = MPEG.shortBands(sampleRateIndex: sampleRateIndex)
    // Quality 0 lowers thresholds by 4 dB, quality 9 raises them by 5 dB.
    maskingScale = Float(pow(10, Double(quality - 4) / 10))

    let nyquist = Double(sampleRate) / 2
    func bark(_ f: Double) -> Double {
      13 * atan(0.00076 * f) + 3.5 * atan((f / 7500) * (f / 7500))
    }
    // Absolute threshold of hearing in dB SPL (Terhardt), clamped for the extremes.
    func athDB(_ f: Double) -> Double {
      let khz = max(f, 20) / 1000
      let db = 3.64 * pow(khz, -0.8) - 6.5 * exp(-0.6 * (khz - 3.3) * (khz - 3.3)) + 0.001 * pow(khz, 4)
      return min(db, 60)
    }
    // A full-scale sine is taken to be 90 dB SPL. `fullScale` is the MDCT energy
    // of a full-scale sine in one line for long and short blocks.
    func athEnergy(_ f: Double, fullScale: Double) -> Double {
      fullScale * pow(10, (athDB(f) - 90) / 10)
    }

    func tables(bands: [Int], lines: Int, fullScale: Double) -> (ath: [Float], spreading: [Float], bark: [Float]) {
      let count = bands.count - 1
      var ath = [Float](repeating: 0, count: count)
      var barks = [Float](repeating: 0, count: count)
      for b in 0..<count {
        var minimum = Double.greatestFiniteMagnitude
        for line in bands[b]..<bands[b + 1] {
          let f = (Double(line) + 0.5) * nyquist / Double(lines)
          minimum = min(minimum, athEnergy(f, fullScale: fullScale))
        }
        ath[b] = Float(minimum * Double(bands[b + 1] - bands[b]))
        let center = Double(bands[b] + bands[b + 1]) / 2 * nyquist / Double(lines)
        barks[b] = Float(bark(center))
      }
      var spreading = [Float](repeating: 0, count: count * count)
      for maskee in 0..<count {
        for masker in 0..<count {
          let dz = Double(barks[maskee] - barks[masker]) + 0.474
          let db = 15.81 + 7.5 * dz - 17.5 * (1 + dz * dz).squareRoot()
          spreading[maskee * count + masker] = db < -60 ? 0 : Float(pow(10, db / 10))
        }
      }
      return (ath, spreading, barks)
    }

    let long = tables(bands: longBands, lines: 576, fullScale: Self.fullScaleLong)
    longATH = long.ath
    longSpreading = long.spreading
    longBark = long.bark
    let short = tables(bands: shortBands, lines: 192, fullScale: Self.fullScaleShort)
    shortATH = short.ath
    shortSpreading = short.spreading
    shortBark = short.bark
  }

  /// Approximate MDCT energy of a full-scale sine, concentrated in one band.
  static let fullScaleLong: Double = 1
  static let fullScaleShort: Double = 1

  /// Computes masking for a long-block (normal, start, or stop) granule.
  ///
  /// - Parameters:
  ///   - spectrum: 576 MDCT lines.
  ///   - previous: The previous granule's allowed noise for this channel, used to limit
  ///     pre-echo. Updated with this granule's thresholds.
  func analyzeLong(_ spectrum: UnsafePointer<Float>, previous: inout [Float]) -> MaskingResult {
    let count = longBands.count - 1
    var energy = [Float](repeating: 0, count: count)
    var tonalOffset = [Float](repeating: 0, count: count)
    for b in 0..<count {
      let (e, offset) = bandStatistics(spectrum, from: longBands[b], to: longBands[b + 1], bark: longBark[b])
      energy[b] = e
      tonalOffset[b] = offset
    }
    var allowed = spread(energy: energy, offset: tonalOffset, spreading: longSpreading)
    var pe: Float = 0
    for b in 0..<count {
      var threshold = allowed[b] * maskingScale
      if previous.count == count {
        // Pre-echo control: the threshold cannot jump more than 6 dB above the last one.
        threshold = min(threshold, 4 * previous[b] + longATH[b])
      }
      allowed[b] = max(threshold, longATH[b])
      pe += entropy(energy: energy[b], allowed: allowed[b], width: longBands[b + 1] - longBands[b])
    }
    previous = allowed
    return MaskingResult(allowedNoise: allowed, energy: energy, perceptualEntropy: pe)
  }

  /// Computes masking for a short-block granule (spectrum in band, window, line order).
  func analyzeShort(_ spectrum: UnsafePointer<Float>, previous: inout [Float]) -> MaskingResult {
    let count = shortBands.count - 1
    var allowed = [Float](repeating: 0, count: count * 3)
    var energy = [Float](repeating: 0, count: count * 3)
    var pe: Float = 0
    var windowEnergy = [Float](repeating: 0, count: count)
    var tonalOffset = [Float](repeating: 0, count: count)
    for w in 0..<3 {
      for b in 0..<count {
        let width = shortBands[b + 1] - shortBands[b]
        let start = shortBands[b] * 3 + w * width
        let (e, offset) = bandStatistics(spectrum, from: start, to: start + width, bark: shortBark[b])
        windowEnergy[b] = e
        tonalOffset[b] = offset
      }
      let thresholds = spread(energy: windowEnergy, offset: tonalOffset, spreading: shortSpreading)
      for b in 0..<count {
        let value = max(thresholds[b] * maskingScale, shortATH[b])
        allowed[b * 3 + w] = value
        energy[b * 3 + w] = windowEnergy[b]
        pe += entropy(energy: windowEnergy[b], allowed: value, width: shortBands[b + 1] - shortBands[b])
      }
    }
    // A transient resets pre-echo history; the next long block starts fresh.
    previous = []
    return MaskingResult(allowedNoise: allowed, energy: energy, perceptualEntropy: pe)
  }

  /// Energy and masking offset of one band.
  ///
  /// The offset is the signal-to-mask ratio for a masker of this band: 6 dB
  /// for noise and 14.5 + Bark dB (at least 29 dB) for a pure tone, interpolated by spectral flatness.
  @inline(__always)
  private func bandStatistics(_ x: UnsafePointer<Float>, from start: Int, to end: Int, bark: Float) -> (Float, Float) {
    var sum: Float = 0
    var logSum: Float = 0
    for i in start..<end {
      let e = x[i] * x[i]
      sum += e
      logSum += log2(e + 1e-30)
    }
    let n = Float(end - start)
    guard sum > 1e-24 else { return (sum, 1) }
    // Spectral flatness in dB: 0 for white noise, strongly negative for a tone.
    let flatness = 10 * 0.30103 * (logSum / n - log2(sum / n))
    let tonality = min(max(flatness / -6, 0), 1)
    let offsetDB = tonality * max(29, 14.5 + bark) + (1 - tonality) * 6
    return (sum, pow(10, -offsetDB / 10))
  }

  /// Spreads each band's energy, attenuated by its masking offset, across neighboring bands.
  @inline(__always)
  private func spread(energy: [Float], offset: [Float], spreading: [Float]) -> [Float] {
    let count = energy.count
    var result = [Float](repeating: 0, count: count)
    for maskee in 0..<count {
      var sum: Float = 0
      let row = maskee * count
      for masker in 0..<count {
        sum += spreading[row + masker] * energy[masker] * offset[masker]
      }
      result[maskee] = sum
    }
    return result
  }

  @inline(__always)
  private func entropy(energy: Float, allowed: Float, width: Int) -> Float {
    guard energy > allowed else { return 0 }
    return Float(width) * 0.5 * log2(energy / allowed)
  }
}
