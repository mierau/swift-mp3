// SwiftMP3
// Tables.swift
//
// MPEG-1 Layer III constants from ISO/IEC 11172-3.

enum MPEG {
  static let samplesPerFrame = 1152
  static let samplesPerGranule = 576
  static let subbands = 32

  /// Layer III bitrates in kbps, indexed by `bitrate_index`.
  static let bitrates = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]

  /// Supported sample rates in Hz, indexed by `sampling_frequency`.
  static let sampleRates = [44_100, 48_000, 32_000]

  /// Returns the `bitrate_index` whose bitrate is closest to `kbps`.
  static func bitrateIndex(for kbps: Int) -> Int {
    var best = 1
    for index in 1..<bitrates.count where abs(bitrates[index] - kbps) < abs(bitrates[best] - kbps) {
      best = index
    }
    return best
  }

  /// Returns the `sampling_frequency` index for a sample rate, or `nil` if unsupported.
  static func sampleRateIndex(for sampleRate: Int) -> Int? {
    sampleRates.firstIndex(of: sampleRate)
  }

  /// Frame length in bytes for a bitrate index, sample rate, and padding bit.
  static func frameLength(bitrateIndex: Int, sampleRate: Int, padding: Bool = false) -> Int {
    144_000 * bitrates[bitrateIndex] / sampleRate + (padding ? 1 : 0)
  }

  /// Scalefactor band boundaries for long blocks (Table B.8), 23 entries.
  static func longBands(sampleRateIndex: Int) -> [Int] {
    switch sampleRateIndex {
    case 1: return [0, 4, 8, 12, 16, 20, 24, 30, 36, 42, 50, 60, 72, 88, 106, 128, 156, 190, 230, 276, 330, 384, 576]
    case 2: return [0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 54, 66, 82, 102, 126, 156, 194, 240, 296, 364, 448, 550, 576]
    default: return [0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418, 576]
    }
  }

  /// Scalefactor band boundaries for short blocks (Table B.8), 14 entries per window.
  static func shortBands(sampleRateIndex: Int) -> [Int] {
    switch sampleRateIndex {
    case 1: return [0, 4, 8, 12, 16, 22, 28, 38, 50, 64, 80, 100, 126, 192]
    case 2: return [0, 4, 8, 12, 16, 22, 30, 42, 58, 78, 104, 138, 180, 192]
    default: return [0, 4, 8, 12, 16, 22, 30, 40, 52, 66, 84, 106, 136, 192]
    }
  }

  /// `(slen1, slen2)` for each `scalefac_compress` value.
  static let slen: [(Int, Int)] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (3, 0), (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3),
  ]

  /// Pre-emphasis table applied to long-block scalefactors when `preflag` is set.
  static let pretab = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2, 0]

  /// Default `(region0_count, region1_count)` by number of bands in the big-values region.
  static let subdivision: [(Int, Int)] = [
    (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (1, 1), (1, 1),
    (1, 2), (2, 2), (2, 3), (2, 3), (3, 4), (3, 4), (3, 4), (4, 5),
    (4, 5), (4, 6), (5, 6), (5, 6), (5, 7), (6, 7), (6, 7),
  ]
}

/// MDCT window shape for one granule (`block_type`).
enum BlockType: Int {
  case normal = 0
  case start = 1
  case short = 2
  case stop = 3
}
