// SwiftMP3
// Tables.swift
//
// Layer III constants from ISO/IEC 11172-3 (MPEG-1) and 13818-3 (MPEG-2 and 2.5).

enum MPEG {
  static let samplesPerGranule = 576
  static let subbands = 32

  /// `(slen1, slen2)` for each MPEG-1 `scalefac_compress` value.
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

/// MPEG audio version: sets the sample rates, bitrates, and frame layout.
enum MPEGVersion {
  case mpeg1
  case mpeg2
  case mpeg25

  /// Sample rates in Hz, indexed by `sampling_frequency`.
  var sampleRates: [Int] {
    switch self {
    case .mpeg1: return [44_100, 48_000, 32_000]
    case .mpeg2: return [22_050, 24_000, 16_000]
    case .mpeg25: return [11_025, 12_000, 8_000]
    }
  }

  /// Layer III bitrates in kbps, indexed by `bitrate_index`.
  var bitrates: [Int] {
    switch self {
    case .mpeg1: return [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
    case .mpeg2, .mpeg25: return [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
    }
  }

  /// The two-bit version ID in the frame header.
  var headerBits: Int {
    switch self {
    case .mpeg1: return 0b11
    case .mpeg2: return 0b10
    case .mpeg25: return 0b00
    }
  }

  /// The version whose sample rates include `sampleRate`.
  init?(sampleRate: Int) {
    guard let version = [MPEGVersion.mpeg1, .mpeg2, .mpeg25].first(where: { $0.sampleRates.contains(sampleRate) }) else {
      return nil
    }
    self = version
  }

  /// Every supported sample rate.
  static let allSampleRates = [MPEGVersion.mpeg1, .mpeg2, .mpeg25].flatMap(\.sampleRates)
}

/// The fixed layout of a stream: version, sample rate, and channel count.
struct StreamFormat {
  let version: MPEGVersion
  let sampleRate: Int
  /// `sampling_frequency` in the frame header.
  let sampleRateIndex: Int
  let channels: Int
  /// Granules per frame: 2 for MPEG-1, 1 for MPEG-2 and 2.5.
  let granules: Int
  /// Side information bytes per frame.
  let sideInfoSize: Int
  /// Scalefactor band boundaries for long blocks (23 entries) and short blocks (14 entries).
  let longBands: [Int]
  let shortBands: [Int]

  init?(sampleRate: Int, channels: Int) {
    guard let version = MPEGVersion(sampleRate: sampleRate) else { return nil }
    self.version = version
    self.sampleRate = sampleRate
    self.channels = channels
    sampleRateIndex = version.sampleRates.firstIndex(of: sampleRate)!
    granules = version == .mpeg1 ? 2 : 1
    switch (version, channels) {
    case (.mpeg1, 1): sideInfoSize = 17
    case (.mpeg1, _): sideInfoSize = 32
    case (_, 1): sideInfoSize = 9
    default: sideInfoSize = 17
    }
    (longBands, shortBands) = Self.bands(sampleRate: sampleRate)
  }

  var isMPEG1: Bool { version == .mpeg1 }
  var samplesPerFrame: Int { granules * MPEG.samplesPerGranule }
  var bitrates: [Int] { version.bitrates }

  /// Returns the `bitrate_index` for a bitrate, or `nil` if the version doesn't have it.
  func bitrateIndex(for kbps: Int) -> Int? {
    guard kbps > 0 else { return nil }
    return bitrates.firstIndex(of: kbps)
  }

  /// Frame length in bytes, before padding, times the sample rate.
  func scaledFrameLength(bitrateIndex: Int) -> Int {
    samplesPerFrame / 8 * 1000 * bitrates[bitrateIndex]
  }

  /// Frame length in bytes for a bitrate index and padding bit.
  func frameLength(bitrateIndex: Int, padding: Bool = false) -> Int {
    scaledFrameLength(bitrateIndex: bitrateIndex) / sampleRate + (padding ? 1 : 0)
  }

  /// Bytes of a frame left for main data after the header, CRC, and side information.
  func mainDataSize(frameLength: Int, crc: Bool) -> Int {
    frameLength - 4 - (crc ? 2 : 0) - sideInfoSize
  }

  /// Largest `main_data_begin`, in bytes.
  var maxMainDataBegin: Int { isMPEG1 ? 511 : 255 }

  /// Scalefactor band boundaries (ISO/IEC 11172-3 Table B.8, 13818-3 Table B.2).
  private static func bands(sampleRate: Int) -> (long: [Int], short: [Int]) {
    switch sampleRate {
    case 48_000:
      return ([0, 4, 8, 12, 16, 20, 24, 30, 36, 42, 50, 60, 72, 88, 106, 128, 156, 190, 230, 276, 330, 384, 576],
              [0, 4, 8, 12, 16, 22, 28, 38, 50, 64, 80, 100, 126, 192])
    case 32_000:
      return ([0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 54, 66, 82, 102, 126, 156, 194, 240, 296, 364, 448, 550, 576],
              [0, 4, 8, 12, 16, 22, 30, 42, 58, 78, 104, 138, 180, 192])
    case 22_050:
      return ([0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 116, 140, 168, 200, 238, 284, 336, 396, 464, 522, 576],
              [0, 4, 8, 12, 18, 24, 32, 42, 56, 74, 100, 132, 174, 192])
    case 24_000:
      return ([0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 114, 136, 162, 194, 232, 278, 332, 394, 464, 540, 576],
              [0, 4, 8, 12, 18, 26, 36, 48, 62, 80, 104, 136, 180, 192])
    case 16_000, 11_025, 12_000:
      return ([0, 6, 12, 18, 24, 30, 36, 44, 54, 66, 80, 96, 116, 140, 168, 200, 238, 284, 336, 396, 464, 522, 576],
              [0, 4, 8, 12, 18, 26, 36, 48, 62, 80, 104, 134, 174, 192])
    case 8_000:
      return ([0, 12, 24, 36, 48, 60, 72, 88, 108, 132, 160, 192, 232, 280, 336, 400, 476, 566, 568, 570, 572, 574, 576],
              [0, 8, 16, 24, 36, 52, 72, 96, 124, 160, 162, 164, 166, 192])
    default:
      return ([0, 4, 8, 12, 16, 20, 24, 30, 36, 44, 52, 62, 74, 90, 110, 134, 162, 196, 238, 288, 342, 418, 576],
              [0, 4, 8, 12, 16, 22, 30, 40, 52, 66, 84, 106, 136, 192])
    }
  }
}

/// MDCT window shape for one granule (`block_type`).
enum BlockType: Int {
  case normal = 0
  case start = 1
  case short = 2
  case stop = 3
}
