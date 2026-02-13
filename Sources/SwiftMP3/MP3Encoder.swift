// SwiftMP3
// MP3Encoder.swift

import Foundation
import Accelerate

/// ID3v2.3 metadata for MP3 files.
public struct ID3Tag: Sendable, Equatable {
  /// The track title.
  public var title: String?
  /// The artist name.
  public var artist: String?
  /// The album name.
  public var album: String?
  /// The track number.
  public var track: UInt16?
  /// The total number of tracks.
  public var trackTotal: UInt16?
  /// The release year.
  public var year: UInt16?
  /// The genre name.
  public var genre: String?
  /// A comment.
  public var comment: String?
  /// Album artwork image data (JPEG or PNG).
  public var albumArt: Data?
  /// MIME type for the album artwork.
  public var albumArtMIMEType: String

  /// Creates a new ID3 tag with the given metadata.
  public init(
    title: String? = nil,
    artist: String? = nil,
    album: String? = nil,
    track: UInt16? = nil,
    trackTotal: UInt16? = nil,
    year: UInt16? = nil,
    genre: String? = nil,
    comment: String? = nil,
    albumArt: Data? = nil,
    albumArtMIMEType: String = "image/jpeg"
  ) {
    self.title = title
    self.artist = artist
    self.album = album
    self.track = track
    self.trackTotal = trackTotal
    self.year = year
    self.genre = genre
    self.comment = comment
    self.albumArt = albumArt
    self.albumArtMIMEType = albumArtMIMEType
  }
}

/// Configuration options for the MP3 encoder.
public struct MP3EncoderOptions: Sendable, Equatable {
  /// The channel mode for the encoded audio.
  public enum Mode: String, Sendable, Equatable {
    case mono
    case stereo
    case jointStereo
  }

  /// Sample rate in Hz (e.g. 44100, 48000, 32000).
  public var sampleRate: Int
  /// Target bitrate in kbps for CBR, or base bitrate for VBR.
  public var bitrateKbps: Int
  /// Whether to use variable bitrate encoding.
  public var vbr: Bool
  /// Channel mode (mono, stereo, or joint stereo).
  public var mode: Mode
  /// Encoding quality from 0 (highest) to 9 (lowest).
  public var quality: Int
  /// Whether to include CRC error protection in each frame.
  public var crcProtected: Bool
  /// Whether to set the original bit in the frame header.
  public var original: Bool
  /// Whether to set the copyright bit in the frame header.
  public var copyright: Bool
  /// ID3v2.3 metadata tag to embed at the start of the file.
  public var id3Tag: ID3Tag?

  /// Creates a new set of encoder options.
  /// - Parameters:
  ///   - sampleRate: Sample rate in Hz. Defaults to 44100.
  ///   - bitrateKbps: Target bitrate in kbps. Defaults to 128.
  ///   - vbr: Enable variable bitrate encoding. Defaults to `false`.
  ///   - mode: Channel mode. Defaults to `.stereo`.
  ///   - quality: Quality level from 0 (best) to 9 (smallest). Defaults to 5.
  ///   - crcProtected: Enable CRC protection. Defaults to `false`.
  ///   - original: Set the original flag. Defaults to `true`.
  ///   - copyright: Set the copyright flag. Defaults to `false`.
  ///   - id3Tag: ID3v2.3 metadata to embed. Defaults to `nil`.
  public init(
    sampleRate: Int = 44_100,
    bitrateKbps: Int = 128,
    vbr: Bool = false,
    mode: Mode = .stereo,
    quality: Int = 5,
    crcProtected: Bool = false,
    original: Bool = true,
    copyright: Bool = false,
    id3Tag: ID3Tag? = nil
  ) {
    self.sampleRate = sampleRate
    self.bitrateKbps = bitrateKbps
    self.vbr = vbr
    self.mode = mode
    self.quality = max(0, min(quality, 9))
    self.crcProtected = crcProtected
    self.original = original
    self.copyright = copyright
    self.id3Tag = id3Tag
  }
}

/// MPEG-1 Layer III (MP3) encoder.
///
/// A stateless, `Sendable` value type that holds encoding configuration. Create an
/// ``EncoderSession`` via ``newSession()`` for synchronous encoding, or use the async
/// ``encode(_:)`` and ``encode(_:to:)`` methods for streaming and file output.
///
/// Usage:
/// ```swift
/// let encoder = MP3Encoder(options: MP3EncoderOptions())
/// var session = encoder.newSession()
/// var mp3Data = session.encode(samples: pcmSamples)
/// mp3Data.append(session.flush())
/// let finalData = session.generateXingHeader() + mp3Data
/// ```
public struct MP3Encoder: Sendable {
  /// The encoding configuration.
  public let options: MP3EncoderOptions

  /// Creates a new MP3 encoder with the given options.
  /// - Parameter options: Configuration for sample rate, bitrate, channel mode, etc.
  public init(options: MP3EncoderOptions) {
    self.options = options
  }

  /// Creates a mutable encoding session for synchronous use.
  public func newSession() -> EncoderSession {
    EncoderSession(options: self.options)
  }

  /// Streaming encode: yields MP3 frames as `Data` chunks. No Xing header is included.
  ///
  /// - Parameter input: An async sequence of interleaved PCM float sample buffers.
  /// - Returns: An `AsyncThrowingStream` that yields encoded MP3 frame data.
  public func encode<S: AsyncSequence & Sendable>(
    _ input: S
  ) -> AsyncThrowingStream<Data, Error> where S.Element == [Float] {
    let options = self.options
    return AsyncThrowingStream { continuation in
      let task = Task {
        var state = EncoderSession(options: options)
        do {
          for try await samples in input {
            try Task.checkCancellation()
            let data = state.encode(samples: samples)
            if !data.isEmpty {
              continuation.yield(data)
            }
          }
          let final = state.flush()
          if !final.isEmpty {
            continuation.yield(final)
          }
          continuation.finish()
        } catch {
          continuation.finish(throwing: error)
        }
      }
      continuation.onTermination = { _ in
        task.cancel()
      }
    }
  }

  /// File encode: writes MP3 frames incrementally to disk with a Xing header.
  ///
  /// Creates the file at `url`, writes encoded frames as they are produced, then
  /// seeks back to the beginning to write the Xing header for accurate seeking.
  ///
  /// - Parameters:
  ///   - input: An async sequence of interleaved PCM float sample buffers.
  ///   - url: The file URL to write the MP3 output to.
  public func encode<S: AsyncSequence & Sendable>(
    _ input: S, to url: URL
  ) async throws where S.Element == [Float] {
    var state = EncoderSession(options: self.options)

    // Generate ID3 tag if metadata is configured
    let id3Data = state.generateID3Tag()

    // Calculate Xing frame size for the placeholder
    let bitrateIndex = MP3Tables.bitrateIndex(for: self.options.bitrateKbps, sampleRate: self.options.sampleRate)
    let bitrateValue = MP3Tables.bitrateValue(for: bitrateIndex)
    let xingFrameSize = (144 * bitrateValue * 1000) / self.options.sampleRate

    // Create the file: [ID3 tag] + [Xing placeholder]
    var fileHeader = id3Data
    fileHeader.append(Data(count: xingFrameSize))
    try fileHeader.write(to: url)

    let handle = try FileHandle(forWritingTo: url)
    defer { try? handle.close() }

    // Seek past ID3 tag and Xing placeholder
    try handle.seek(toOffset: UInt64(id3Data.count + xingFrameSize))

    for try await samples in input {
      try Task.checkCancellation()
      let data = state.encode(samples: samples)
      if !data.isEmpty {
        try handle.write(contentsOf: data)
      }
    }

    let finalData = state.flush()
    if !finalData.isEmpty {
      try handle.write(contentsOf: finalData)
    }

    // Write the real Xing header after the ID3 tag
    let xingHeader = state.generateXingHeader()
    try handle.seek(toOffset: UInt64(id3Data.count))
    try handle.write(contentsOf: xingHeader)
  }
}

/// Mutable encoding state for synchronous MP3 encoding.
///
/// Holds all per-session state (PCM buffer, filterbank buffers, MDCT overlap, frame
/// statistics, etc.). Created via ``MP3Encoder/newSession()``.
public struct EncoderSession {
  private static let samplesPerFrame = 1152
  private static let subbands = 32
  private static let samplesPerGranule = 576

  private let options: MP3EncoderOptions
  private var pcmBuffer: [Float] = []
  private var vbrState = VBRState()
  private var reservoir = BitReservoir()

  private var filterbankBuffers: [[Float]] = []
  private var mdctOverlap: [[[Float]]] = []

  private var workingSpectrum: [Float] = []
  private var workingQuantized: [Int] = []
  private var workingSubbands: [[Float]] = []

  private var frameCount: UInt32 = 0
  private var totalBytes: UInt32 = 0
  private var frameSizes: [Int] = []

  /// The number of frames encoded so far.
  public var encodedFrameCount: UInt32 { frameCount }

  /// The total bytes of encoded audio data so far (excluding the Xing header).
  public var encodedByteCount: UInt32 { totalBytes }

  /// Creates a new encoding state with the given options.
  /// - Parameter options: Configuration for sample rate, bitrate, channel mode, etc.
  init(options: MP3EncoderOptions) {
    self.options = options
    let channels = options.mode == .mono ? 1 : 2

    self.filterbankBuffers = Array(repeating: Array(repeating: 0, count: 512), count: channels)
    self.mdctOverlap = Array(
      repeating: Array(repeating: Array(repeating: 0, count: 18), count: Self.subbands),
      count: channels
    )

    self.workingSpectrum = [Float](repeating: 0, count: Self.samplesPerGranule)
    self.workingQuantized = [Int](repeating: 0, count: Self.samplesPerGranule)
    self.workingSubbands = Array(repeating: [Float](repeating: 0, count: 18), count: Self.subbands)
    self.frameSizes.reserveCapacity(10000)
  }

  /// Appends interleaved PCM samples to the internal buffer and encodes any complete frames.
  ///
  /// Samples should be interleaved for stereo (L, R, L, R, ...) and normalized to [-1.0, 1.0].
  /// Any leftover samples that don't fill a complete frame are buffered for the next call.
  ///
  /// - Parameter samples: Interleaved PCM float samples.
  /// - Returns: Encoded MP3 data for any complete frames, or empty data if no frames were completed.
  public mutating func encode(samples: [Float]) -> Data {
    self.pcmBuffer.append(contentsOf: samples)
    var output = Data()
    let channels = self.options.mode == .mono ? 1 : 2
    let frameSampleCount = Self.samplesPerFrame * channels

    while self.pcmBuffer.count >= frameSampleCount {
      let frameSamples = Array(self.pcmBuffer.prefix(frameSampleCount))
      self.pcmBuffer.removeFirst(frameSampleCount)
      output.append(self.encodeFrame(samples: frameSamples))
    }

    return output
  }

  /// Encodes any remaining buffered samples as a final frame, zero-padding if necessary.
  ///
  /// Call this once after all samples have been passed to ``encode(samples:)``.
  ///
  /// - Returns: Encoded MP3 data for the final frame, or empty data if the buffer was empty.
  public mutating func flush() -> Data {
    if self.pcmBuffer.isEmpty {
      return Data()
    }
    let channels = self.options.mode == .mono ? 1 : 2
    let frameSampleCount = Self.samplesPerFrame * channels
    let needed = frameSampleCount - self.pcmBuffer.count
    if needed > 0 {
      self.pcmBuffer.append(contentsOf: repeatElement(0, count: needed))
    }
    let frameSamples = self.pcmBuffer
    self.pcmBuffer.removeAll()
    return self.encodeFrame(samples: frameSamples)
  }

  /// Generates the ID3v2.3 tag data for the configured metadata.
  ///
  /// Returns empty `Data` if no ID3 tag is configured or all fields are empty.
  public func generateID3Tag() -> Data {
    guard let tag = self.options.id3Tag else { return Data() }
    return ID3TagWriter.build(tag: tag)
  }

  /// Generates a Xing/Info header frame for seeking and duration metadata.
  ///
  /// Call this after encoding is complete to get accurate frame and byte counts.
  /// The returned data should be prepended to the encoded audio data. Uses the
  /// "Xing" tag for VBR and "Info" tag for CBR streams.
  ///
  /// - Returns: A complete MP3 frame containing the Xing/Info header.
  public func generateXingHeader() -> Data {
    let channels = self.options.mode == .mono ? 1 : 2
    let sideInfoSize = channels == 1 ? 17 : 32

    let bitrateIndex = MP3Tables.bitrateIndex(for: self.options.bitrateKbps, sampleRate: self.options.sampleRate)
    let sampleRateIndex = MP3Tables.sampleRateIndex(for: self.options.sampleRate)
    let bitrateValue = MP3Tables.bitrateValue(for: bitrateIndex)
    let frameSize = (144 * bitrateValue * 1000) / self.options.sampleRate

    let (modeBits, modeExtension) = MP3Tables.modeBits(for: self.options.mode)

    // MPEG-1 Layer III frame header
    var header = BitstreamWriter()
    header.write(bits: 0x7FF, count: 11)   // Sync word
    header.write(bits: 0b11, count: 2)     // MPEG Version 1
    header.write(bits: 0b01, count: 2)     // Layer III
    header.write(bits: 1, count: 1)        // No CRC
    header.write(bits: bitrateIndex, count: 4)
    header.write(bits: sampleRateIndex, count: 2)
    header.write(bits: 0, count: 1)        // No padding
    header.write(bits: 0, count: 1)        // Private bit
    header.write(bits: modeBits, count: 2)
    header.write(bits: modeExtension, count: 2)
    header.write(bits: 0, count: 1)        // Not copyrighted
    header.write(bits: 1, count: 1)        // Original
    header.write(bits: 0, count: 2)        // No emphasis

    var frame = Data()
    frame.append(header.data)
    frame.append(contentsOf: repeatElement(UInt8(0), count: sideInfoSize))

    let tag = self.options.vbr ? "Xing" : "Info"
    frame.append(contentsOf: tag.utf8)

    // Flags: frames present, bytes present, TOC present
    let flags: UInt32 = 0x07
    frame.append(contentsOf: flags.bigEndianBytes)

    let totalFrames = self.frameCount + 1
    frame.append(contentsOf: totalFrames.bigEndianBytes)

    let xingFrameSize = UInt32(frameSize)
    let totalByteCount = self.totalBytes + xingFrameSize
    frame.append(contentsOf: totalByteCount.bigEndianBytes)

    let toc = self.generateTOC()
    frame.append(contentsOf: toc)

    if frame.count < frameSize {
      frame.append(contentsOf: repeatElement(UInt8(0), count: frameSize - frame.count))
    }

    return frame
  }

  /// Generates the 100-byte TOC (Table of Contents) for seeking.
  private func generateTOC() -> [UInt8] {
    guard !self.frameSizes.isEmpty else {
      return (0..<100).map { UInt8($0 * 255 / 99) }
    }

    var cumulative = [Int]()
    var sum = 0
    for size in self.frameSizes {
      sum += size
      cumulative.append(sum)
    }

    let totalBytes = sum
    guard totalBytes > 0 else {
      return (0..<100).map { UInt8($0 * 255 / 99) }
    }

    var toc = [UInt8]()
    for percent in 0..<100 {
      let targetFrame = (percent * self.frameSizes.count) / 100
      let bytePosition = targetFrame > 0 ? cumulative[targetFrame - 1] : 0
      let scaled = (bytePosition * 255) / totalBytes
      toc.append(UInt8(min(scaled, 255)))
    }

    return toc
  }

  /// Encodes a single MPEG-1 Layer III frame from PCM samples.
  private mutating func encodeFrame(samples: [Float]) -> Data {
    let channels = self.options.mode == .mono ? 1 : 2
    let frameEnergy = FrameAnalysis.energy(samples: samples)
    let targetBitrate = self.options.vbr
      ? self.vbrState.chooseBitrate(base: self.options.bitrateKbps, energy: frameEnergy, quality: self.options.quality)
      : self.options.bitrateKbps
    let bitrateIndex = MP3Tables.bitrateIndex(for: targetBitrate, sampleRate: self.options.sampleRate)
    let sampleRateIndex = MP3Tables.sampleRateIndex(for: self.options.sampleRate)

    let bitrateValue = MP3Tables.bitrateValue(for: bitrateIndex)
    let baseFrameSize = (144 * bitrateValue * 1000) / self.options.sampleRate
    let sideInfoSize = channels == 1 ? 17 : 32
    let headerSize = 4
    let crcSize = self.options.crcProtected ? 2 : 0
    let padding = 0
    let frameSize = baseFrameSize + padding
    let mainDataSize = frameSize - headerSize - crcSize - sideInfoSize

    let (modeBits, modeExtension) = MP3Tables.modeBits(for: self.options.mode)

    // MPEG-1 Layer III frame header
    var header = BitstreamWriter()
    header.write(bits: 0x7FF, count: 11)
    header.write(bits: 0b11, count: 2)
    header.write(bits: 0b01, count: 2)
    header.write(bits: self.options.crcProtected ? 0 : 1, count: 1)
    header.write(bits: bitrateIndex, count: 4)
    header.write(bits: sampleRateIndex, count: 2)
    header.write(bits: padding, count: 1)
    header.write(bits: 0, count: 1)
    header.write(bits: modeBits, count: 2)
    header.write(bits: modeExtension, count: 2)
    header.write(bits: self.options.copyright ? 1 : 0, count: 1)
    header.write(bits: self.options.original ? 1 : 0, count: 1)
    header.write(bits: 0, count: 2)

    let (mainData, sideInfo) = self.buildMainData(
      samples: samples,
      channels: channels,
      targetMainDataSize: mainDataSize
    )

    var frame = Data()
    frame.append(header.data)
    if self.options.crcProtected {
      let crc = CRC16.mpeg(data: frame)
      frame.append(contentsOf: [UInt8(crc >> 8), UInt8(crc & 0xFF)])
    }
    frame.append(sideInfo)

    var paddedMainData = mainData
    if paddedMainData.count < mainDataSize {
      paddedMainData.append(contentsOf: repeatElement(0, count: mainDataSize - paddedMainData.count))
    } else if paddedMainData.count > mainDataSize {
      paddedMainData = paddedMainData.prefix(mainDataSize)
    }
    frame.append(paddedMainData)

    self.frameCount += 1
    self.totalBytes += UInt32(frame.count)
    self.frameSizes.append(frame.count)

    return frame
  }

  /// Builds the MPEG-1 Layer III side information for a frame.
  private func buildSideInfo(channels: Int, granules: [[GranuleInfo]], scfsi: [[Int]]) -> Data {
    var writer = BitstreamWriter()
    let sideInfoBits = channels == 1 ? 136 : 256

    // main_data_begin = 0 (no bit reservoir, each frame is self-contained)
    writer.write(bits: 0, count: 9)

    let privateBits = channels == 1 ? 5 : 3
    writer.write(bits: 0, count: privateBits)

    for ch in 0..<channels {
      for scfsiBand in 0..<4 {
        writer.write(bits: scfsi[ch][scfsiBand], count: 1)
      }
    }

    for gr in 0..<2 {
      for ch in 0..<channels {
        let info = granules[gr][ch]
        writer.write(bits: info.part23Length, count: 12)
        writer.write(bits: info.bigValues, count: 9)
        writer.write(bits: info.globalGain, count: 8)
        writer.write(bits: info.scalefacCompress, count: 4)
        writer.write(bits: info.windowSwitching, count: 1)
        if info.windowSwitching == 1 {
          writer.write(bits: info.blockType, count: 2)
          writer.write(bits: info.mixedBlockFlag, count: 1)
          writer.write(bits: info.tableSelect[0], count: 5)
          writer.write(bits: info.tableSelect[1], count: 5)
          writer.write(bits: info.subblockGain[0], count: 3)
          writer.write(bits: info.subblockGain[1], count: 3)
          writer.write(bits: info.subblockGain[2], count: 3)
        } else {
          writer.write(bits: info.tableSelect[0], count: 5)
          writer.write(bits: info.tableSelect[1], count: 5)
          writer.write(bits: info.tableSelect[2], count: 5)
          writer.write(bits: info.region0Count, count: 4)
          writer.write(bits: info.region1Count, count: 3)
        }
        writer.write(bits: info.preflag, count: 1)
        writer.write(bits: info.scalefacScale, count: 1)
        writer.write(bits: info.count1TableSelect, count: 1)
      }
    }

    writer.padToByte()
    let data = writer.data
    if data.count * 8 < sideInfoBits {
      var padded = data
      let bytesNeeded = (sideInfoBits / 8) - data.count
      padded.append(contentsOf: repeatElement(0, count: bytesNeeded))
      return padded
    }
    return data
  }

  /// Builds the main data and side information for a single frame.
  private mutating func buildMainData(
    samples: [Float],
    channels: Int,
    targetMainDataSize: Int
  ) -> (Data, Data) {
    let perChannel = self.deinterleave(samples: samples, channels: channels)
    let stereoDecision = StereoDecision.make(
      mode: self.options.mode,
      left: perChannel.first ?? [],
      right: perChannel.dropFirst().first ?? []
    )
    var granules: [[GranuleInfo]] = Array(
      repeating: Array(repeating: GranuleInfo(), count: channels),
      count: 2
    )
    let scfsi = Array(repeating: Array(repeating: 0, count: 4), count: channels)
    var writer = BitstreamWriter()

    let totalMainDataBits = targetMainDataSize * 8
    let granuleCount = 2 * channels
    let bitsPerGranule = totalMainDataBits / granuleCount

    for granuleIndex in 0..<2 {
      for channelIndex in 0..<channels {
        let channelSamples = stereoDecision.samples(for: channelIndex)

        let granuleStart = granuleIndex * Self.samplesPerGranule
        let granuleEnd = min(granuleStart + Self.samplesPerGranule, channelSamples.count)
        let granuleSamples: [Float]
        if granuleStart < channelSamples.count {
          granuleSamples = Array(channelSamples[granuleStart..<granuleEnd])
        } else {
          granuleSamples = Array(repeating: 0, count: Self.samplesPerGranule)
        }

        let spectrum = self.encodeSpectrum(
          granuleSamples,
          channel: channelIndex,
          sampleRate: self.options.sampleRate
        )

        self.vbrState.update(
          globalGain: spectrum.globalGain,
          energy: FrameAnalysis.energy(samples: granuleSamples)
        )

        let (finalGain, quantized, huffmanBits) = self.quantizeToFitBudget(
          spectral: spectrum.spectral,
          scalefactors: spectrum.scalefactors,
          thresholds: spectrum.maskingThresholds,
          initialGain: spectrum.globalGain,
          maxBits: bitsPerGranule,
          writer: &writer
        )

        let scalefacCompress = 0

        let preflag = PreEmphasis.shouldEnable(
          spectral: spectrum.spectral,
          scalefactors: spectrum.scalefactors
        ) ? 1 : 0

        var lastNonZero = 0
        for i in stride(from: quantized.count - 1, through: 0, by: -1) {
          if quantized[i] != 0 {
            lastNonZero = i + 1
            break
          }
        }
        let significantCount = (lastNonZero + 1) & ~1
        let bigValues = min(significantCount / 2, 288)

        let (region0Count, region1Count) = self.calculateRegionCounts(
          bigValues: bigValues,
          sampleRate: self.options.sampleRate
        )

        let part23Length = huffmanBits

        let info = GranuleInfo(
          part23Length: part23Length,
          bigValues: bigValues,
          globalGain: finalGain,
          scalefacCompress: scalefacCompress,
          windowSwitching: spectrum.windowSwitching,
          blockType: spectrum.blockType,
          mixedBlockFlag: spectrum.mixedBlockFlag,
          tableSelect: [15, 15, 15],
          subblockGain: spectrum.subblockGain,
          region0Count: region0Count,
          region1Count: region1Count,
          preflag: preflag,
          scalefacScale: 0,
          count1TableSelect: 0
        )
        granules[granuleIndex][channelIndex] = info
      }
    }

    writer.padToByte()
    let mainData = self.reservoir.consume(with: writer.data)
    let sideInfo = self.buildSideInfo(channels: channels, granules: granules, scfsi: scfsi)
    return (mainData, sideInfo)
  }

  /// Iteratively adjusts global gain to quantize spectral data within a bit budget.
  private func quantizeToFitBudget(
    spectral: [Float],
    scalefactors: [Float],
    thresholds: [Float],
    initialGain: Int,
    maxBits: Int,
    writer: inout BitstreamWriter
  ) -> (Int, [Int], Int) {
    var gain = min(max(initialGain, 0), 255)
    var quantized = [Int]()
    var estimatedBits = 0
    let maxIterations = 20

    for iteration in 0..<maxIterations {
      quantized = self.quantizeWithGain(spectral: spectral, globalGain: gain)

      var lastNonZero = 0
      for i in stride(from: quantized.count - 1, through: 0, by: -1) {
        if quantized[i] != 0 {
          lastNonZero = i + 1
          break
        }
      }

      if lastNonZero == 0 && iteration == 0 {
        gain = max(gain - 40, 0)
        continue
      }

      let significantCount = min((lastNonZero + 1) & ~1, 576)
      let bigValues = min(significantCount / 2, 288)
      let valuesToCount = Array(quantized.prefix(bigValues * 2))
      estimatedBits = self.countHuffmanBits(valuesToCount)

      if estimatedBits <= maxBits {
        break
      }

      gain = min(gain + 4, 255)
      if gain >= 255 {
        break
      }
    }

    var lastNonZero = 0
    for i in stride(from: quantized.count - 1, through: 0, by: -1) {
      if quantized[i] != 0 {
        lastNonZero = i + 1
        break
      }
    }

    let significantCount = min((lastNonZero + 1) & ~1, 576)
    let bigValues = min(significantCount / 2, 288)
    let valuesToEncode = Array(quantized.prefix(bigValues * 2))

    let huffman = HuffmanEncoder()
    let actualBits = huffman.encodeWithTable15(valuesToEncode, writer: &writer)

    return (gain, quantized, actualBits)
  }

  /// Quantizes spectral coefficients at a specific global gain using vectorized operations.
  private func quantizeWithGain(spectral: [Float], globalGain: Int) -> [Int] {
    let stepPower = Double(globalGain - 210) / 4.0
    let quantizerStep = Float(max(pow(2.0, stepPower), 0.0001))
    var invStep = 1.0 / quantizerStep
    let maxQuantized: Int32 = 15
    let count = spectral.count

    var absValues = [Float](repeating: 0, count: count)
    vDSP_vabs(spectral, 1, &absValues, 1, vDSP_Length(count))

    var exponent = [Float](repeating: 0.75, count: count)
    var magnitudes = [Float](repeating: 0, count: count)

    var threshold: Float = 1e-10
    vDSP_vthr(absValues, 1, &threshold, &absValues, 1, vDSP_Length(count))

    vvpowf(&magnitudes, &exponent, absValues, [Int32(count)])

    var scaled = [Float](repeating: 0, count: count)
    vDSP_vsmul(magnitudes, 1, &invStep, &scaled, 1, vDSP_Length(count))

    var result = [Int](repeating: 0, count: count)
    for i in 0..<count {
      let quantized = min(Int32(scaled[i].rounded()), maxQuantized)
      result[i] = spectral[i] < 0 ? -Int(quantized) : Int(quantized)
    }

    return result
  }

  /// Estimates the number of Huffman-coded bits for a set of quantized values without writing.
  private func countHuffmanBits(_ values: [Int]) -> Int {
    var bits = 0
    var index = 0

    while index + 1 < values.count {
      let absX = min(abs(values[index]), 15)
      let absY = min(abs(values[index + 1]), 15)

      let code = MP3Tables.table15.table[absX][absY]
      bits += code.length

      if absX != 0 { bits += 1 }
      if absY != 0 { bits += 1 }

      index += 2
    }

    if index < values.count {
      let absX = min(abs(values[index]), 15)
      let code = MP3Tables.table15.table[absX][0]
      bits += code.length
      if absX != 0 { bits += 1 }
    }

    return bits
  }

  /// Calculates region boundary indices for Huffman table region selection.
  private func calculateRegionCounts(bigValues: Int, sampleRate: Int) -> (Int, Int) {
    let bigValuesRegion = bigValues * 2
    let bands = ScaleFactorBands.bandTable(for: sampleRate)

    var boundaries = [Int]()
    var cumulative = 0
    for width in bands {
      cumulative += width
      boundaries.append(cumulative)
    }

    var region0Count = 0
    for i in 0..<min(15, boundaries.count) {
      if boundaries[i] <= bigValuesRegion {
        region0Count = i
      } else {
        break
      }
    }

    var region1Count = 0
    let region1Start = region0Count + 1
    for i in region1Start..<min(region1Start + 7, boundaries.count) {
      if i < boundaries.count && boundaries[i] <= bigValuesRegion {
        region1Count = i - region0Count - 1
      } else {
        break
      }
    }

    return (min(region0Count, 15), min(region1Count, 7))
  }

  /// Separates interleaved multi-channel samples into per-channel arrays using vDSP.
  private func deinterleave(samples: [Float], channels: Int) -> [[Float]] {
    guard channels > 1 else {
      return [samples]
    }

    let samplesPerChannel = samples.count / channels
    var result = [[Float]](repeating: [Float](repeating: 0, count: samplesPerChannel), count: channels)

    samples.withUnsafeBufferPointer { srcPtr in
      for ch in 0..<channels {
        result[ch].withUnsafeMutableBufferPointer { dstPtr in
          vDSP_mmov(
            srcPtr.baseAddress! + ch,
            dstPtr.baseAddress!,
            1,
            vDSP_Length(samplesPerChannel),
            vDSP_Length(channels),
            1
          )
        }
      }
    }

    return result
  }

  /// Processes PCM samples through the polyphase analysis filterbank.
  private mutating func analyzeSubbands(_ samples: [Float], channel: Int) -> [[Float]] {
    var subbands = Array(repeating: [Float](), count: Self.subbands)

    var offset = 0
    for _ in 0..<18 {
      let chunk: [Float]
      if offset + 32 <= samples.count {
        chunk = Array(samples[offset..<(offset + 32)])
      } else {
        var partial = Array(samples[offset...])
        partial.append(contentsOf: repeatElement(0, count: 32 - partial.count))
        chunk = partial
      }

      let subbandSamples = PolyphaseFilterbank.analyze(
        newSamples: chunk,
        buffer: &self.filterbankBuffers[channel]
      )

      for sb in 0..<Self.subbands {
        subbands[sb].append(subbandSamples[sb])
      }

      offset += 32
    }

    return subbands
  }

  /// Encodes one granule through the full spectral processing pipeline.
  private mutating func encodeSpectrum(
    _ samples: [Float],
    channel: Int,
    sampleRate: Int
  ) -> SpectrumResult {
    let subbands = self.analyzeSubbands(samples, channel: channel)
    let transient = TransientDetector.analyze(samples: samples)

    let mdct = MDCT.apply(
      subbandSamples: subbands,
      overlap: &self.mdctOverlap[channel],
      blockType: transient.blockType
    )

    let thresholds = PsychoacousticModel.maskingThresholds(
      spectrum: mdct,
      sampleRate: sampleRate,
      quality: self.options.quality
    )

    // Scalefactors are unity since scalefac_compress=0 (no scalefactors encoded).
    // Global gain alone controls the quantization step size.
    let scalefactors = [Float](repeating: 1.0, count: mdct.count)

    let gain = self.computeGlobalGain(mdct)

    let windowSwitching = transient.blockType == .long ? 0 : 1
    let mixedBlockFlag = transient.blockType == .mixed ? 1 : 0

    return SpectrumResult(
      spectral: mdct,
      scalefactors: scalefactors,
      globalGain: gain,
      windowSwitching: windowSwitching,
      blockType: transient.blockType.rawValue,
      mixedBlockFlag: mixedBlockFlag,
      subblockGain: transient.subblockGain,
      maskingThresholds: thresholds
    )
  }

  /// Computes an initial global gain from the peak spectral magnitude.
  private func computeGlobalGain(_ spectral: [Float]) -> Int {
    let peak = spectral.map { abs($0) }.max() ?? 0

    guard peak > 0 else {
      return 210
    }

    let targetMax: Float = 15.0
    let peakPow = pow(peak, 0.75)
    let ratio = peakPow / targetMax

    if ratio <= 0 {
      return 210
    }

    let gain = 210 + Int(4.0 * log2(Double(ratio)))
    return min(max(gain, 0), 255)
  }

  /// Quantizes spectral coefficients using ISO 11172-3 power-law quantization.
  private func quantizeSpectral(
    _ spectral: [Float],
    scalefactors: [Float],
    thresholds: [Float],
    globalGain: Int
  ) -> [Int] {
    let stepPower = Double(globalGain - 210) / 4.0
    let quantizerStep = max(pow(2.0, stepPower), 0.0001)

    return spectral.enumerated().map { _, value in
      let absValue = abs(value)

      guard absValue > 1e-10 else {
        return 0
      }

      let magnitude = pow(Double(absValue), 0.75)
      let quantized = Int((magnitude / quantizerStep).rounded())
      let clamped = min(quantized, 15)

      return value < 0 ? -clamped : clamped
    }
  }
}

// MARK: - ID3v2.3 Tag Writer

/// Assembles ID3v2.3 binary tag data from an ``ID3Tag``.
enum ID3TagWriter {
  /// Builds a complete ID3v2.3 tag from the given metadata.
  /// Returns empty `Data` if no fields are set.
  static func build(tag: ID3Tag) -> Data {
    var frames = Data()

    if let title = tag.title { frames.append(textFrame(id: "TIT2", value: title)) }
    if let artist = tag.artist { frames.append(textFrame(id: "TPE1", value: artist)) }
    if let album = tag.album { frames.append(textFrame(id: "TALB", value: album)) }
    if let genre = tag.genre { frames.append(textFrame(id: "TCON", value: genre)) }

    if let year = tag.year {
      frames.append(textFrame(id: "TYER", value: String(year)))
    }

    if let track = tag.track {
      let value = tag.trackTotal != nil ? "\(track)/\(tag.trackTotal!)" : "\(track)"
      frames.append(textFrame(id: "TRCK", value: value))
    }

    if let comment = tag.comment {
      frames.append(commentFrame(comment: comment))
    }

    if let art = tag.albumArt {
      frames.append(pictureFrame(art: art, mimeType: tag.albumArtMIMEType))
    }

    if frames.isEmpty { return Data() }

    // ID3v2.3 header: "ID3" + version 2.3 + no flags + synchsafe size
    var header = Data()
    header.append(contentsOf: [0x49, 0x44, 0x33]) // "ID3"
    header.append(contentsOf: [0x03, 0x00])        // Version 2.3
    header.append(0x00)                             // Flags
    header.append(contentsOf: synchsafeSize(UInt32(frames.count)))

    return header + frames
  }

  /// Builds a text frame (TIT2, TPE1, TALB, TCON, TYER, TRCK).
  private static func textFrame(id: String, value: String) -> Data {
    let payload = Data(value.utf8)
    // Encoding byte (0x03 = UTF-8) + string bytes
    let contentSize = UInt32(1 + payload.count)
    var frame = frameHeader(id: id, size: contentSize)
    frame.append(0x03)  // UTF-8 encoding
    frame.append(payload)
    return frame
  }

  /// Builds a COMM (comment) frame.
  private static func commentFrame(comment: String) -> Data {
    let text = Data(comment.utf8)
    // Encoding byte + "eng" + null-terminated description + text
    let contentSize = UInt32(1 + 3 + 1 + text.count)
    var frame = frameHeader(id: "COMM", size: contentSize)
    frame.append(0x03)                                // UTF-8
    frame.append(contentsOf: [0x65, 0x6E, 0x67])     // "eng"
    frame.append(0x00)                                // Empty description
    frame.append(text)
    return frame
  }

  /// Builds an APIC (attached picture) frame.
  private static func pictureFrame(art: Data, mimeType: String) -> Data {
    let mime = Data(mimeType.utf8)
    // Encoding byte + null-terminated MIME + picture type + null-terminated description + image
    let contentSize = UInt32(1 + mime.count + 1 + 1 + 1 + art.count)
    var frame = frameHeader(id: "APIC", size: contentSize)
    frame.append(0x03)       // UTF-8
    frame.append(mime)
    frame.append(0x00)       // Null terminator for MIME
    frame.append(0x03)       // Picture type: front cover
    frame.append(0x00)       // Empty description
    frame.append(art)
    return frame
  }

  /// Encodes an integer as a 4-byte synchsafe integer (7 bits per byte).
  private static func synchsafeSize(_ size: UInt32) -> [UInt8] {
    [
      UInt8((size >> 21) & 0x7F),
      UInt8((size >> 14) & 0x7F),
      UInt8((size >> 7) & 0x7F),
      UInt8(size & 0x7F),
    ]
  }

  /// Builds a 10-byte frame header: 4-char ID + 4-byte big-endian size + 2 flag bytes.
  private static func frameHeader(id: String, size: UInt32) -> Data {
    var header = Data(id.utf8)
    header.append(UInt8((size >> 24) & 0xFF))
    header.append(UInt8((size >> 16) & 0xFF))
    header.append(UInt8((size >> 8) & 0xFF))
    header.append(UInt8(size & 0xFF))
    header.append(contentsOf: [0x00, 0x00])  // No flags
    return header
  }
}

/// Tracks recent frame statistics for variable bitrate encoding decisions.
private struct VBRState {
  private var gainHistory: [Int] = []
  private var energyHistory: [Float] = []

  /// Records the gain and energy of the most recent granule.
  mutating func update(globalGain: Int, energy: Float) {
    gainHistory.append(globalGain)
    if gainHistory.count > 10 {
      gainHistory.removeFirst()
    }
    energyHistory.append(energy)
    if energyHistory.count > 10 {
      energyHistory.removeFirst()
    }
  }
  
  /// Suggests a global gain based on recent history and quality setting.
  func globalGain(quality: Int) -> Int {
    let average = gainHistory.isEmpty ? 180 : gainHistory.reduce(0, +) / gainHistory.count
    return min(max(average + (9 - quality) * 2, 0), 255)
  }
  
  /// Estimates the part2_3_length for a given quality level.
  func estimatePart23Length(quality: Int) -> Int {
    let base = 450
    return max(0, base - quality * 30)
  }
  
  /// Chooses a bitrate for VBR encoding based on frame energy and quality setting.
  ///
  /// Compares the current frame's energy to the running average and adjusts the
  /// bitrate proportionally. Higher quality settings allow wider bitrate swings.
  ///
  /// - Parameters:
  ///   - base: Base bitrate in kbps.
  ///   - energy: Energy of the current frame.
  ///   - quality: Quality level (0 = highest quality, 9 = lowest).
  /// - Returns: Selected bitrate in kbps, clamped to valid bounds.
  func chooseBitrate(base: Int, energy: Float, quality: Int) -> Int {
    let average = energyHistory.isEmpty ? energy : energyHistory.reduce(0, +) / Float(energyHistory.count)
    let ratio = min(max(energy / max(average, 0.0001), 0.5), 2.0)

    let qualityFactor = Float(9 - quality) / 9.0
    let maxAdjustment = Int(32.0 + 32.0 * qualityFactor)
    let adjustment = Int((ratio - 1.0) * Float(maxAdjustment))

    let minBitrate = max(32, base - 64 + quality * 8)
    let maxBitrate = min(320, base + 64 - quality * 4)

    return max(minBitrate, min(base + adjustment, maxBitrate))
  }
}

// MARK: - Polyphase Analysis Filterbank

/// ISO 11172-3 polyphase analysis filterbank for splitting PCM into 32 frequency subbands.
private enum PolyphaseFilterbank {
  /// 32x64 analysis cosine matrix: `M[k][n] = cos((2k+1)(n-16) * PI/64)`.
  private static let analysisMatrix: [[Float]] = {
    var matrix = [[Float]](repeating: [Float](repeating: 0, count: 64), count: 32)
    for k in 0..<32 {
      for n in 0..<64 {
        let angle = Double.pi / 64.0 * Double(2 * k + 1) * (Double(n) - 16.0)
        matrix[k][n] = Float(cos(angle))
      }
    }
    return matrix
  }()

  /// ISO 11172-3 Table C.1 analysis window coefficients (512 values).
  private static let isoAnalysisWindow: [Float] = [
    // Row 0
    0.000000000, -0.000000477, -0.000000477, -0.000000477,
    -0.000000477, -0.000000477, -0.000000477, -0.000000954,
    -0.000000954, -0.000000954, -0.000000954, -0.000001431,
    -0.000001431, -0.000001907, -0.000001907, -0.000002384,
    -0.000002384, -0.000002861, -0.000003338, -0.000003338,
    -0.000003815, -0.000004292, -0.000004768, -0.000005245,
    -0.000006199, -0.000006676, -0.000007629, -0.000008106,
    -0.000009060, -0.000010014, -0.000011444, -0.000012398,
    // Row 1
    -0.000013828, -0.000014782, -0.000016689, -0.000018120,
    -0.000019550, -0.000021458, -0.000023365, -0.000025272,
    -0.000027657, -0.000030041, -0.000032425, -0.000034809,
    -0.000037670, -0.000040531, -0.000043392, -0.000046253,
    -0.000049591, -0.000052929, -0.000055790, -0.000059605,
    -0.000062943, -0.000066280, -0.000070095, -0.000073433,
    -0.000076771, -0.000080585, -0.000083923, -0.000087261,
    -0.000090599, -0.000093460, -0.000096321, -0.000099182,
    // Row 2
    0.000101566, 0.000103951, 0.000105858, 0.000107288,
    0.000108242, 0.000108719, 0.000108719, 0.000108242,
    0.000106812, 0.000105381, 0.000102520, 0.000099182,
    0.000095367, 0.000090122, 0.000084400, 0.000077724,
    0.000069618, 0.000060558, 0.000050545, 0.000039577,
    0.000027180, 0.000013828, -0.000000954, -0.000017166,
    -0.000034332, -0.000052929, -0.000072956, -0.000093937,
    -0.000116348, -0.000140190, -0.000165462, -0.000191212,
    // Row 3
    -0.000218868, -0.000247478, -0.000277042, -0.000307560,
    -0.000339031, -0.000371456, -0.000404358, -0.000438213,
    -0.000472546, -0.000507355, -0.000542164, -0.000576973,
    -0.000611782, -0.000646591, -0.000680923, -0.000714302,
    -0.000747204, -0.000779152, -0.000809669, -0.000838757,
    -0.000866413, -0.000891685, -0.000915051, -0.000935555,
    -0.000954151, -0.000968933, -0.000980854, -0.000989437,
    -0.000994205, -0.000995159, -0.000991821, -0.000983715,
    // Row 4
    0.000971317, 0.000953674, 0.000930786, 0.000902653,
    0.000868797, 0.000829220, 0.000783920, 0.000731945,
    0.000674248, 0.000610352, 0.000539303, 0.000462532,
    0.000378609, 0.000288486, 0.000191689, 0.000088215,
    -0.000021458, -0.000137329, -0.000259876, -0.000388145,
    -0.000522137, -0.000661850, -0.000806808, -0.000956535,
    -0.001111031, -0.001269817, -0.001432419, -0.001597881,
    -0.001766682, -0.001937389, -0.002110004, -0.002283096,
    // Row 5
    -0.002457142, -0.002630711, -0.002803326, -0.002974033,
    -0.003141880, -0.003306866, -0.003467083, -0.003622532,
    -0.003771782, -0.003914356, -0.004048824, -0.004174709,
    -0.004290581, -0.004395962, -0.004489899, -0.004570484,
    -0.004638195, -0.004691124, -0.004728317, -0.004748821,
    -0.004752159, -0.004737377, -0.004703045, -0.004649162,
    -0.004573822, -0.004477024, -0.004357815, -0.004215240,
    -0.004049301, -0.003858566, -0.003643036, -0.003401756,
    // Row 6
    0.003134727, 0.002841473, 0.002521515, 0.002174854,
    0.001800537, 0.001399517, 0.000971317, 0.000515938,
    0.000033379, -0.000475883, -0.001011848, -0.001573563,
    -0.002161503, -0.002774239, -0.003411293, -0.004072189,
    -0.004756451, -0.005462170, -0.006189346, -0.006937027,
    -0.007703304, -0.008487225, -0.009287834, -0.010103703,
    -0.010933399, -0.011775017, -0.012627602, -0.013489246,
    -0.014358521, -0.015233517, -0.016112804, -0.016994476,
    // Row 7
    -0.017876148, -0.018756866, -0.019634247, -0.020506859,
    -0.021372318, -0.022228718, -0.023074150, -0.023907185,
    -0.024725437, -0.025527000, -0.026310921, -0.027073860,
    -0.027815342, -0.028532982, -0.029224873, -0.029890060,
    -0.030526638, -0.031132698, -0.031706810, -0.032248020,
    -0.032754898, -0.033225536, -0.033659935, -0.034055710,
    -0.034412861, -0.034730434, -0.035007000, -0.035242081,
    -0.035435200, -0.035586357, -0.035694122, -0.035758972,
    // Row 8 (center - symmetric point)
    0.035780907, 0.035758972, 0.035694122, 0.035586357,
    0.035435200, 0.035242081, 0.035007000, 0.034730434,
    0.034412861, 0.034055710, 0.033659935, 0.033225536,
    0.032754898, 0.032248020, 0.031706810, 0.031132698,
    0.030526638, 0.029890060, 0.029224873, 0.028532982,
    0.027815342, 0.027073860, 0.026310921, 0.025527000,
    0.024725437, 0.023907185, 0.023074150, 0.022228718,
    0.021372318, 0.020506859, 0.019634247, 0.018756866,
    // Row 9
    0.017876148, 0.016994476, 0.016112804, 0.015233517,
    0.014358521, 0.013489246, 0.012627602, 0.011775017,
    0.010933399, 0.010103703, 0.009287834, 0.008487225,
    0.007703304, 0.006937027, 0.006189346, 0.005462170,
    0.004756451, 0.004072189, 0.003411293, 0.002774239,
    0.002161503, 0.001573563, 0.001011848, 0.000475883,
    -0.000033379, -0.000515938, -0.000971317, -0.001399517,
    -0.001800537, -0.002174854, -0.002521515, -0.002841473,
    // Row 10
    0.003134727, 0.003401756, 0.003643036, 0.003858566,
    0.004049301, 0.004215240, 0.004357815, 0.004477024,
    0.004573822, 0.004649162, 0.004703045, 0.004737377,
    0.004752159, 0.004748821, 0.004728317, 0.004691124,
    0.004638195, 0.004570484, 0.004489899, 0.004395962,
    0.004290581, 0.004174709, 0.004048824, 0.003914356,
    0.003771782, 0.003622532, 0.003467083, 0.003306866,
    0.003141880, 0.002974033, 0.002803326, 0.002630711,
    // Row 11
    0.002457142, 0.002283096, 0.002110004, 0.001937389,
    0.001766682, 0.001597881, 0.001432419, 0.001269817,
    0.001111031, 0.000956535, 0.000806808, 0.000661850,
    0.000522137, 0.000388145, 0.000259876, 0.000137329,
    0.000021458, -0.000088215, -0.000191689, -0.000288486,
    -0.000378609, -0.000462532, -0.000539303, -0.000610352,
    -0.000674248, -0.000731945, -0.000783920, -0.000829220,
    -0.000868797, -0.000902653, -0.000930786, -0.000953674,
    // Row 12
    0.000971317, 0.000983715, 0.000991821, 0.000995159,
    0.000994205, 0.000989437, 0.000980854, 0.000968933,
    0.000954151, 0.000935555, 0.000915051, 0.000891685,
    0.000866413, 0.000838757, 0.000809669, 0.000779152,
    0.000747204, 0.000714302, 0.000680923, 0.000646591,
    0.000611782, 0.000576973, 0.000542164, 0.000507355,
    0.000472546, 0.000438213, 0.000404358, 0.000371456,
    0.000339031, 0.000307560, 0.000277042, 0.000247478,
    // Row 13
    0.000218868, 0.000191212, 0.000165462, 0.000140190,
    0.000116348, 0.000093937, 0.000072956, 0.000052929,
    0.000034332, 0.000017166, 0.000000954, -0.000013828,
    -0.000027180, -0.000039577, -0.000050545, -0.000060558,
    -0.000069618, -0.000077724, -0.000084400, -0.000090122,
    -0.000095367, -0.000099182, -0.000102520, -0.000105381,
    -0.000106812, -0.000108242, -0.000108719, -0.000108719,
    -0.000108242, -0.000107288, -0.000105858, -0.000103951,
    // Row 14
    0.000101566, 0.000099182, 0.000096321, 0.000093460,
    0.000090599, 0.000087261, 0.000083923, 0.000080585,
    0.000076771, 0.000073433, 0.000070095, 0.000066280,
    0.000062943, 0.000059605, 0.000055790, 0.000052929,
    0.000049591, 0.000046253, 0.000043392, 0.000040531,
    0.000037670, 0.000034809, 0.000032425, 0.000030041,
    0.000027657, 0.000025272, 0.000023365, 0.000021458,
    0.000019550, 0.000018120, 0.000016689, 0.000014782,
    // Row 15
    0.000013828, 0.000012398, 0.000011444, 0.000010014,
    0.000009060, 0.000008106, 0.000007629, 0.000006676,
    0.000006199, 0.000005245, 0.000004768, 0.000004292,
    0.000003815, 0.000003338, 0.000003338, 0.000002861,
    0.000002384, 0.000002384, 0.000001907, 0.000001907,
    0.000001431, 0.000001431, 0.000000954, 0.000000954,
    0.000000954, 0.000000954, 0.000000477, 0.000000477,
    0.000000477, 0.000000477, 0.000000477, 0.000000477
  ]


  /// Converts 32 new PCM samples into 32 subband samples.
  ///
  /// Implements the three-step ISO 11172-3 analysis filterbank: windowing the
  /// 512-sample buffer, partial summation at stride 64, and matrixing with
  /// the cosine analysis matrix. Uses vDSP for all vectorized operations.
  ///
  /// - Parameters:
  ///   - newSamples: 32 new PCM samples to push into the filterbank.
  ///   - buffer: 512-sample sliding buffer, updated in place.
  /// - Returns: 32 subband samples.
  static func analyze(newSamples: [Float], buffer: inout [Float]) -> [Float] {
    if buffer.count < 512 {
      buffer = [Float](repeating: 0, count: 512 - buffer.count) + buffer
    }

    _ = buffer.withUnsafeMutableBufferPointer { ptr in
      memmove(ptr.baseAddress!, ptr.baseAddress! + 32, 480 * MemoryLayout<Float>.size)
    }
    let copyCount = min(32, newSamples.count)
    for i in 0..<copyCount {
      buffer[480 + i] = newSamples[i]
    }
    for i in copyCount..<32 {
      buffer[480 + i] = 0
    }

    let window = isoAnalysisWindow

    // Step 1: Reverse and window the 512-sample buffer
    var reversed = buffer
    vDSP_vrvrs(&reversed, 1, 512)
    var windowed = [Float](repeating: 0, count: 512)
    vDSP_vmul(reversed, 1, window, 1, &windowed, 1, 512)

    // Step 2: Partial sum — Y[j] = sum(Z[i*64 + j]) for i=0..7
    var partial = [Float](repeating: 0, count: 64)
    windowed.withUnsafeBufferPointer { ptr in
      for j in 0..<64 {
        var sum: Float = 0
        vDSP_sve(ptr.baseAddress! + j, 64, &sum, 8)
        partial[j] = sum
      }
    }

    // Step 3: Matrix multiply with cosine analysis matrix
    let matrix = analysisMatrix
    var subbands = [Float](repeating: 0, count: 32)
    for k in 0..<32 {
      var result: Float = 0
      vDSP_dotpr(partial, 1, matrix[k], 1, &result, 64)
      subbands[k] = result
    }

    return subbands
  }
}

// MARK: - MDCT

/// Modified Discrete Cosine Transform for MPEG-1 Layer III.
///
/// Transforms subband samples from the polyphase filterbank into
/// frequency-domain coefficients with overlap-add windowing.
private enum MDCT {
  /// 18x36 long-block MDCT cosine matrix.
  private static let longMDCTMatrix: [[Float]] = {
    let n = 36
    let halfN = 18
    var matrix = [[Float]](repeating: [Float](repeating: 0, count: n), count: halfN)
    for m in 0..<halfN {
      for k in 0..<n {
        let angle = Double.pi / Double(2 * n) * Double(2 * k + 1 + n / 2) * Double(2 * m + 1)
        matrix[m][k] = Float(cos(angle))
      }
    }
    return matrix
  }()

  /// 6x12 short-block MDCT cosine matrix.
  private static let shortMDCTMatrix: [[Float]] = {
    let n = 12
    let halfN = 6
    var matrix = [[Float]](repeating: [Float](repeating: 0, count: n), count: halfN)
    for m in 0..<halfN {
      for k in 0..<n {
        let angle = Double.pi / Double(2 * n) * Double(2 * k + 1 + n / 2) * Double(2 * m + 1)
        matrix[m][k] = Float(cos(angle))
      }
    }
    return matrix
  }()

  /// 36-sample sine window for long blocks.
  private static let longWindow: [Float] = {
    let n = 36
    var window = [Float](repeating: 0, count: n)
    for i in 0..<n {
      window[i] = Float(sin(Double.pi / Double(n) * (Double(i) + 0.5)))
    }
    return window
  }()

  /// 12-sample sine window for short blocks.
  private static let shortWindow: [Float] = {
    let n = 12
    var window = [Float](repeating: 0, count: n)
    for i in 0..<n {
      window[i] = Float(sin(Double.pi / Double(n) * (Double(i) + 0.5)))
    }
    return window
  }()

  /// 36-sample start window for long-to-short block transitions.
  private static let startWindow: [Float] = {
    var window = [Float](repeating: 0, count: 36)
    for i in 0..<18 {
      window[i] = Float(sin(Double.pi / 36.0 * (Double(i) + 0.5)))
    }
    for i in 18..<24 {
      window[i] = 1.0
    }
    for i in 24..<30 {
      window[i] = Float(sin(Double.pi / 12.0 * (Double(i - 18) + 0.5)))
    }
    for i in 30..<36 {
      window[i] = 0.0
    }
    return window
  }()

  /// 36-sample stop window for short-to-long block transitions.
  private static let stopWindow: [Float] = {
    var window = [Float](repeating: 0, count: 36)
    for i in 0..<6 {
      window[i] = 0.0
    }
    for i in 6..<12 {
      window[i] = Float(sin(Double.pi / 12.0 * (Double(i - 6) + 0.5)))
    }
    for i in 12..<18 {
      window[i] = 1.0
    }
    for i in 18..<36 {
      window[i] = Float(sin(Double.pi / 36.0 * (Double(i) + 0.5)))
    }
    return window
  }()

  /// Applies MDCT to all 32 subbands for one granule with overlap-add.
  ///
  /// - Parameters:
  ///   - subbandSamples: 32 subbands, each with 18 time-domain samples.
  ///   - overlap: Previous 18 samples per subband, updated in place for the next granule.
  ///   - blockType: Window shape selection (long, short, or mixed).
  /// - Returns: 576 frequency-domain MDCT coefficients.
  static func apply(subbandSamples: [[Float]], overlap: inout [[Float]], blockType: BlockType) -> [Float] {
    var output = [Float](repeating: 0, count: 576)

    for sb in 0..<32 {
      var currentSamples = sb < subbandSamples.count ? subbandSamples[sb] : [Float](repeating: 0, count: 18)
      let previousSamples = sb < overlap.count ? overlap[sb] : [Float](repeating: 0, count: 18)

      // Compensate for frequency inversion in odd-numbered subbands
      if (sb & 1) != 0 {
        for k in stride(from: 1, to: min(currentSamples.count, 18), by: 2) {
          currentSamples[k] *= -1
        }
      }

      var combined = [Float](repeating: 0, count: 36)
      for i in 0..<18 {
        combined[i] = i < previousSamples.count ? previousSamples[i] : 0
      }
      for i in 0..<18 {
        combined[18 + i] = i < currentSamples.count ? currentSamples[i] : 0
      }

      if sb < overlap.count {
        overlap[sb] = Array(currentSamples.prefix(18))
        while overlap[sb].count < 18 {
          overlap[sb].append(0)
        }
      }

      let mdctCoeffs: [Float]
      switch blockType {
      case .long:
        mdctCoeffs = mdctLong(samples: combined, window: longWindow)
      case .short:
        mdctCoeffs = mdctShort(samples: combined)
      case .mixed:
        if sb < 2 {
          mdctCoeffs = mdctLong(samples: combined, window: longWindow)
        } else {
          mdctCoeffs = mdctShort(samples: combined)
        }
      }

      for i in 0..<18 {
        output[sb * 18 + i] = i < mdctCoeffs.count ? mdctCoeffs[i] : 0
      }
    }

    if blockType == .long {
      applyAliasingReduction(&output)
    }

    return output
  }

  /// Aliasing reduction coefficients from ISO 11172-3 Table B.9 (cs[i]^2 + ca[i]^2 = 1).
  private static let aliasingCS = SIMD8<Float>(
    0.857492926, 0.881741997, 0.949628649, 0.983314592,
    0.995517816, 0.999160558, 0.999899195, 0.999993155
  )
  private static let aliasingCA = SIMD8<Float>(
    -0.514495755, -0.471731969, -0.313377454, -0.181913200,
    -0.094574193, -0.040965583, -0.014198569, -0.003699975
  )

  /// Applies the ISO 11172-3 aliasing reduction butterfly between adjacent subbands using SIMD.
  ///
  /// Only applied to long blocks. Performs 8-wide butterfly operations across
  /// the 31 subband boundaries to cancel aliasing from the polyphase filterbank.
  private static func applyAliasingReduction(_ spectrum: inout [Float]) {
    for sb in 0..<31 {
      let sbEnd = sb * 18 + 17
      let sbNextStart = (sb + 1) * 18

      let upper = SIMD8<Float>(
        spectrum[sbEnd], spectrum[sbEnd - 1], spectrum[sbEnd - 2], spectrum[sbEnd - 3],
        spectrum[sbEnd - 4], spectrum[sbEnd - 5], spectrum[sbEnd - 6], spectrum[sbEnd - 7]
      )
      let lower = SIMD8<Float>(
        spectrum[sbNextStart], spectrum[sbNextStart + 1], spectrum[sbNextStart + 2], spectrum[sbNextStart + 3],
        spectrum[sbNextStart + 4], spectrum[sbNextStart + 5], spectrum[sbNextStart + 6], spectrum[sbNextStart + 7]
      )

      let newUpper = lower * aliasingCA + upper * aliasingCS
      let newLower = lower * aliasingCS - upper * aliasingCA

      spectrum[sbEnd] = newUpper[0]
      spectrum[sbEnd - 1] = newUpper[1]
      spectrum[sbEnd - 2] = newUpper[2]
      spectrum[sbEnd - 3] = newUpper[3]
      spectrum[sbEnd - 4] = newUpper[4]
      spectrum[sbEnd - 5] = newUpper[5]
      spectrum[sbEnd - 6] = newUpper[6]
      spectrum[sbEnd - 7] = newUpper[7]

      spectrum[sbNextStart] = newLower[0]
      spectrum[sbNextStart + 1] = newLower[1]
      spectrum[sbNextStart + 2] = newLower[2]
      spectrum[sbNextStart + 3] = newLower[3]
      spectrum[sbNextStart + 4] = newLower[4]
      spectrum[sbNextStart + 5] = newLower[5]
      spectrum[sbNextStart + 6] = newLower[6]
      spectrum[sbNextStart + 7] = newLower[7]
    }
  }

  /// Computes the MDCT for a long block (36 input samples to 18 coefficients).
  private static func mdctLong(samples: [Float], window: [Float]) -> [Float] {
    let n = 36
    let halfN = 18
    let normalization: Float = 9.0

    var windowed = [Float](repeating: 0, count: n)
    vDSP_vmul(samples, 1, window, 1, &windowed, 1, vDSP_Length(n))

    let matrix = longMDCTMatrix
    var output = [Float](repeating: 0, count: halfN)
    for m in 0..<halfN {
      var result: Float = 0
      vDSP_dotpr(windowed, 1, matrix[m], 1, &result, vDSP_Length(n))
      output[m] = result / normalization
    }

    return output
  }

  /// Computes the MDCT for a short block (3 windows of 12 samples each, 18 total coefficients).
  private static func mdctShort(samples: [Float]) -> [Float] {
    var output = [Float](repeating: 0, count: 18)
    let n = 12
    let normalization: Float = 3.0

    let matrix = shortMDCTMatrix

    for w in 0..<3 {
      let offset = w * 6 + 6
      var windowSamples = [Float](repeating: 0, count: n)
      for i in 0..<n {
        let idx = offset + i
        windowSamples[i] = idx < samples.count ? samples[idx] * shortWindow[i] : 0
      }

      for m in 0..<6 {
        var result: Float = 0
        vDSP_dotpr(windowSamples, 1, matrix[m], 1, &result, vDSP_Length(n))
        output[w + m * 3] = result / normalization
      }
    }

    return output
  }
}

/// Huffman encoder for MPEG-1 Layer III spectral data.
///
/// Encodes quantized spectral value pairs using the ISO 11172-3 Huffman code tables.
private struct HuffmanEncoder {
  /// Encodes value pairs using Huffman table 1 (values 0-1).
  func encodeWithTable1(_ values: [Int], writer: inout BitstreamWriter) -> Int {
    let startBits = writer.bitCount
    var index = 0

    while index + 1 < values.count {
      let x = values[index]
      let y = values[index + 1]
      self.writePairTable1(x: x, y: y, writer: &writer)
      index += 2
    }

    if index < values.count {
      self.writePairTable1(x: values[index], y: 0, writer: &writer)
    }

    return writer.bitCount - startBits
  }

  /// Writes a value pair using Huffman table 1.
  private func writePairTable1(x: Int, y: Int, writer: inout BitstreamWriter) {
    let absX = min(abs(x), 1)
    let absY = min(abs(y), 1)

    let code = MP3Tables.table1.table[absX][absY]
    writer.write(bits: code.bits, count: code.length)

    if absX != 0 {
      writer.write(bits: x < 0 ? 1 : 0, count: 1)
    }
    if absY != 0 {
      writer.write(bits: y < 0 ? 1 : 0, count: 1)
    }
  }

  /// Encodes value pairs using Huffman table 15 (values 0-15, no linbits).
  func encodeWithTable15(_ values: [Int], writer: inout BitstreamWriter) -> Int {
    let startBits = writer.bitCount
    var index = 0

    while index + 1 < values.count {
      let x = values[index]
      let y = values[index + 1]
      self.writePairTable15(x: x, y: y, writer: &writer)
      index += 2
    }

    if index < values.count {
      self.writePairTable15(x: values[index], y: 0, writer: &writer)
    }

    return writer.bitCount - startBits
  }

  /// Writes a value pair using Huffman table 15.
  private func writePairTable15(x: Int, y: Int, writer: inout BitstreamWriter) {
    let absX = min(abs(x), 15)
    let absY = min(abs(y), 15)

    let code = MP3Tables.table15.table[absX][absY]
    writer.write(bits: code.bits, count: code.length)

    if absX != 0 {
      writer.write(bits: x < 0 ? 1 : 0, count: 1)
    }
    if absY != 0 {
      writer.write(bits: y < 0 ? 1 : 0, count: 1)
    }
  }

  /// Encodes value pairs with automatic Huffman table selection.
  func encode(_ values: [Int], writer: inout BitstreamWriter) -> Int {
    let startBits = writer.bitCount
    var index = 0

    while index + 1 < values.count {
      let x = values[index]
      let y = values[index + 1]
      self.writePair(x: x, y: y, writer: &writer)
      index += 2
    }

    if index < values.count {
      self.writePair(x: values[index], y: 0, writer: &writer)
    }

    return writer.bitCount - startBits
  }

  /// Writes a value pair using the smallest suitable Huffman table.
  private func writePair(x: Int, y: Int, writer: inout BitstreamWriter) {
    let absX = min(abs(x), MP3Tables.table15.maxValue)
    let absY = min(abs(y), MP3Tables.table15.maxValue)
    let table = self.selectTable(absX: absX, absY: absY)

    guard absX < table.table.count && absY < table.table[absX].count else {
      let code = MP3Tables.table15.table[min(absX, 15)][min(absY, 15)]
      writer.write(bits: code.bits, count: code.length)
      if absX != 0 {
        writer.write(bits: x < 0 ? 1 : 0, count: 1)
      }
      if absY != 0 {
        writer.write(bits: y < 0 ? 1 : 0, count: 1)
      }
      return
    }

    let code = table.table[absX][absY]
    writer.write(bits: code.bits, count: code.length)

    if absX != 0 {
      writer.write(bits: x < 0 ? 1 : 0, count: 1)
    }
    if absY != 0 {
      writer.write(bits: y < 0 ? 1 : 0, count: 1)
    }
  }

  /// Selects the smallest Huffman table that can encode both values.
  private func selectTable(absX: Int, absY: Int) -> MP3Tables.HuffmanTable {
    let maxValue = max(absX, absY)
    if maxValue <= MP3Tables.table1.maxValue {
      return MP3Tables.table1
    }
    if maxValue <= MP3Tables.table2.maxValue {
      return MP3Tables.table2
    }
    if maxValue <= MP3Tables.table5.maxValue {
      return MP3Tables.table5
    }
    if maxValue <= MP3Tables.table7.maxValue {
      return MP3Tables.table7
    }
    if maxValue <= MP3Tables.table10.maxValue {
      return MP3Tables.table10
    }
    return MP3Tables.table15
  }
}

/// Scale factor band definitions from ISO 11172-3 Table B.8.
///
/// These define frequency groupings that approximate human auditory critical bands.
private enum ScaleFactorBands {
  /// Long block band widths for 44100 Hz (21 bands).
  static let longBands44100 = [4, 4, 4, 4, 4, 4, 6, 6, 8, 8, 10, 12, 16, 20, 24, 28, 34, 42, 50, 54, 76]

  /// Long block band widths for 48000 Hz (21 bands).
  static let longBands48000 = [4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 10, 12, 16, 18, 22, 28, 34, 40, 46, 54, 54]

  /// Long block band widths for 32000 Hz (21 bands).
  static let longBands32000 = [4, 4, 4, 4, 4, 4, 6, 6, 8, 10, 12, 16, 20, 24, 30, 38, 46, 56, 68, 84, 102]

  /// Short block band widths for 44100 Hz (12 bands, applied 3 times).
  static let shortBands44100 = [4, 4, 4, 4, 6, 8, 10, 12, 14, 18, 22, 30]

  /// Normalizes spectral coefficients per band and returns per-coefficient scale factors.
  ///
  /// - Parameters:
  ///   - spectrum: Raw spectral coefficients.
  ///   - sampleRate: Sample rate for band table lookup.
  /// - Returns: A tuple of (normalized spectrum, per-coefficient scale factors).
  static func scale(spectrum: [Float], sampleRate: Int) -> ([Float], [Float]) {
    let bands = bandTable(for: sampleRate)
    var scalefactors = [Float]()
    var scaledSpectrum = spectrum
    var cursor = 0

    for (_, bandWidth) in bands.enumerated() {
      let start = cursor
      let end = min(cursor + bandWidth, spectrum.count)

      if start >= spectrum.count {
        break
      }

      let slice = spectrum[start..<end]
      let peak = slice.map { abs($0) }.max() ?? 0.0
      let scale = max(peak, 0.0001)
      scalefactors.append(scale)

      for index in start..<end {
        scaledSpectrum[index] = spectrum[index] / scale
      }

      cursor = end
    }

    var expandedScalefactors = [Float](repeating: 0.0001, count: spectrum.count)
    cursor = 0
    for (idx, bandWidth) in bands.enumerated() {
      let start = cursor
      let end = min(cursor + bandWidth, spectrum.count)

      if idx < scalefactors.count {
        for i in start..<end {
          expandedScalefactors[i] = scalefactors[idx]
        }
      }

      cursor = end
      if cursor >= spectrum.count {
        break
      }
    }

    return (scaledSpectrum, expandedScalefactors)
  }

  /// Returns the long-block band width table for the given sample rate.
  static func bandTable(for sampleRate: Int) -> [Int] {
    switch sampleRate {
    case 48_000:
      return longBands48000
    case 32_000:
      return longBands32000
    default:
      return longBands44100
    }
  }

  /// Returns the number of scale factor bands for a given sample rate and block type.
  static func bandCount(for sampleRate: Int, isShort: Bool) -> Int {
    if isShort {
      return 12
    }
    return bandTable(for: sampleRate).count
  }
}

/// Utility for frame-level signal analysis.
private enum FrameAnalysis {
  /// Calculates the mean square energy of samples using vDSP.
  static func energy(samples: [Float]) -> Float {
    guard !samples.isEmpty else { return 0 }
    var sumOfSquares: Float = 0
    vDSP_svesq(samples, 1, &sumOfSquares, vDSP_Length(samples.count))
    return sumOfSquares / Float(samples.count)
  }
}

/// Result of spectral encoding for one granule.
private struct SpectrumResult {
  let spectral: [Float]
  let scalefactors: [Float]
  let globalGain: Int
  let windowSwitching: Int
  let blockType: Int
  let mixedBlockFlag: Int
  let subblockGain: [Int]
  let maskingThresholds: [Float]
}

/// MDCT block type selection.
private enum BlockType: Int {
  case long = 0
  case short = 2
  case mixed = 1
}

/// Result of transient detection for block type selection.
private struct TransientResult {
  let blockType: BlockType
  let subblockGain: [Int]
}

/// Detects transients to decide between long and short MDCT block types.
private enum TransientDetector {
  /// Analyzes energy distribution across sub-blocks to detect transients.
  ///
  /// If the energy ratio between sub-blocks exceeds a threshold, a short or
  /// mixed block type is selected to improve temporal resolution.
  ///
  /// - Parameter samples: PCM samples for one granule.
  /// - Returns: Block type and per-subblock gain values.
  static func analyze(samples: [Float]) -> TransientResult {
    let subblockCount = 3
    let subblockSize = max(samples.count / subblockCount, 1)
    var energies: [Float] = []
    for index in 0..<subblockCount {
      let start = index * subblockSize
      let end = min(start + subblockSize, samples.count)
      let slice = Array(samples[start..<end])
      energies.append(FrameAnalysis.energy(samples: slice))
    }
    let maxEnergy = energies.max() ?? 0
    let minEnergy = energies.min() ?? 0.0001
    let ratio = maxEnergy / max(minEnergy, 0.0001)
    let blockType: BlockType
    if ratio > 6.0 {
      blockType = energies.firstIndex(of: maxEnergy) == 0 ? .mixed : .short
    } else {
      blockType = .long
    }
    let subblockGain = energies.map { energy in
      let normalized = min(max(energy / max(maxEnergy, 0.0001), 0.0), 1.0)
      return Int((1.0 - normalized) * 7.0)
    }
    return TransientResult(blockType: blockType, subblockGain: subblockGain)
  }
}

/// Simplified psychoacoustic model for computing masking thresholds.
private enum PsychoacousticModel {
  /// Computes per-coefficient masking thresholds based on band energy and quality.
  ///
  /// Higher quality settings produce lower thresholds, allowing more spectral detail
  /// to be preserved during quantization.
  ///
  /// - Parameters:
  ///   - spectrum: MDCT spectral coefficients.
  ///   - sampleRate: Sample rate for band table lookup.
  ///   - quality: Quality level (0 = highest, 9 = lowest).
  /// - Returns: Per-coefficient masking thresholds.
  static func maskingThresholds(spectrum: [Float], sampleRate: Int, quality: Int) -> [Float] {
    let bands = ScaleFactorBands.bandTable(for: sampleRate)
    let qualityScale = Float(max(0.1, Double(10 - quality) / 10.0))
    var thresholds = Array(repeating: Float(0.0001), count: spectrum.count)
    var cursor = 0
    for band in bands {
      let start = cursor
      let end = min(cursor + band, spectrum.count)
      let bandSize = end - start
      guard bandSize > 0 else {
        cursor = end
        continue
      }

      var energy: Float = 0
      spectrum.withUnsafeBufferPointer { ptr in
        vDSP_svesq(ptr.baseAddress! + start, 1, &energy, vDSP_Length(bandSize))
      }

      let average = energy / Float(bandSize)
      let threshold = max(average * qualityScale, 0.0001)
      for index in start..<end {
        thresholds[index] = threshold
      }
      cursor = end
      if cursor >= spectrum.count {
        break
      }
    }
    return thresholds
  }
}

/// Computes the `scalefac_compress` field from scale factor statistics.
private enum ScaleFactorCompression {
  /// Returns a scalefac_compress index (0-15) based on scale factor variance.
  static func compress(scalefactors: [Float]) -> Int {
    guard !scalefactors.isEmpty else { return 0 }
    let count = vDSP_Length(scalefactors.count)

    var mean: Float = 0
    vDSP_meanv(scalefactors, 1, &mean, count)

    var centered = [Float](repeating: 0, count: scalefactors.count)
    var negMean = -mean
    vDSP_vsadd(scalefactors, 1, &negMean, &centered, 1, count)

    var variance: Float = 0
    vDSP_svesq(centered, 1, &variance, count)
    variance /= Float(scalefactors.count)

    let normalized = min(max(variance / max(mean * mean, 0.0001), 0.0), 1.0)
    return min(Int(normalized * 15.0), 15)
  }
}

/// Determines whether the pre-emphasis flag should be set for a granule.
private enum PreEmphasis {
  /// Returns `true` if high-frequency energy significantly exceeds low-frequency energy.
  static func shouldEnable(spectral: [Float], scalefactors: [Float]) -> Bool {
    guard !spectral.isEmpty else { return false }
    let highStart = max(spectral.count * 3 / 4, 0)
    let highCount = spectral.count - highStart

    var highEnergy: Float = 0
    if highCount > 0 {
      spectral.withUnsafeBufferPointer { ptr in
        vDSP_svesq(ptr.baseAddress! + highStart, 1, &highEnergy, vDSP_Length(highCount))
      }
    }

    var lowEnergy: Float = 0
    if highStart > 0 {
      vDSP_svesq(spectral, 1, &lowEnergy, vDSP_Length(highStart))
    }

    var scalefactorSum: Float = 0
    if !scalefactors.isEmpty {
      vDSP_sve(scalefactors, 1, &scalefactorSum, vDSP_Length(scalefactors.count))
    }
    let scalefactorAverage = scalefactorSum / Float(max(scalefactors.count, 1))

    return highEnergy > lowEnergy * 1.5 && scalefactorAverage > 0.5
  }
}

/// Side information fields for one granule of one channel (ISO 11172-3 Section 2.4.1.7).
private struct GranuleInfo {
  var part23Length: Int = 0
  var bigValues: Int = 0
  var globalGain: Int = 0
  var scalefacCompress: Int = 0
  var windowSwitching: Int = 0
  var blockType: Int = 0
  var mixedBlockFlag: Int = 0
  var tableSelect: [Int] = [0, 0, 0]
  var subblockGain: [Int] = [0, 0, 0]
  var region0Count: Int = 0
  var region1Count: Int = 0
  var preflag: Int = 0
  var scalefacScale: Int = 0
  var count1TableSelect: Int = 0
}

/// Simplified bit reservoir that keeps each frame self-contained (main_data_begin = 0).
private struct BitReservoir {
  private(set) var mainDataBegin: Int = 0

  /// Returns the main data unchanged. No cross-frame buffering is used.
  mutating func consume(with mainData: Data) -> Data {
    return mainData
  }
}

/// Decides whether to use raw L/R stereo or mid/side encoding for joint stereo mode.
private enum StereoDecision {
  case raw([Float], [Float])
  case midSide([Float], [Float])

  /// Evaluates whether mid/side encoding is beneficial for the given stereo pair.
  ///
  /// Mid/side encoding is chosen when the side (difference) channel has significantly
  /// less energy than the mid (sum) channel, indicating high stereo correlation.
  static func make(mode: MP3EncoderOptions.Mode, left: [Float], right: [Float]) -> StereoDecision {
    guard mode == .jointStereo, left.count == right.count else {
      return .raw(left, right)
    }

    let count = left.count

    var mid = [Float](repeating: 0, count: count)
    vDSP_vadd(left, 1, right, 1, &mid, 1, vDSP_Length(count))
    var halfScalar: Float = 0.5
    vDSP_vsmul(mid, 1, &halfScalar, &mid, 1, vDSP_Length(count))

    var side = [Float](repeating: 0, count: count)
    vDSP_vsub(right, 1, left, 1, &side, 1, vDSP_Length(count))
    vDSP_vsmul(side, 1, &halfScalar, &side, 1, vDSP_Length(count))

    let midEnergy = FrameAnalysis.energy(samples: mid)
    let sideEnergy = FrameAnalysis.energy(samples: side)
    if sideEnergy < midEnergy * 0.4 {
      return .midSide(mid, side)
    }
    return .raw(left, right)
  }

  /// Returns samples for the given channel index (0 = left/mid, 1 = right/side).
  func samples(for channel: Int) -> [Float] {
    switch self {
    case .raw(let left, let right):
      return channel == 0 ? left : right
    case .midSide(let mid, let side):
      return channel == 0 ? mid : side
    }
  }
}

// MARK: - UInt32 Extension

private extension UInt32 {
  /// The value serialized as 4 bytes in big-endian (network) byte order.
  var bigEndianBytes: [UInt8] {
    return [
      UInt8((self >> 24) & 0xFF),
      UInt8((self >> 16) & 0xFF),
      UInt8((self >> 8) & 0xFF),
      UInt8(self & 0xFF)
    ]
  }
}

/// CRC-16 implementation using the MPEG polynomial (0x8005).
private enum CRC16 {
  private static let table: [UInt16] = {
    var table = [UInt16](repeating: 0, count: 256)
    for i in 0..<256 {
      var crc = UInt16(i) << 8
      for _ in 0..<8 {
        if (crc & 0x8000) != 0 {
          crc = (crc << 1) ^ 0x8005
        } else {
          crc <<= 1
        }
      }
      table[i] = crc
    }
    return table
  }()

  /// Computes the MPEG CRC-16 checksum over the given data.
  static func mpeg(data: Data) -> UInt16 {
    var crc: UInt16 = 0xFFFF
    for byte in data {
      let index = Int((crc >> 8) ^ UInt16(byte))
      crc = (crc << 8) ^ table[index]
    }
    return crc
  }
}

/// Bit-level writer for building MP3 frame headers, side information, and Huffman-coded data.
private struct BitstreamWriter {
  private(set) var data = Data()
  private var buffer: UInt32 = 0
  private var bitsInBuffer: Int = 0

  /// Total number of bits written so far.
  var bitCount: Int {
    data.count * 8 + bitsInBuffer
  }

  /// Writes up to 24 bits at once in MSB-first order.
  mutating func write(bits: Int, count: Int) {
    guard count > 0 && count <= 24 else {
      for i in (0..<count).reversed() {
        writeBit((bits >> i) & 1)
      }
      return
    }

    buffer = (buffer << count) | UInt32(bits & ((1 << count) - 1))
    bitsInBuffer += count

    while bitsInBuffer >= 8 {
      bitsInBuffer -= 8
      let byte = UInt8((buffer >> bitsInBuffer) & 0xFF)
      data.append(byte)
    }

    if bitsInBuffer > 0 {
      buffer &= (1 << bitsInBuffer) - 1
    } else {
      buffer = 0
    }
  }

  /// Writes a single bit.
  private mutating func writeBit(_ bit: Int) {
    buffer = (buffer << 1) | UInt32(bit & 1)
    bitsInBuffer += 1
    if bitsInBuffer == 8 {
      data.append(UInt8(buffer & 0xFF))
      buffer = 0
      bitsInBuffer = 0
    }
  }

  /// Pads the bitstream to the next byte boundary with zero bits.
  mutating func padToByte() {
    if bitsInBuffer > 0 {
      let padding = 8 - bitsInBuffer
      buffer <<= padding
      data.append(UInt8(buffer & 0xFF))
      buffer = 0
      bitsInBuffer = 0
    }
  }
}

/// ISO 11172-3 lookup tables for MPEG-1 Layer III encoding.
private enum MP3Tables {
  /// A Huffman code table mapping value pairs to (bit length, codeword) entries.
  struct HuffmanTable {
    let maxValue: Int
    let table: [[(length: Int, bits: Int)]]
  }

  // MARK: Huffman Code Tables (ISO 11172-3 Table B.7)

  /// Table 1: 2x2 for values 0-1.
  static let table1 = HuffmanTable(
    maxValue: 1,
    table: [
      [(1, 1), (3, 1)],   // (0,0), (0,1)
      [(2, 1), (3, 0)]    // (1,0), (1,1)
    ]
  )

  /// Table 2: 3x3 for values 0-2.
  static let table2 = HuffmanTable(
    maxValue: 2,
    table: [
      [(1, 1), (3, 2), (6, 1)],
      [(3, 3), (3, 1), (5, 1)],
      [(5, 3), (5, 2), (6, 0)]
    ]
  )

  /// Table 3: 3x3 for values 0-2.
  static let table3 = HuffmanTable(
    maxValue: 2,
    table: [
      [(2, 3), (2, 2), (6, 1)],
      [(3, 1), (2, 1), (5, 1)],
      [(5, 3), (5, 2), (6, 0)]
    ]
  )

  /// Table 5: 4x4 for values 0-3.
  static let table5 = HuffmanTable(
    maxValue: 3,
    table: [
      [(1, 1), (3, 2), (6, 6), (7, 5)],
      [(3, 3), (3, 1), (6, 4), (7, 4)],
      [(6, 7), (6, 5), (7, 7), (8, 1)],
      [(7, 6), (6, 1), (7, 1), (8, 0)]
    ]
  )

  /// Table 6: 4x4 for values 0-3.
  static let table6 = HuffmanTable(
    maxValue: 3,
    table: [
      [(3, 7), (3, 3), (5, 5), (7, 1)],
      [(3, 6), (2, 2), (4, 3), (5, 2)],
      [(4, 5), (4, 4), (5, 4), (6, 1)],
      [(6, 3), (5, 3), (6, 2), (7, 0)]
    ]
  )

  /// Table 7: 6x6 for values 0-5.
  static let table7 = HuffmanTable(
    maxValue: 5,
    table: [
      [(1, 1), (3, 2), (6, 10), (8, 19), (8, 16), (9, 10)],
      [(3, 3), (4, 3), (6, 7), (7, 10), (7, 5), (8, 3)],
      [(6, 11), (5, 4), (7, 13), (8, 17), (8, 8), (9, 4)],
      [(7, 12), (7, 11), (8, 18), (9, 15), (9, 11), (9, 2)],
      [(7, 7), (7, 6), (8, 9), (9, 14), (9, 3), (10, 1)],
      [(8, 6), (8, 4), (9, 5), (10, 3), (10, 2), (10, 0)]
    ]
  )

  /// Table 8: 6x6 for values 0-5.
  static let table8 = HuffmanTable(
    maxValue: 5,
    table: [
      [(2, 3), (3, 4), (6, 6), (8, 18), (8, 12), (9, 5)],
      [(3, 5), (2, 1), (4, 2), (8, 16), (8, 9), (8, 3)],
      [(6, 7), (4, 3), (6, 5), (8, 14), (8, 7), (9, 3)],
      [(8, 19), (8, 17), (8, 15), (9, 13), (9, 10), (10, 4)],
      [(8, 13), (7, 5), (8, 8), (9, 11), (10, 5), (10, 1)],
      [(9, 12), (8, 4), (9, 4), (9, 1), (11, 1), (11, 0)]
    ]
  )

  /// Table 9: 6x6 for values 0-5.
  static let table9 = HuffmanTable(
    maxValue: 5,
    table: [
      [(3, 7), (3, 5), (5, 9), (6, 14), (8, 15), (9, 7)],
      [(3, 6), (3, 4), (4, 5), (5, 5), (6, 6), (8, 7)],
      [(4, 7), (4, 6), (5, 8), (6, 8), (7, 8), (8, 5)],
      [(6, 15), (5, 6), (6, 9), (7, 10), (7, 5), (8, 1)],
      [(7, 11), (6, 7), (7, 9), (7, 6), (8, 4), (9, 1)],
      [(8, 14), (7, 4), (8, 6), (8, 2), (9, 6), (9, 0)]
    ]
  )

  /// Table 10: 8x8 for values 0-7.
  static let table10 = HuffmanTable(
    maxValue: 7,
    table: [
      [(1, 1), (3, 2), (6, 10), (8, 23), (9, 35), (9, 30), (9, 12), (10, 17)],
      [(3, 3), (4, 3), (6, 8), (7, 12), (8, 18), (9, 21), (8, 12), (8, 7)],
      [(6, 11), (6, 9), (7, 15), (8, 21), (9, 32), (10, 40), (9, 19), (9, 6)],
      [(7, 14), (7, 13), (8, 22), (9, 34), (10, 46), (10, 23), (9, 18), (10, 7)],
      [(8, 20), (8, 19), (9, 33), (10, 47), (10, 27), (10, 22), (10, 9), (10, 3)],
      [(9, 31), (9, 22), (10, 41), (10, 26), (11, 53), (11, 52), (10, 5), (11, 3)],
      [(8, 14), (8, 13), (9, 10), (10, 11), (10, 16), (10, 6), (11, 5), (11, 1)],
      [(9, 9), (8, 8), (9, 7), (10, 8), (10, 4), (11, 4), (11, 2), (11, 0)]
    ]
  )

  /// Table 13: 16x16 for values 0-15.
  static let table13 = HuffmanTable(
    maxValue: 15,
    table: Self.buildTable13()
  )

  /// Table 15: 16x16 for values 0-15.
  static let table15 = HuffmanTable(
    maxValue: 15,
    table: Self.buildTable15()
  )

  /// Builds Huffman table 13 (16x16 for values 0-15).
  private static func buildTable13() -> [[(length: Int, bits: Int)]] {
    let lengths: [Int] = [
      1, 4, 6, 7, 8, 9, 9, 10, 9, 10, 11, 11, 12, 12, 13, 13,
      3, 4, 6, 7, 8, 8, 9, 9, 9, 9, 10, 10, 11, 12, 12, 12,
      6, 6, 7, 8, 9, 9, 10, 10, 9, 10, 10, 11, 11, 12, 13, 13,
      7, 7, 8, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 13, 13,
      8, 7, 9, 9, 10, 10, 11, 11, 10, 11, 11, 12, 12, 13, 13, 14,
      9, 8, 9, 10, 10, 10, 11, 11, 11, 11, 12, 11, 13, 13, 14, 14,
      9, 9, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14, 14,
      10, 9, 10, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 14, 16, 16,
      9, 8, 9, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 14, 15, 15,
      10, 9, 10, 10, 11, 11, 11, 13, 12, 13, 13, 14, 14, 14, 16, 15,
      10, 10, 10, 11, 11, 12, 12, 13, 12, 13, 14, 13, 14, 15, 16, 17,
      11, 10, 10, 11, 12, 12, 12, 12, 13, 13, 13, 14, 15, 15, 15, 16,
      11, 11, 11, 12, 12, 13, 12, 13, 14, 14, 15, 15, 15, 16, 16, 16,
      12, 11, 12, 13, 13, 13, 14, 14, 14, 14, 14, 15, 16, 15, 16, 16,
      13, 12, 12, 13, 13, 13, 15, 14, 14, 17, 15, 15, 15, 17, 16, 16,
      12, 12, 13, 14, 14, 14, 15, 14, 15, 15, 16, 16, 19, 18, 19, 16
    ]

    let codes: [Int] = [
      1, 5, 14, 21, 34, 51, 46, 71, 42, 52, 68, 52, 259, 172, 683, 275,
      3, 4, 12, 19, 31, 26, 44, 33, 31, 24, 32, 24, 31, 35, 22, 14,
      15, 13, 23, 36, 59, 49, 77, 65, 29, 40, 30, 40, 27, 33, 682, 272,
      22, 20, 37, 61, 56, 79, 73, 64, 43, 76, 56, 37, 26, 31, 57, 14,
      35, 16, 60, 57, 97, 75, 114, 91, 54, 73, 55, 161, 144, 117, 87, 48,
      58, 27, 50, 96, 76, 70, 93, 84, 77, 58, 79, 29, 74, 49, 169, 145,
      47, 45, 78, 74, 115, 94, 90, 79, 69, 83, 71, 50, 475, 38, 164, 15,
      72, 34, 56, 95, 92, 85, 91, 90, 86, 73, 77, 65, 51, 172, 2987, 2986,
      43, 20, 30, 44, 55, 78, 72, 87, 78, 61, 94, 118, 37, 62, 84, 80,
      52, 25, 41, 37, 44, 59, 54, 81, 66, 76, 57, 54, 37, 18, 167, 11,
      35, 33, 31, 57, 42, 82, 72, 80, 95, 58, 55, 21, 22, 26, 166, 22,
      53, 25, 23, 38, 70, 60, 51, 36, 55, 26, 34, 23, 27, 14, 9, 7,
      45, 32, 28, 39, 49, 75, 30, 52, 48, 40, 52, 28, 18, 33, 9, 5,
      173, 21, 34, 64, 56, 50, 49, 45, 31, 19, 12, 15, 10, 7, 6, 3,
      48, 23, 20, 39, 36, 35, 37, 21, 16, 23, 13, 10, 6, 1, 4, 2,
      16, 15, 17, 27, 25, 20, 29, 11, 17, 12, 16, 8, 1, 1, 0, 1
    ]

    var table = [[(length: Int, bits: Int)]](repeating: [], count: 16)
    for x in 0..<16 {
      table[x] = []
      for y in 0..<16 {
        let idx = x * 16 + y
        table[x].append((lengths[idx], codes[idx]))
      }
    }
    return table
  }

  /// Builds Huffman table 15 (16x16 for values 0-15, no linbits).
  private static func buildTable15() -> [[(length: Int, bits: Int)]] {
    let lengths: [Int] = [
      3, 4, 5, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11, 12, 13,
      4, 3, 5, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11,
      5, 5, 5, 6, 7, 7, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11,
      6, 6, 6, 7, 7, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11,
      7, 6, 7, 7, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11,
      8, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 11, 11, 11, 12,
      9, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 12, 12,
      9, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 12,
      9, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12, 12, 12,
      9, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12,
      10, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 12, 13, 12,
      10, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 13,
      11, 10, 9, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 13, 13,
      11, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13,
      12, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 12, 13,
      12, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13
    ]

    let codes: [Int] = [
      7, 12, 18, 53, 47, 76, 124, 108, 89, 123, 108, 119, 107, 81, 122, 63,
      13, 5, 16, 27, 46, 36, 61, 51, 42, 70, 52, 83, 65, 41, 59, 36,
      19, 17, 15, 24, 41, 34, 59, 48, 40, 64, 50, 78, 62, 80, 56, 33,
      29, 28, 25, 43, 39, 63, 55, 93, 76, 59, 93, 72, 54, 75, 50, 29,
      52, 22, 42, 40, 67, 57, 95, 79, 72, 57, 89, 69, 49, 66, 46, 27,
      77, 37, 35, 66, 58, 52, 91, 74, 62, 48, 79, 63, 90, 62, 40, 38,
      125, 32, 60, 56, 50, 92, 78, 65, 55, 87, 71, 51, 73, 51, 70, 30,
      109, 53, 49, 94, 88, 75, 66, 122, 91, 73, 56, 42, 64, 44, 21, 25,
      90, 43, 41, 77, 73, 63, 56, 92, 77, 66, 47, 67, 48, 53, 36, 20,
      71, 34, 67, 60, 58, 49, 88, 76, 67, 106, 71, 54, 38, 39, 23, 15,
      109, 53, 51, 47, 90, 82, 58, 57, 48, 72, 57, 41, 23, 27, 62, 9,
      86, 42, 40, 37, 70, 64, 52, 43, 70, 55, 42, 25, 29, 18, 11, 11,
      118, 68, 30, 55, 50, 46, 74, 65, 49, 39, 24, 16, 22, 13, 14, 7,
      91, 44, 39, 38, 34, 63, 52, 45, 31, 52, 28, 19, 14, 8, 9, 3,
      123, 60, 58, 53, 47, 43, 32, 22, 37, 24, 17, 12, 15, 10, 2, 1,
      71, 37, 34, 30, 28, 20, 17, 26, 21, 16, 10, 6, 8, 6, 2, 0
    ]

    var table = [[(length: Int, bits: Int)]](repeating: [], count: 16)
    for x in 0..<16 {
      table[x] = []
      for y in 0..<16 {
        let idx = x * 16 + y
        table[x].append((lengths[idx], codes[idx]))
      }
    }
    return table
  }

  // MARK: Bitrate and Sample Rate Lookup

  /// Returns the MPEG-1 Layer III bitrate index for a given bitrate in kbps.
  static func bitrateIndex(for bitrate: Int, sampleRate: Int) -> Int {
    let table: [Int]
    if sampleRate >= 32_000 {
      table = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    } else {
      table = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0]
    }
    if let index = table.firstIndex(of: bitrate) {
      return index
    }
    if let closest = table.enumerated().min(by: { abs($0.element - bitrate) < abs($1.element - bitrate) }) {
      return closest.offset
    }
    return 9
  }

  /// Returns the bitrate in kbps for a given bitrate index (MPEG-1 Layer III).
  static func bitrateValue(for index: Int) -> Int {
    let table = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0]
    guard index >= 0 && index < table.count else { return 128 }
    return table[index]
  }

  /// Returns the MPEG-1 sample rate index for a given sample rate in Hz.
  static func sampleRateIndex(for sampleRate: Int) -> Int {
    switch sampleRate {
    case 44_100:
      return 0
    case 48_000:
      return 1
    case 32_000:
      return 2
    default:
      return 0
    }
  }
  
  /// Returns the (mode, mode_extension) header bits for a channel mode.
  static func modeBits(for mode: MP3EncoderOptions.Mode) -> (Int, Int) {
    switch mode {
    case .mono:
      return (0b11, 0)
    case .jointStereo:
      return (0b01, 0b10)
    case .stereo:
      return (0b00, 0)
    }
  }
}
