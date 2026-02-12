// SwiftMP3
// MP3Encoder.swift

import Foundation
import Accelerate

public struct MP3EncoderOptions: Sendable, Equatable {
  public enum Mode: String, Sendable, Equatable {
    case mono
    case stereo
    case jointStereo
  }

  public var sampleRate: Int
  public var bitrateKbps: Int
  public var vbr: Bool
  public var mode: Mode
  public var quality: Int
  public var crcProtected: Bool
  public var original: Bool
  public var copyright: Bool

  public init(
    sampleRate: Int = 44_100,
    bitrateKbps: Int = 128,
    vbr: Bool = false,
    mode: Mode = .stereo,
    quality: Int = 5,
    crcProtected: Bool = false,
    original: Bool = true,
    copyright: Bool = false
  ) {
    self.sampleRate = sampleRate
    self.bitrateKbps = bitrateKbps
    self.vbr = vbr
    self.mode = mode
    self.quality = max(0, min(quality, 9))
    self.crcProtected = crcProtected
    self.original = original
    self.copyright = copyright
  }
}

@preconcurrency
public final class MP3Encoder {
  private static let samplesPerFrame = 1152
  private static let subbands = 32
  private static let samplesPerGranule = 576

  private let options: MP3EncoderOptions
  private var pcmBuffer: [Float] = []
  private var vbrState = VBRState()
  private var reservoir = BitReservoir()

  // Polyphase filterbank state: 512-sample buffer per channel
  private var filterbankBuffers: [[Float]] = []
  // MDCT overlap: 18 samples per subband per channel
  private var mdctOverlap: [[[Float]]] = []

  // Pre-allocated working buffers to reduce allocations
  private var workingSpectrum: [Float] = []
  private var workingQuantized: [Int] = []
  private var workingSubbands: [[Float]] = []

  // Xing header tracking
  private var frameCount: UInt32 = 0
  private var totalBytes: UInt32 = 0
  private var frameSizes: [Int] = []  // For TOC generation

  /// Returns the number of frames encoded so far.
  public var encodedFrameCount: UInt32 { frameCount }

  /// Returns the total bytes of audio data encoded so far (excluding Xing header).
  public var encodedByteCount: UInt32 { totalBytes }

  public init(options: MP3EncoderOptions) {
    self.options = options
    let channels = options.mode == .mono ? 1 : 2

    // Initialize polyphase filterbank buffers (512 samples each)
    self.filterbankBuffers = Array(repeating: Array(repeating: 0, count: 512), count: channels)

    // Initialize MDCT overlap buffers (18 samples per subband per channel)
    self.mdctOverlap = Array(
      repeating: Array(repeating: Array(repeating: 0, count: 18), count: Self.subbands),
      count: channels
    )

    // Pre-allocate working buffers
    self.workingSpectrum = [Float](repeating: 0, count: Self.samplesPerGranule)
    self.workingQuantized = [Int](repeating: 0, count: Self.samplesPerGranule)
    self.workingSubbands = Array(repeating: [Float](repeating: 0, count: 18), count: Self.subbands)

    // Reserve space for frame sizes (estimate based on typical file lengths)
    self.frameSizes.reserveCapacity(10000)
  }

  public func appendSamples(_ samples: [Float]) -> Data {
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

  public func flush() -> Data {
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

  /// Generates a Xing/Info header frame to prepend to the MP3 data.
  /// Call this after encoding is complete to get accurate frame/byte counts.
  /// The returned data should be prepended to the encoded audio data.
  public func makeXingHeader() -> Data {
    let channels = self.options.mode == .mono ? 1 : 2
    let sideInfoSize = channels == 1 ? 17 : 32

    // Use the base bitrate for the Xing frame (common practice)
    let bitrateIndex = MP3Tables.bitrateIndex(for: self.options.bitrateKbps, sampleRate: self.options.sampleRate)
    let sampleRateIndex = MP3Tables.sampleRateIndex(for: self.options.sampleRate)
    let bitrateValue = MP3Tables.bitrateValue(for: bitrateIndex)
    let frameSize = (144 * bitrateValue * 1000) / self.options.sampleRate

    let (modeBits, modeExtension) = MP3Tables.modeBits(for: self.options.mode)

    // Build MP3 frame header
    var header = BitstreamWriter()
    header.write(bits: 0x7FF, count: 11)  // Sync
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

    // Empty side info (zeroed)
    frame.append(contentsOf: repeatElement(UInt8(0), count: sideInfoSize))

    // Xing header content
    // Use "Xing" for VBR, "Info" for CBR (though both work)
    let tag = self.options.vbr ? "Xing" : "Info"
    frame.append(contentsOf: tag.utf8)

    // Flags: frames present (0x01), bytes present (0x02), TOC present (0x04)
    let flags: UInt32 = 0x07
    frame.append(contentsOf: flags.bigEndianBytes)

    // Frame count (including this Xing frame)
    let totalFrames = self.frameCount + 1
    frame.append(contentsOf: totalFrames.bigEndianBytes)

    // Byte count (including this Xing frame)
    let xingFrameSize = UInt32(frameSize)
    let totalByteCount = self.totalBytes + xingFrameSize
    frame.append(contentsOf: totalByteCount.bigEndianBytes)

    // TOC (Table of Contents) - 100 bytes
    // Maps percentage (0-99) to byte position as fraction of file (0-255)
    let toc = self.generateTOC()
    frame.append(contentsOf: toc)

    // Pad frame to full size
    let currentSize = frame.count
    if currentSize < frameSize {
      frame.append(contentsOf: repeatElement(UInt8(0), count: frameSize - currentSize))
    }

    return frame
  }

  /// Generates the 100-byte TOC (Table of Contents) for seeking.
  /// Each entry maps a percentage (0-99%) to a byte position (0-255 scale).
  private func generateTOC() -> [UInt8] {
    guard !self.frameSizes.isEmpty else {
      // No frames encoded, return linear TOC
      return (0..<100).map { UInt8($0 * 255 / 99) }
    }

    // Calculate cumulative byte positions
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

    // For each percentage point, find the byte position
    var toc = [UInt8]()
    for percent in 0..<100 {
      let targetFrame = (percent * self.frameSizes.count) / 100
      let bytePosition = targetFrame > 0 ? cumulative[targetFrame - 1] : 0
      let scaled = (bytePosition * 255) / totalBytes
      toc.append(UInt8(min(scaled, 255)))
    }

    return toc
  }
  
  private func encodeFrame(samples: [Float]) -> Data {
    let channels = self.options.mode == .mono ? 1 : 2
    let frameEnergy = FrameAnalysis.energy(samples: samples)
    let targetBitrate = self.options.vbr
      ? self.vbrState.chooseBitrate(base: self.options.bitrateKbps, energy: frameEnergy, quality: self.options.quality)
      : self.options.bitrateKbps
    let bitrateIndex = MP3Tables.bitrateIndex(for: targetBitrate, sampleRate: self.options.sampleRate)
    let sampleRateIndex = MP3Tables.sampleRateIndex(for: self.options.sampleRate)

    // Calculate frame size: floor(144 * bitrate / sampleRate) + padding
    // For MPEG-1 Layer III
    let bitrateValue = MP3Tables.bitrateValue(for: bitrateIndex)
    let baseFrameSize = (144 * bitrateValue * 1000) / self.options.sampleRate
    let sideInfoSize = channels == 1 ? 17 : 32
    let headerSize = 4
    let crcSize = self.options.crcProtected ? 2 : 0

    // Determine if padding is needed (for this simple implementation, no padding)
    let padding = 0
    let frameSize = baseFrameSize + padding
    let mainDataSize = frameSize - headerSize - crcSize - sideInfoSize

    let (modeBits, modeExtension) = MP3Tables.modeBits(for: self.options.mode)

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

    // Pad or truncate main data to exact size
    var paddedMainData = mainData
    if paddedMainData.count < mainDataSize {
      paddedMainData.append(contentsOf: repeatElement(0, count: mainDataSize - paddedMainData.count))
    } else if paddedMainData.count > mainDataSize {
      paddedMainData = paddedMainData.prefix(mainDataSize)
    }
    frame.append(paddedMainData)

    // Track for Xing header
    self.frameCount += 1
    self.totalBytes += UInt32(frame.count)
    self.frameSizes.append(frame.count)

    return frame
  }
  
  private func buildSideInfo(channels: Int, granules: [[GranuleInfo]], scfsi: [[Int]]) -> Data {
    var writer = BitstreamWriter()

    // Side info size: 17 bytes (136 bits) for mono, 32 bytes (256 bits) for stereo
    let sideInfoBits = channels == 1 ? 136 : 256

    // MPEG-1 Layer III side information structure:
    // - main_data_begin: 9 bits (byte offset back from frame sync)
    // - private_bits: 5 bits (mono) or 3 bits (stereo)
    // - scfsi[ch][band]: 1 bit × 4 bands × channels
    // - granule info for 2 granules × channels

    // Use 0 for main_data_begin (no bit reservoir - simpler and more compatible)
    writer.write(bits: 0, count: 9)

    // Private bits
    let privateBits = channels == 1 ? 5 : 3
    writer.write(bits: 0, count: privateBits)

    // Scale factor selection information (scfsi) - 4 bits per channel
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
  
  private func buildMainData(
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

    // Calculate bit budget per granule
    let totalMainDataBits = targetMainDataSize * 8
    let granuleCount = 2 * channels
    let bitsPerGranule = totalMainDataBits / granuleCount

    // MP3 Layer III has 2 granules per frame, each with 576 samples per channel
    for granuleIndex in 0..<2 {
      for channelIndex in 0..<channels {
        let channelSamples = stereoDecision.samples(for: channelIndex)

        // Extract samples for this granule (576 samples)
        let granuleStart = granuleIndex * Self.samplesPerGranule
        let granuleEnd = min(granuleStart + Self.samplesPerGranule, channelSamples.count)
        let granuleSamples: [Float]
        if granuleStart < channelSamples.count {
          granuleSamples = Array(channelSamples[granuleStart..<granuleEnd])
        } else {
          granuleSamples = Array(repeating: 0, count: Self.samplesPerGranule)
        }

        // Encode through polyphase filterbank + MDCT
        let spectrum = self.encodeSpectrum(
          granuleSamples,
          channel: channelIndex,
          sampleRate: self.options.sampleRate
        )

        self.vbrState.update(
          globalGain: spectrum.globalGain,
          energy: FrameAnalysis.energy(samples: granuleSamples)
        )

        // Iterate on global_gain to fit bit budget (like real encoders)
        let (finalGain, quantized, huffmanBits) = self.quantizeToFitBudget(
          spectral: spectrum.spectral,
          scalefactors: spectrum.scalefactors,
          thresholds: spectrum.maskingThresholds,
          initialGain: spectrum.globalGain,
          maxBits: bitsPerGranule,
          writer: &writer
        )

        // Use scalefac_compress = 0, which means slen1=0, slen2=0
        let scalefacCompress = 0

        let preflag = PreEmphasis.shouldEnable(
          spectral: spectrum.spectral,
          scalefactors: spectrum.scalefactors
        ) ? 1 : 0

        // Find bigValues from quantized data
        var lastNonZero = 0
        for i in stride(from: quantized.count - 1, through: 0, by: -1) {
          if quantized[i] != 0 {
            lastNonZero = i + 1
            break
          }
        }
        let significantCount = (lastNonZero + 1) & ~1
        let bigValues = min(significantCount / 2, 288)

        // Calculate region boundaries based on bigValues
        let (region0Count, region1Count) = self.calculateRegionCounts(
          bigValues: bigValues,
          sampleRate: self.options.sampleRate
        )

        // Side info debug disabled - verified: bigValues ~141-143, region0Count=14, region1Count=2

        // part2_3_length = scalefactor_bits + huffman_bits
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

  /// Quantizes and encodes spectral data with gain adjustment to fit bit budget.
  /// Returns (finalGain, quantizedValues, bitsWritten)
  private func quantizeToFitBudget(
    spectral: [Float],
    scalefactors: [Float],
    thresholds: [Float],
    initialGain: Int,
    maxBits: Int,
    writer: inout BitstreamWriter
  ) -> (Int, [Int], Int) {
    var gain = initialGain

    // Clamp gain to valid range
    gain = min(max(gain, 0), 255)

    // Iterate to find a gain that fits within bit budget
    // Higher gain = larger quantizer step = smaller values = fewer bits
    var quantized = [Int]()
    var estimatedBits = 0
    let maxIterations = 20

    for iteration in 0..<maxIterations {
      quantized = self.quantizeWithGain(spectral: spectral, globalGain: gain)

      // Find significant (non-zero) values
      var lastNonZero = 0
      for i in stride(from: quantized.count - 1, through: 0, by: -1) {
        if quantized[i] != 0 {
          lastNonZero = i + 1
          break
        }
      }

      // If no non-zero values and we haven't reduced gain much, try lower gain
      if lastNonZero == 0 && iteration == 0 {
        gain = max(gain - 40, 0)
        continue
      }

      // Count bits needed for this quantization
      let significantCount = min((lastNonZero + 1) & ~1, 576)
      let bigValues = min(significantCount / 2, 288)
      let valuesToCount = Array(quantized.prefix(bigValues * 2))
      estimatedBits = self.countHuffmanBits(valuesToCount)

      // If we fit within budget, we're done
      if estimatedBits <= maxBits {
        break
      }

      // Increase gain to reduce bit usage (larger step = smaller quantized values)
      gain = min(gain + 4, 255)

      if gain >= 255 {
        break
      }
    }

    // Find final significant count
    var lastNonZero = 0
    for i in stride(from: quantized.count - 1, through: 0, by: -1) {
      if quantized[i] != 0 {
        lastNonZero = i + 1
        break
      }
    }

    // Quantization debug disabled - verified working correctly
    // Non-zero values concentrated in low frequencies (100-115 in indices 0-143, 0 in indices 432-575)

    // Limit to what we can encode
    let significantCount = min((lastNonZero + 1) & ~1, 576)
    let bigValues = min(significantCount / 2, 288)
    let valuesToEncode = Array(quantized.prefix(bigValues * 2))

    // Encode with Huffman
    let huffman = HuffmanEncoder()
    let actualBits = huffman.encodeWithTable15(valuesToEncode, writer: &writer)

    return (gain, quantized, actualBits)
  }

  /// Quantize spectral data with a specific global gain using vectorized operations.
  private func quantizeWithGain(spectral: [Float], globalGain: Int) -> [Int] {
    let stepPower = Double(globalGain - 210) / 4.0
    let quantizerStep = Float(max(pow(2.0, stepPower), 0.0001))
    var invStep = 1.0 / quantizerStep

    // Table 15 is a 16x16 table for values 0-15, no linbits
    let maxQuantized: Int32 = 15
    let count = spectral.count

    // Get absolute values using vDSP
    var absValues = [Float](repeating: 0, count: count)
    vDSP_vabs(spectral, 1, &absValues, 1, vDSP_Length(count))

    // Apply power of 0.75 using vForce
    // pow(x, 0.75) = exp(0.75 * log(x))
    // But vvpowf is simpler: vvpowf(result, exponent, base, count)
    var exponent = [Float](repeating: 0.75, count: count)
    var magnitudes = [Float](repeating: 0, count: count)

    // Replace zeros with tiny values to avoid log(0)
    var threshold: Float = 1e-10
    vDSP_vthr(absValues, 1, &threshold, &absValues, 1, vDSP_Length(count))

    // Vectorized power: magnitudes = absValues ^ 0.75
    vvpowf(&magnitudes, &exponent, absValues, [Int32(count)])

    // Scale by inverse step and round
    var scaled = [Float](repeating: 0, count: count)
    vDSP_vsmul(magnitudes, 1, &invStep, &scaled, 1, vDSP_Length(count))

    // Convert to integers with clamping
    var result = [Int](repeating: 0, count: count)
    for i in 0..<count {
      let quantized = min(Int32(scaled[i].rounded()), maxQuantized)
      result[i] = spectral[i] < 0 ? -Int(quantized) : Int(quantized)
    }

    return result
  }

  /// Count huffman bits without writing (for iteration).
  /// Table 15 is a 16x16 table for values 0-15, NO linbits.
  private func countHuffmanBits(_ values: [Int]) -> Int {
    var bits = 0
    var index = 0

    while index + 1 < values.count {
      let absX = min(abs(values[index]), 15)
      let absY = min(abs(values[index + 1]), 15)

      let code = MP3Tables.table15.table[absX][absY]
      bits += code.length

      // Sign bits (only if value is non-zero)
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

  /// Calculates region0_count and region1_count based on bigValues.
  /// These define where the three Huffman table regions end within the big_values area.
  private func calculateRegionCounts(bigValues: Int, sampleRate: Int) -> (Int, Int) {
    // bigValues * 2 = number of spectral samples in regions 0, 1, 2
    let bigValuesRegion = bigValues * 2

    // Get scalefactor band boundaries for this sample rate
    let bands = ScaleFactorBands.bandTable(for: sampleRate)

    // Calculate cumulative boundaries
    var boundaries = [Int]()
    var cumulative = 0
    for width in bands {
      cumulative += width
      boundaries.append(cumulative)
    }

    // Find region0_count: highest band index where boundary <= bigValuesRegion
    // region0 ends at scalefactor band (region0_count + 1)
    var region0Count = 0
    for i in 0..<min(15, boundaries.count) {
      if boundaries[i] <= bigValuesRegion {
        region0Count = i
      } else {
        break
      }
    }

    // Find region1_count: additional bands after region0
    // region1 ends at scalefactor band (region0_count + region1_count + 2)
    var region1Count = 0
    let region1Start = region0Count + 1
    for i in region1Start..<min(region1Start + 7, boundaries.count) {
      if i < boundaries.count && boundaries[i] <= bigValuesRegion {
        region1Count = i - region0Count - 1
      } else {
        break
      }
    }

    // Clamp to valid ranges (region0_count: 0-15, region1_count: 0-7)
    return (min(region0Count, 15), min(region1Count, 7))
  }
  
  /// Deinterleaves stereo samples using vDSP stride operations.
  private func deinterleave(samples: [Float], channels: Int) -> [[Float]] {
    guard channels > 1 else {
      return [samples]
    }

    let samplesPerChannel = samples.count / channels
    var result = [[Float]](repeating: [Float](repeating: 0, count: samplesPerChannel), count: channels)

    // Use vDSP to extract each channel with stride
    samples.withUnsafeBufferPointer { srcPtr in
      for ch in 0..<channels {
        result[ch].withUnsafeMutableBufferPointer { dstPtr in
          // Copy every nth sample starting at offset ch
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
  
  /// Processes PCM samples through the polyphase filterbank to generate subband samples.
  /// - Parameters:
  ///   - samples: 576 PCM samples for one granule
  ///   - channel: Channel index for accessing filterbank buffer
  /// - Returns: 32 subbands, each with 18 samples
  private func analyzeSubbands(_ samples: [Float], channel: Int) -> [[Float]] {
    var subbands = Array(repeating: [Float](), count: Self.subbands)

    // Process 32 samples at a time through the filterbank
    // 576 samples / 32 = 18 filterbank iterations = 18 subband samples per subband
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

  private func encodeSpectrum(
    _ samples: [Float],
    channel: Int,
    sampleRate: Int
  ) -> SpectrumResult {
    // Step 1: Analyze PCM through polyphase filterbank → 32 subbands × 18 samples
    let subbands = self.analyzeSubbands(samples, channel: channel)

    // Filterbank debug disabled - verified working correctly
    // Energy distribution: ~92-94% in low subbands (0-7), ~5-7% in mid, ~0% in high

    // Step 2: Detect transients for block type decision
    let transient = TransientDetector.analyze(samples: samples)

    // Step 3: Apply MDCT to subband samples → 576 frequency coefficients
    let mdct = MDCT.apply(
      subbandSamples: subbands,
      overlap: &self.mdctOverlap[channel],
      blockType: transient.blockType
    )

    // MDCT debug disabled - verified working correctly
    // Energy distribution: ~93-94% in low subbands, ~0% in high subbands

    // Step 4: Psychoacoustic masking thresholds (for noise allocation, not used yet)
    let thresholds = PsychoacousticModel.maskingThresholds(
      spectrum: mdct,
      sampleRate: sampleRate,
      quality: self.options.quality
    )

    // Don't normalize the spectrum - pass MDCT output directly to quantization.
    // Scalefactors in MP3 are for the decoder to scale reconstructed values,
    // but since we're using scalefac_compress=0, we encode no scalefactors.
    // The global_gain alone controls the quantization step size.
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
  
  /// Computes global gain from MDCT spectral coefficients.
  /// According to ISO 11172-3: global_gain controls the quantizer step size.
  /// Formula: quantizer_step = 2^((global_gain - 210) / 4)
  private func computeGlobalGain(_ spectral: [Float]) -> Int {
    // Find the peak magnitude in the spectrum
    let peak = spectral.map { abs($0) }.max() ?? 0

    guard peak > 0 else {
      return 210  // Default gain for silence
    }

    // We want to find gain such that the largest quantized value fits in table range.
    // Since we're using table 15 (max value 15), target max = 15
    // quantized = |x|^0.75 / 2^((gain - 210) / 4)
    // Solve for gain: gain = 210 + 4 * log2(|x|^0.75 / target)
    let targetMax: Float = 15.0
    let peakPow = pow(peak, 0.75)
    let ratio = peakPow / targetMax

    if ratio <= 0 {
      return 210
    }

    let gain = 210 + Int(4.0 * log2(Double(ratio)))
    return min(max(gain, 0), 255)
  }

  /// Quantizes spectral coefficients according to ISO 11172-3.
  /// Formula: ix = nint(|xr|^0.75 / 2^((global_gain - 210) / 4))
  private func quantizeSpectral(
    _ spectral: [Float],
    scalefactors: [Float],
    thresholds: [Float],
    globalGain: Int
  ) -> [Int] {
    // Quantizer step: 2^((global_gain - 210) / 4)
    let stepPower = Double(globalGain - 210) / 4.0
    let quantizerStep = max(pow(2.0, stepPower), 0.0001)

    return spectral.enumerated().map { _, value in
      // MP3 power-law quantization: ix = nint(|xr|^0.75 / step)
      let absValue = abs(value)

      // Skip very small values to avoid numerical issues
      guard absValue > 1e-10 else {
        return 0
      }

      let magnitude = pow(Double(absValue), 0.75)
      let quantized = Int((magnitude / quantizerStep).rounded())

      // Global gain is calculated to target max 15, but clamp just in case
      let clamped = min(quantized, 15)

      return value < 0 ? -clamped : clamped
    }
  }
}

private struct VBRState {
  private var gainHistory: [Int] = []
  private var energyHistory: [Float] = []
  
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
  
  func globalGain(quality: Int) -> Int {
    let average = gainHistory.isEmpty ? 180 : gainHistory.reduce(0, +) / gainHistory.count
    return min(max(average + (9 - quality) * 2, 0), 255)
  }
  
  func estimatePart23Length(quality: Int) -> Int {
    let base = 450
    return max(0, base - quality * 30)
  }
  
  /// Chooses bitrate for VBR encoding based on frame energy and quality setting.
  /// Quality 0 = highest quality (use more bits), quality 9 = lowest (use fewer bits).
  func chooseBitrate(base: Int, energy: Float, quality: Int) -> Int {
    let average = energyHistory.isEmpty ? energy : energyHistory.reduce(0, +) / Float(energyHistory.count)
    let ratio = min(max(energy / max(average, 0.0001), 0.5), 2.0)

    // Quality affects how much we can deviate from base and the floor/ceiling
    // Quality 0: wide range (64-320), aggressive adjustments
    // Quality 9: narrow range, conservative adjustments
    let qualityFactor = Float(9 - quality) / 9.0  // 1.0 for quality 0, 0.0 for quality 9
    let maxAdjustment = Int(32.0 + 32.0 * qualityFactor)  // 32-64 kbps swing
    let adjustment = Int((ratio - 1.0) * Float(maxAdjustment))

    // Quality also affects the minimum bitrate floor
    let minBitrate = max(32, base - 64 + quality * 8)  // Higher quality = lower allowed minimum
    let maxBitrate = min(320, base + 64 - quality * 4)  // Higher quality = higher allowed maximum

    return max(minBitrate, min(base + adjustment, maxBitrate))
  }
}

// MARK: - Polyphase Analysis Filterbank
// MP3 encoding requires splitting PCM into 32 frequency subbands before MDCT

private enum PolyphaseFilterbank {
  // Pre-computed analysis matrix: M[k][n] = cos((2k+1)(n-16)*PI/64) for k=0..31, n=0..63
  // Computed once at initialization to avoid repeated cos() calls
  nonisolated(unsafe) private static var analysisMatrix: [[Float]]?

  private static func getAnalysisMatrix() -> [[Float]] {
    if let matrix = analysisMatrix { return matrix }
    var matrix = [[Float]](repeating: [Float](repeating: 0, count: 64), count: 32)
    for k in 0..<32 {
      for n in 0..<64 {
        let angle = Double.pi / 64.0 * Double(2 * k + 1) * (Double(n) - 16.0)
        matrix[k][n] = Float(cos(angle))
      }
    }
    analysisMatrix = matrix
    return matrix
  }

  // ISO 11172-3 Table C.1 - Analysis window coefficients (512 values)
  // These are the exact coefficients from the MPEG Audio standard.
  // Source: ISO/IEC 11172-3:1993 Table 3-C.1
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

  /// Gets the 512-tap analysis window coefficients.
  /// Based on ISO 11172-3 Table C.1 - analysis window coefficients.
  private static func getAnalysisWindow() -> [Float] {
    return isoAnalysisWindow
  }

  /// Analysis filterbank: converts 32 new PCM samples to 32 subband samples.
  /// Implements the ISO 11172-3 analysis filterbank algorithm.
  /// Uses Accelerate framework for vectorized operations.
  /// - Parameters:
  ///   - newSamples: 32 new PCM samples
  ///   - buffer: 512-sample FIFO buffer (updated in place)
  /// - Returns: 32 subband samples
  static func analyze(newSamples: [Float], buffer: inout [Float]) -> [Float] {
    // Ensure buffer is 512 samples
    if buffer.count < 512 {
      buffer = [Float](repeating: 0, count: 512 - buffer.count) + buffer
    }

    // Shift buffer left by 32 and add new samples at end using memmove
    // This is more efficient than removeFirst which is O(n)
    _ = buffer.withUnsafeMutableBufferPointer { ptr in
      // Move 480 samples from offset 32 to offset 0
      memmove(ptr.baseAddress!, ptr.baseAddress! + 32, 480 * MemoryLayout<Float>.size)
    }
    // Copy new samples to the end
    let copyCount = min(32, newSamples.count)
    for i in 0..<copyCount {
      buffer[480 + i] = newSamples[i]
    }
    // Zero-fill if newSamples is short
    for i in copyCount..<32 {
      buffer[480 + i] = 0
    }

    let window = getAnalysisWindow()

    // Step 1: Window the 512-sample buffer using vDSP
    // Reverse the buffer first for proper alignment with window using vDSP
    var reversed = buffer
    vDSP_vrvrs(&reversed, 1, 512)
    var windowed = [Float](repeating: 0, count: 512)
    vDSP_vmul(reversed, 1, window, 1, &windowed, 1, 512)

    // Step 2: Partial calculation - sum every 64th sample
    // Y[j] = sum(Z[i*64 + j]) for i=0..7
    // Use pointer arithmetic to avoid creating intermediate arrays
    var partial = [Float](repeating: 0, count: 64)
    windowed.withUnsafeBufferPointer { ptr in
      for j in 0..<64 {
        var sum: Float = 0
        // Sum 8 values at stride 64 using vDSP
        vDSP_sve(ptr.baseAddress! + j, 64, &sum, 8)
        partial[j] = sum
      }
    }

    // Step 3: Matrix multiply using pre-computed analysis matrix and vDSP
    let matrix = getAnalysisMatrix()
    var subbands = [Float](repeating: 0, count: 32)
    for k in 0..<32 {
      var result: Float = 0
      vDSP_dotpr(partial, 1, matrix[k], 1, &result, 64)
      subbands[k] = result
    }

    return subbands
  }
}

// MARK: - MDCT for MP3 Layer III

private enum MDCT {
  // Windows for MDCT - sized correctly per MP3 spec
  nonisolated(unsafe) private static var longWindow: [Float]?
  nonisolated(unsafe) private static var shortWindow: [Float]?
  nonisolated(unsafe) private static var startWindow: [Float]?
  nonisolated(unsafe) private static var stopWindow: [Float]?

  // Pre-computed MDCT cosine tables
  // Long block: cos(PI/(2*36) * (2k + 1 + 18) * (2m + 1)) for k=0..35, m=0..17
  nonisolated(unsafe) private static var longMDCTMatrix: [[Float]]?
  // Short block: cos(PI/(2*12) * (2k + 1 + 6) * (2m + 1)) for k=0..11, m=0..5
  nonisolated(unsafe) private static var shortMDCTMatrix: [[Float]]?

  private static func getLongMDCTMatrix() -> [[Float]] {
    if let matrix = longMDCTMatrix { return matrix }
    let n = 36
    let halfN = 18
    var matrix = [[Float]](repeating: [Float](repeating: 0, count: n), count: halfN)
    for m in 0..<halfN {
      for k in 0..<n {
        let angle = Double.pi / Double(2 * n) * Double(2 * k + 1 + n / 2) * Double(2 * m + 1)
        matrix[m][k] = Float(cos(angle))
      }
    }
    longMDCTMatrix = matrix
    return matrix
  }

  private static func getShortMDCTMatrix() -> [[Float]] {
    if let matrix = shortMDCTMatrix { return matrix }
    let n = 12
    let halfN = 6
    var matrix = [[Float]](repeating: [Float](repeating: 0, count: n), count: halfN)
    for m in 0..<halfN {
      for k in 0..<n {
        let angle = Double.pi / Double(2 * n) * Double(2 * k + 1 + n / 2) * Double(2 * m + 1)
        matrix[m][k] = Float(cos(angle))
      }
    }
    shortMDCTMatrix = matrix
    return matrix
  }

  /// Long block window: 36 samples (used for MDCT input of 36 → 18 output)
  private static func getLongWindow() -> [Float] {
    if let window = longWindow { return window }
    let n = 36
    var window = [Float](repeating: 0, count: n)
    for i in 0..<n {
      window[i] = Float(sin(Double.pi / Double(n) * (Double(i) + 0.5)))
    }
    longWindow = window
    return window
  }

  /// Short block window: 12 samples (used for MDCT input of 12 → 6 output)
  private static func getShortWindow() -> [Float] {
    if let window = shortWindow { return window }
    let n = 12
    var window = [Float](repeating: 0, count: n)
    for i in 0..<n {
      window[i] = Float(sin(Double.pi / Double(n) * (Double(i) + 0.5)))
    }
    shortWindow = window
    return window
  }

  /// Start block window for long-to-short transition
  private static func getStartWindow() -> [Float] {
    if let window = startWindow { return window }
    var window = [Float](repeating: 0, count: 36)
    // First half: long window
    for i in 0..<18 {
      window[i] = Float(sin(Double.pi / 36.0 * (Double(i) + 0.5)))
    }
    // Transition region
    for i in 18..<24 {
      window[i] = 1.0
    }
    // Short window tail
    for i in 24..<30 {
      window[i] = Float(sin(Double.pi / 12.0 * (Double(i - 18) + 0.5)))
    }
    for i in 30..<36 {
      window[i] = 0.0
    }
    startWindow = window
    return window
  }

  /// Stop block window for short-to-long transition
  private static func getStopWindow() -> [Float] {
    if let window = stopWindow { return window }
    var window = [Float](repeating: 0, count: 36)
    for i in 0..<6 {
      window[i] = 0.0
    }
    // Short window rise
    for i in 6..<12 {
      window[i] = Float(sin(Double.pi / 12.0 * (Double(i - 6) + 0.5)))
    }
    // Transition region
    for i in 12..<18 {
      window[i] = 1.0
    }
    // Long window tail
    for i in 18..<36 {
      window[i] = Float(sin(Double.pi / 36.0 * (Double(i) + 0.5)))
    }
    stopWindow = window
    return window
  }

  /// Applies MDCT to subband samples for one granule.
  /// For long blocks: 18 subband samples → 18 MDCT coefficients per subband
  /// For short blocks: 18 subband samples → 3 windows × 6 coefficients per subband
  /// - Parameters:
  ///   - subbandSamples: Array of 32 subbands, each with 18 samples
  ///   - overlap: Previous 18 samples per subband for overlap-add
  ///   - blockType: Block type for windowing
  /// - Returns: 576 frequency-domain coefficients
  static func apply(subbandSamples: [[Float]], overlap: inout [[Float]], blockType: BlockType) -> [Float] {
    var output = [Float](repeating: 0, count: 576)

    for sb in 0..<32 {
      var currentSamples = sb < subbandSamples.count ? subbandSamples[sb] : [Float](repeating: 0, count: 18)
      let previousSamples = sb < overlap.count ? overlap[sb] : [Float](repeating: 0, count: 18)

      // Compensate for inversion in the analysis filter.
      // For odd-numbered subbands, only odd-indexed samples are inverted.
      if (sb & 1) != 0 {
        for k in stride(from: 1, to: min(currentSamples.count, 18), by: 2) {
          currentSamples[k] *= -1
        }
      }

      // Combine previous and current for 36-sample window
      var combined = [Float](repeating: 0, count: 36)
      for i in 0..<18 {
        combined[i] = i < previousSamples.count ? previousSamples[i] : 0
      }
      for i in 0..<18 {
        combined[18 + i] = i < currentSamples.count ? currentSamples[i] : 0
      }

      // Update overlap buffer
      if sb < overlap.count {
        overlap[sb] = Array(currentSamples.prefix(18))
        while overlap[sb].count < 18 {
          overlap[sb].append(0)
        }
      }

      // Apply MDCT based on block type
      let mdctCoeffs: [Float]
      switch blockType {
      case .long:
        mdctCoeffs = mdctLong(samples: combined, window: getLongWindow())
      case .short:
        mdctCoeffs = mdctShort(samples: combined)
      case .mixed:
        if sb < 2 {
          // Low frequency subbands use long blocks
          mdctCoeffs = mdctLong(samples: combined, window: getLongWindow())
        } else {
          mdctCoeffs = mdctShort(samples: combined)
        }
      }

      // Output in subband-major order (MP3 standard frequency ordering)
      // Each subband's 18 MDCT coefficients are stored consecutively
      for i in 0..<18 {
        output[sb * 18 + i] = i < mdctCoeffs.count ? mdctCoeffs[i] : 0
      }
    }

    // Apply aliasing reduction butterfly (ISO 11172-3 Table B.9)
    // This cancels aliasing between adjacent subbands
    if blockType == .long {
      applyAliasingReduction(&output)
    }

    return output
  }

  /// Aliasing reduction coefficients from ISO 11172-3 Table B.9
  /// cs[i]^2 + ca[i]^2 = 1 for all i
  /// Using SIMD8 for vectorized butterfly operations.
  private static let aliasingCS = SIMD8<Float>(
    0.857492926, 0.881741997, 0.949628649, 0.983314592,
    0.995517816, 0.999160558, 0.999899195, 0.999993155
  )
  private static let aliasingCA = SIMD8<Float>(
    -0.514495755, -0.471731969, -0.313377454, -0.181913200,
    -0.094574193, -0.040965583, -0.014198569, -0.003699975
  )

  /// Applies aliasing reduction butterfly between adjacent subbands using SIMD.
  /// For long blocks only (short blocks don't need this).
  private static func applyAliasingReduction(_ spectrum: inout [Float]) {
    // Apply butterfly between each pair of adjacent subbands (31 boundaries)
    for sb in 0..<31 {
      let sbEnd = sb * 18 + 17
      let sbNextStart = (sb + 1) * 18

      // Gather 8 upper values (reversed order) and 8 lower values
      let upper = SIMD8<Float>(
        spectrum[sbEnd], spectrum[sbEnd - 1], spectrum[sbEnd - 2], spectrum[sbEnd - 3],
        spectrum[sbEnd - 4], spectrum[sbEnd - 5], spectrum[sbEnd - 6], spectrum[sbEnd - 7]
      )
      let lower = SIMD8<Float>(
        spectrum[sbNextStart], spectrum[sbNextStart + 1], spectrum[sbNextStart + 2], spectrum[sbNextStart + 3],
        spectrum[sbNextStart + 4], spectrum[sbNextStart + 5], spectrum[sbNextStart + 6], spectrum[sbNextStart + 7]
      )

      // Vectorized butterfly: bu = lower * ca + upper * cs, bd = lower * cs - upper * ca
      let newUpper = lower * aliasingCA + upper * aliasingCS
      let newLower = lower * aliasingCS - upper * aliasingCA

      // Scatter results back
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

  /// MDCT for long blocks: 36 input → 18 output
  /// Uses ISO 11172-3 formula with pre-computed cosine matrix and vDSP.
  private static func mdctLong(samples: [Float], window: [Float]) -> [Float] {
    let n = 36
    let halfN = 18
    let normalization: Float = 9.0  // N/4 for long blocks

    // Apply window using vDSP
    var windowed = [Float](repeating: 0, count: n)
    vDSP_vmul(samples, 1, window, 1, &windowed, 1, vDSP_Length(n))

    // MDCT using pre-computed matrix and vDSP dot products
    let matrix = getLongMDCTMatrix()
    var output = [Float](repeating: 0, count: halfN)
    for m in 0..<halfN {
      var result: Float = 0
      vDSP_dotpr(windowed, 1, matrix[m], 1, &result, vDSP_Length(n))
      output[m] = result / normalization
    }

    return output
  }

  /// MDCT for short blocks: 3 windows of 12 samples each → 18 total coefficients
  /// Uses ISO 11172-3 formula with pre-computed cosine matrix and vDSP.
  private static func mdctShort(samples: [Float]) -> [Float] {
    let shortWindow = getShortWindow()
    var output = [Float](repeating: 0, count: 18)
    let n = 12
    let normalization: Float = 3.0  // N/4 for short blocks

    let matrix = getShortMDCTMatrix()

    // Three short windows
    for w in 0..<3 {
      // Extract 12 samples for this window and apply window
      let offset = w * 6 + 6
      var windowSamples = [Float](repeating: 0, count: n)
      for i in 0..<n {
        let idx = offset + i
        windowSamples[i] = idx < samples.count ? samples[idx] * shortWindow[i] : 0
      }

      // MDCT using pre-computed matrix and vDSP
      for m in 0..<6 {
        var result: Float = 0
        vDSP_dotpr(windowSamples, 1, matrix[m], 1, &result, vDSP_Length(n))
        output[w + m * 3] = result / normalization
      }
    }

    return output
  }
}

private struct HuffmanEncoder {
  /// Encodes values using table 1 (correct ISO codes) and returns the number of bits written.
  /// Table 1 handles values 0-1 only but has verified correct Huffman codes.
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

  /// Writes a pair of values using table 1 (ISO 11172-3 Table B.7).
  /// Table 1 codes: (0,0)=1, (0,1)=001, (1,0)=011, (1,1)=010
  private func writePairTable1(x: Int, y: Int, writer: inout BitstreamWriter) {
    let absX = min(abs(x), 1)
    let absY = min(abs(y), 1)

    let code = MP3Tables.table1.table[absX][absY]
    writer.write(bits: code.bits, count: code.length)

    // Sign bits follow the Huffman code (only if value is non-zero)
    if absX != 0 {
      writer.write(bits: x < 0 ? 1 : 0, count: 1)
    }
    if absY != 0 {
      writer.write(bits: y < 0 ? 1 : 0, count: 1)
    }
  }

  /// Encodes values using table 15 (forced) and returns the number of bits written.
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

  /// Writes a pair of values using table 15 specifically.
  /// Table 15 is a 16x16 table for values 0-15 with NO linbits (linbits=0).
  private func writePairTable15(x: Int, y: Int, writer: inout BitstreamWriter) {
    let absX = min(abs(x), 15)
    let absY = min(abs(y), 15)

    // Write the Huffman code
    let code = MP3Tables.table15.table[absX][absY]
    writer.write(bits: code.bits, count: code.length)

    // Sign bits follow (only if value is non-zero)
    if absX != 0 {
      writer.write(bits: x < 0 ? 1 : 0, count: 1)
    }
    if absY != 0 {
      writer.write(bits: y < 0 ? 1 : 0, count: 1)
    }
  }

  /// Encodes values and returns the number of bits written (dynamic table selection).
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

  private func writePair(x: Int, y: Int, writer: inout BitstreamWriter) {
    let absX = min(abs(x), MP3Tables.table15.maxValue)
    let absY = min(abs(y), MP3Tables.table15.maxValue)

    // Select appropriate table based on maximum value
    let table = self.selectTable(absX: absX, absY: absY)

    // Bounds check before table lookup
    guard absX < table.table.count && absY < table.table[absX].count else {
      // Fallback: use table15 which handles all values 0-15
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

    // Sign bits follow the Huffman code
    if absX != 0 {
      writer.write(bits: x < 0 ? 1 : 0, count: 1)
    }
    if absY != 0 {
      writer.write(bits: y < 0 ? 1 : 0, count: 1)
    }
  }

  private func selectTable(absX: Int, absY: Int) -> MP3Tables.HuffmanTable {
    let maxValue = max(absX, absY)

    // Select smallest table that can encode these values
    // Tables: 1 (0-1), 2/3 (0-2), 5/6 (0-3), 7/8/9 (0-5), 10 (0-7), 13/15 (0-15)
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

private enum ScaleFactorBands {
  /// Scale factor band widths from ISO 11172-3 Table B.8
  /// These define the frequency groupings that approximate human critical bands.

  /// Long block scalefactor band widths for 44100 Hz (21 bands)
  static let longBands44100 = [4, 4, 4, 4, 4, 4, 6, 6, 8, 8, 10, 12, 16, 20, 24, 28, 34, 42, 50, 54, 76]

  /// Long block scalefactor band widths for 48000 Hz (21 bands)
  static let longBands48000 = [4, 4, 4, 4, 4, 4, 6, 6, 6, 8, 10, 12, 16, 18, 22, 28, 34, 40, 46, 54, 54]

  /// Long block scalefactor band widths for 32000 Hz (21 bands)
  static let longBands32000 = [4, 4, 4, 4, 4, 4, 6, 6, 8, 10, 12, 16, 20, 24, 30, 38, 46, 56, 68, 84, 102]

  /// Short block scalefactor band widths for 44100 Hz (12 bands, applied 3 times)
  static let shortBands44100 = [4, 4, 4, 4, 6, 8, 10, 12, 14, 18, 22, 30]

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

      // Scale factor represents the maximum in this band
      // Used for both normalization and to compute scalefac_compress
      let scale = max(peak, 0.0001)
      scalefactors.append(scale)

      // Normalize spectrum within this band
      for index in start..<end {
        scaledSpectrum[index] = spectrum[index] / scale
      }

      cursor = end
    }

    // Expand scalefactors to per-coefficient for compatibility
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

  /// Returns the number of scalefactor bands for a given sample rate and block type.
  static func bandCount(for sampleRate: Int, isShort: Bool) -> Int {
    if isShort {
      return 12
    }
    return bandTable(for: sampleRate).count
  }
}

private enum FrameAnalysis {
  /// Calculates the average energy (mean square) of samples using vDSP.
  static func energy(samples: [Float]) -> Float {
    guard !samples.isEmpty else { return 0 }
    var sumOfSquares: Float = 0
    vDSP_svesq(samples, 1, &sumOfSquares, vDSP_Length(samples.count))
    return sumOfSquares / Float(samples.count)
  }
}

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

private enum BlockType: Int {
  case long = 0
  case short = 2
  case mixed = 1
}

private struct TransientResult {
  let blockType: BlockType
  let subblockGain: [Int]
}

private enum TransientDetector {
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

private enum PsychoacousticModel {
  /// Calculates masking thresholds using vDSP for energy calculations.
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

      // Calculate energy using vDSP
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

private enum ScaleFactorCompression {
  /// Compresses scalefactors using vDSP for statistical calculations.
  static func compress(scalefactors: [Float]) -> Int {
    guard !scalefactors.isEmpty else { return 0 }
    let count = vDSP_Length(scalefactors.count)

    // Calculate mean using vDSP
    var mean: Float = 0
    vDSP_meanv(scalefactors, 1, &mean, count)

    // Calculate variance: mean of squared differences from mean
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

private enum PreEmphasis {
  /// Determines if pre-emphasis should be enabled using vDSP for energy calculations.
  static func shouldEnable(spectral: [Float], scalefactors: [Float]) -> Bool {
    guard !spectral.isEmpty else { return false }
    let highStart = max(spectral.count * 3 / 4, 0)
    let highCount = spectral.count - highStart

    // Calculate high frequency energy using vDSP
    var highEnergy: Float = 0
    if highCount > 0 {
      spectral.withUnsafeBufferPointer { ptr in
        vDSP_svesq(ptr.baseAddress! + highStart, 1, &highEnergy, vDSP_Length(highCount))
      }
    }

    // Calculate low frequency energy using vDSP
    var lowEnergy: Float = 0
    if highStart > 0 {
      vDSP_svesq(spectral, 1, &lowEnergy, vDSP_Length(highStart))
    }

    // Calculate scalefactor average using vDSP
    var scalefactorSum: Float = 0
    if !scalefactors.isEmpty {
      vDSP_sve(scalefactors, 1, &scalefactorSum, vDSP_Length(scalefactors.count))
    }
    let scalefactorAverage = scalefactorSum / Float(max(scalefactors.count, 1))

    return highEnergy > lowEnergy * 1.5 && scalefactorAverage > 0.5
  }
}

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

private struct BitReservoir {
  // Simplified: no cross-frame buffering for maximum compatibility
  // Each frame is self-contained with main_data_begin = 0
  private(set) var mainDataBegin: Int = 0

  mutating func consume(with mainData: Data) -> Data {
    // Simply return the main data as-is
    // main_data_begin stays 0 (data starts immediately after side info)
    return mainData
  }
}

private enum StereoDecision {
  case raw([Float], [Float])
  case midSide([Float], [Float])

  /// Creates stereo decision using vDSP for mid/side calculation.
  static func make(mode: MP3EncoderOptions.Mode, left: [Float], right: [Float]) -> StereoDecision {
    guard mode == .jointStereo, left.count == right.count else {
      return .raw(left, right)
    }

    let count = left.count

    // mid = (left + right) * 0.5
    var mid = [Float](repeating: 0, count: count)
    vDSP_vadd(left, 1, right, 1, &mid, 1, vDSP_Length(count))
    var halfScalar: Float = 0.5
    vDSP_vsmul(mid, 1, &halfScalar, &mid, 1, vDSP_Length(count))

    // side = (left - right) * 0.5
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

  func samples(for channel: Int) -> [Float] {
    switch self {
    case .raw(let left, let right):
      return channel == 0 ? left : right
    case .midSide(let mid, let side):
      return channel == 0 ? mid : side
    }
  }
}

// MARK: - UInt32 Big-Endian Extension

private extension UInt32 {
  /// Returns the value as 4 bytes in big-endian order.
  var bigEndianBytes: [UInt8] {
    return [
      UInt8((self >> 24) & 0xFF),
      UInt8((self >> 16) & 0xFF),
      UInt8((self >> 8) & 0xFF),
      UInt8(self & 0xFF)
    ]
  }
}

private enum CRC16 {
  /// Pre-computed CRC16 lookup table for MPEG polynomial 0x8005.
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

  /// Computes CRC16 using lookup table.
  static func mpeg(data: Data) -> UInt16 {
    var crc: UInt16 = 0xFFFF
    for byte in data {
      let index = Int((crc >> 8) ^ UInt16(byte))
      crc = (crc << 8) ^ table[index]
    }
    return crc
  }
}

private struct BitstreamWriter {
  private(set) var data = Data()
  private var buffer: UInt32 = 0
  private var bitsInBuffer: Int = 0

  /// Total bits written so far.
  var bitCount: Int {
    data.count * 8 + bitsInBuffer
  }

  /// Writes multiple bits at once, more efficient than bit-by-bit.
  mutating func write(bits: Int, count: Int) {
    guard count > 0 && count <= 24 else {
      // Fall back to bit-by-bit for edge cases
      for i in (0..<count).reversed() {
        writeBit((bits >> i) & 1)
      }
      return
    }

    // Add bits to buffer
    buffer = (buffer << count) | UInt32(bits & ((1 << count) - 1))
    bitsInBuffer += count

    // Flush complete bytes
    while bitsInBuffer >= 8 {
      bitsInBuffer -= 8
      let byte = UInt8((buffer >> bitsInBuffer) & 0xFF)
      data.append(byte)
    }

    // Mask buffer to keep only remaining bits
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

private enum MP3Tables {
  struct HuffmanTable {
    let maxValue: Int
    let table: [[(length: Int, bits: Int)]]
  }

  // ==========================================================================
  // ISO 11172-3 Table B.7 - Huffman code tables for MPEG-1 Layer III
  // Format: (length, code) - length in bits, code is the Huffman codeword
  // ==========================================================================

  /// Table 1: 2x2 for values 0-1
  static let table1 = HuffmanTable(
    maxValue: 1,
    table: [
      [(1, 1), (3, 1)],   // (0,0), (0,1)
      [(2, 1), (3, 0)]    // (1,0), (1,1)
    ]
  )

  /// Table 2: 3x3 for values 0-2
  static let table2 = HuffmanTable(
    maxValue: 2,
    table: [
      [(1, 1), (3, 2), (6, 1)],
      [(3, 3), (3, 1), (5, 1)],
      [(5, 3), (5, 2), (6, 0)]
    ]
  )

  /// Table 3: 3x3 for values 0-2
  static let table3 = HuffmanTable(
    maxValue: 2,
    table: [
      [(2, 3), (2, 2), (6, 1)],
      [(3, 1), (2, 1), (5, 1)],
      [(5, 3), (5, 2), (6, 0)]
    ]
  )

  /// Table 5: 4x4 for values 0-3
  static let table5 = HuffmanTable(
    maxValue: 3,
    table: [
      [(1, 1), (3, 2), (6, 6), (7, 5)],
      [(3, 3), (3, 1), (6, 4), (7, 4)],
      [(6, 7), (6, 5), (7, 7), (8, 1)],
      [(7, 6), (6, 1), (7, 1), (8, 0)]
    ]
  )

  /// Table 6: 4x4 for values 0-3
  static let table6 = HuffmanTable(
    maxValue: 3,
    table: [
      [(3, 7), (3, 3), (5, 5), (7, 1)],
      [(3, 6), (2, 2), (4, 3), (5, 2)],
      [(4, 5), (4, 4), (5, 4), (6, 1)],
      [(6, 3), (5, 3), (6, 2), (7, 0)]
    ]
  )

  /// Table 7: 6x6 for values 0-5
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

  /// Table 8: 6x6 for values 0-5
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

  /// Table 9: 6x6 for values 0-5
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

  /// Table 10: 8x8 for values 0-7
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

  /// Table 13: 16x16 for values 0-15
  static let table13 = HuffmanTable(
    maxValue: 15,
    table: Self.buildTable13()
  )

  /// Table 15: 16x16 for values 0-15
  static let table15 = HuffmanTable(
    maxValue: 15,
    table: Self.buildTable15()
  )

  /// Builds table 13: 16x16 for values 0-15
  /// Codes from ISO 11172-3 Table B.7
  private static func buildTable13() -> [[(length: Int, bits: Int)]] {
    // Huffman code lengths (linearized 16x16)
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

    // Huffman codes (linearized 16x16)
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

  /// Builds table 15: 16x16 for values 0-15.
  private static func buildTable15() -> [[(length: Int, bits: Int)]] {
    // Huffman code lengths (linearized 16x16)
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

    // Huffman codes (linearized 16x16)
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
