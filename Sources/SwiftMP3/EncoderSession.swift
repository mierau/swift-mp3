// SwiftMP3
// EncoderSession.swift

import Foundation

/// Mutable encoding state for synchronous MP3 encoding.
///
/// Holds all per-session state (PCM buffer, filterbank and MDCT state, bit reservoir, frame
/// statistics). Created via ``MP3Encoder/newSession()``.
///
/// Output is a sequence of complete MP3 frames. Because a frame's main data can start in
/// earlier frames (the bit reservoir) and block switching needs one granule of lookahead,
/// output lags input by a few frames; ``flush()`` emits the remainder.
public struct EncoderSession {
  private let options: MP3EncoderOptions
  private let format: StreamFormat
  private let channels: Int
  /// `bitrate_index` of the configured bitrate: the CBR rate, or the VBR average.
  private let targetBitrateIndex: Int

  // Input
  /// Converts the input to the MP3's sample rate, when they differ.
  private var resampler: Resampler?
  /// Input at the MP3's sample rate, interleaved, not yet analyzed.
  private var pcm: [Float] = []
  private var inputSamples = 0
  private var finished = false

  // Analysis
  private var analyzers: [PolyphaseAnalyzer]
  /// Subband samples for granules g-1, g, and g+1 per channel (`[channel][slot * 32 + sb]`).
  private var subbandPrevious: [[Float]]
  private var subbandCurrent: [[Float]]
  private var subbandNext: [[Float]]
  private var subbandGranules = 0
  private var transients: [TransientDetector]
  private var wantsShortCurrent: [Bool]
  private var lateAttack: [Bool]
  private var lastBlockType: [BlockType]
  private let shortOrder: [Int]
  private let lowpassHz: Int
  private let lowpassLong: Int
  private let lowpassShort: Int

  // Granules awaiting a frame: [granule][channel]
  private var spectra: [[[Float]]]
  private var blockTypes: [[BlockType]]
  private var mdctGranules = 0

  // Coding
  private var psychoacoustics: PsychoacousticModel
  private var quantizer: Quantizer
  private var previousAllowedNoise: [[Float]]
  private var paddingRemainder = 0
  private var bitrate = BitrateController()
  /// Running average of perceptual entropy per granule and channel.
  private var averageEntropy: Float = 0

  // Bitstream
  private var reservoirBits = 0
  private var pendingFrames: [(header: [UInt8], slotSize: Int)] = []
  private var pendingFrameIndex = 0
  private var mainData: [UInt8] = []
  private var mainDataOffset = 0

  // Statistics
  private var frameCount = 0
  private var totalBytes = 0
  private var frameSizes: [UInt16] = []
  /// CRC of the audio frames, for the LAME tag.
  private var musicCRC: UInt16 = 0
  /// Loudness and peak of the input, for the LAME tag's ReplayGain fields.
  private var loudness: LoudnessMeter

  /// The sample rate of the input, in Hz.
  public var sampleRate: Int { options.sampleRate }

  /// The sample rate of the encoded MP3, in Hz.
  public var outputSampleRate: Int { format.sampleRate }

  /// The number of interleaved channels in the input.
  public var channelCount: Int { channels }

  /// The number of frames encoded so far.
  public var encodedFrameCount: Int { frameCount }

  /// The total bytes of encoded audio data so far (excluding the Xing header).
  public var encodedByteCount: Int { totalBytes }

  /// The ReplayGain 2.0 track gain of the input so far, in dB: the change in level that
  /// brings it to -18 LUFS (ITU-R BS.1770). `nil` until there's 400 ms of audible input.
  /// ``generateXingHeader()`` records it in the LAME tag.
  public var replayGain: Double? { loudness.trackGain }

  /// The largest absolute sample value of the input so far (1.0 is full scale).
  public var peakAmplitude: Float { loudness.peak }

  /// Creates a new encoding state for options that ``MP3Encoder/init(options:)`` validated.
  init(options: MP3EncoderOptions) {
    self.options = options
    channels = options.channelCount
    format = StreamFormat(sampleRate: options.encodedSampleRate, channels: channels)!
    if options.sampleRate != format.sampleRate {
      resampler = Resampler(inputRate: options.sampleRate, outputRate: format.sampleRate, channels: channels)
    }
    targetBitrateIndex = format.bitrateIndex(for: options.bitrateKbps)!

    let granule = [Float](repeating: 0, count: MPEG.samplesPerGranule)
    analyzers = Array(repeating: PolyphaseAnalyzer(), count: channels)
    subbandPrevious = Array(repeating: granule, count: channels)
    subbandCurrent = Array(repeating: granule, count: channels)
    subbandNext = Array(repeating: granule, count: channels)
    transients = Array(repeating: TransientDetector(), count: channels)
    wantsShortCurrent = Array(repeating: false, count: channels)
    lateAttack = Array(repeating: false, count: channels)
    lastBlockType = Array(repeating: .normal, count: channels)
    spectra = Array(repeating: Array(repeating: granule, count: channels), count: 2)
    blockTypes = Array(repeating: Array(repeating: .normal, count: channels), count: 2)
    previousAllowedNoise = Array(repeating: [], count: channels)

    // Map short-block MDCT output (window, line) to scalefactor band order.
    let shortBands = format.shortBands
    var order = [Int](repeating: 0, count: 3 * 192)
    for band in 0..<13 {
      let width = shortBands[band + 1] - shortBands[band]
      for w in 0..<3 {
        for line in shortBands[band]..<shortBands[band + 1] {
          order[w * 192 + line] = shortBands[band] * 3 + w * width + (line - shortBands[band])
        }
      }
    }
    shortOrder = order

    let sampleRate = format.sampleRate
    let cutoff = Self.lowpassFrequency(kbps: options.bitrateKbps, channels: channels, sampleRate: sampleRate)
    lowpassHz = Int(cutoff.rounded())
    lowpassLong = min(576, Int((cutoff / (Double(sampleRate) / 2) * 576).rounded()))
    lowpassShort = min(192, Int((cutoff / (Double(sampleRate) / 2) * 192).rounded()))

    loudness = LoudnessMeter(sampleRate: format.sampleRate, channels: channels)
    psychoacoustics = PsychoacousticModel(format: format, quality: options.quality)
    quantizer = Quantizer(format: format, quality: options.quality)
    frameSizes.reserveCapacity(4096)
  }

  // MARK: - Public API

  /// Appends interleaved PCM samples to the internal buffer and encodes any complete frames.
  ///
  /// Samples should be interleaved for stereo (L, R, L, R, ...) and normalized to [-1.0, 1.0].
  /// Any leftover samples that don't fill a complete granule are buffered for the next call.
  ///
  /// - Parameter samples: Interleaved PCM float samples.
  /// - Returns: Encoded MP3 data for any complete frames, or empty data if none were completed.
  public mutating func encode(samples: [Float]) -> Data {
    samples.withUnsafeBufferPointer { encode(samples: $0) }
  }

  /// Appends interleaved PCM samples from a buffer and encodes any complete frames.
  public mutating func encode(samples: UnsafeBufferPointer<Float>) -> Data {
    append(count: samples.count) { $0.append(contentsOf: samples) }
  }

  /// Appends interleaved 16-bit PCM samples and encodes any complete frames.
  public mutating func encode(samples: [Int16]) -> Data {
    samples.withUnsafeBufferPointer { encode(samples: $0) }
  }

  /// Appends interleaved 16-bit PCM samples from a buffer and encodes any complete frames.
  public mutating func encode(samples: UnsafeBufferPointer<Int16>) -> Data {
    append(count: samples.count) { pcm in
      pcm.append(contentsOf: samples.lazy.map { Float($0) * (1 / 32_768) })
    }
  }

  /// Appends non-interleaved PCM samples, one array per channel, and encodes any complete frames.
  ///
  /// A mono encoder averages any number of channels; a stereo encoder plays one channel
  /// in both.
  ///
  /// - Parameter channels: One array per channel (left then right for stereo), all the
  ///   same length, normalized to [-1.0, 1.0]. A stereo encoder takes one or two.
  public mutating func encode(planar channels: [[Float]]) -> Data {
    precondition(!channels.isEmpty, "Expected at least one channel")
    precondition(self.channels == 1 || channels.count <= 2, "Expected 1 or 2 channels, got \(channels.count)")
    let length = channels.first?.count ?? 0
    precondition(channels.allSatisfy { $0.count == length }, "Channels must have the same length")
    if self.channels == 1 {
      guard channels.count > 1 else { return encode(samples: channels[0]) }
      return append(count: length) { pcm in
        let scale = 1 / Float(channels.count)
        pcm.reserveCapacity(pcm.count + length)
        for i in 0..<length {
          var sum: Float = 0
          for channel in channels { sum += channel[i] }
          pcm.append(sum * scale)
        }
      }
    }
    return append(count: length * 2) { pcm in
      let left = channels[0]
      let right = channels[channels.count - 1]
      pcm.reserveCapacity(pcm.count + length * 2)
      for i in 0..<length {
        pcm.append(left[i])
        pcm.append(right[i])
      }
    }
  }

  /// Adds `count` samples through `body` and encodes whatever they complete.
  private mutating func append(count: Int, _ body: (inout [Float]) -> Void) -> Data {
    guard !finished else { return Data() }
    let before = pcm.count
    if resampler == nil {
      body(&pcm)
    } else {
      var input = [Float]()
      input.reserveCapacity(count)
      body(&input)
      input.withUnsafeBufferPointer { resampler!.process($0, into: &pcm) }
    }
    inputSamples += pcm.count - before
    loudness.add(pcm[before...])
    var output: [UInt8] = []
    processBufferedInput(into: &output)
    return Data(output)
  }

  /// Encodes any remaining buffered samples and emits all pending frames.
  ///
  /// Call this once after all samples have been passed to ``encode(samples:)``. The audio
  /// is padded with silence so that every input sample is fully represented.
  ///
  /// - Returns: Encoded MP3 data for any remaining frames, or empty data if nothing was buffered.
  public mutating func flush() -> Data {
    guard !finished else { return Data() }
    finished = true
    var output: [UInt8] = []
    if resampler != nil {
      let before = pcm.count
      resampler!.finish(into: &pcm)
      inputSamples += pcm.count - before
      loudness.add(pcm[before...])
    }
    guard inputSamples > 0 else { return Data() }

    // A trailing partial sample frame is completed with silence.
    if pcm.count % channels != 0 {
      pcm.append(contentsOf: repeatElement(0, count: channels - pcm.count % channels))
    }
    // An input sample affects subband granules up to (n + 480) / 576; the MDCT needs one
    // more granule, and block-type lookahead one more after that.
    let lastSubbandGranule = (inputFrameCount - 1 + 480) / MPEG.samplesPerGranule
    let frames = (lastSubbandGranule + 2 + format.granules - 1) / format.granules
    let neededGranules = frames * format.granules + 1
    let missing = neededGranules * MPEG.samplesPerGranule - (subbandGranules * MPEG.samplesPerGranule + pcm.count / channels)
    if missing > 0 {
      pcm.append(contentsOf: repeatElement(0, count: missing * channels))
    }
    processBufferedInput(into: &output)

    // Fill the remaining main data slots.
    let outstanding = pendingFrames[pendingFrameIndex...].reduce(0) { $0 + $1.slotSize }
    let available = mainData.count - mainDataOffset
    if outstanding > available {
      mainData.append(contentsOf: repeatElement(0, count: outstanding - available))
    }
    emitReadyFrames(into: &output)
    return Data(output)
  }

  /// Number of sample frames (one sample per channel) received so far, at the MP3's rate.
  private var inputFrameCount: Int { (inputSamples + channels - 1) / channels }

  /// Generates the ID3v2.3 tag data for the configured metadata.
  ///
  /// Returns empty `Data` if no ID3 tag is configured or all fields are empty.
  public func generateID3Tag() -> Data {
    guard let tag = options.id3Tag else { return Data() }
    return ID3TagWriter.build(tag: tag)
  }

  // MARK: - Analysis

  private mutating func processBufferedInput(into output: inout [UInt8]) {
    let granuleSamples = MPEG.samplesPerGranule * channels
    var offset = 0
    while pcm.count - offset >= granuleSamples {
      let start = offset
      pcm.withUnsafeBufferPointer { buffer in
        analyzeGranule(buffer.baseAddress! + start)
      }
      offset += granuleSamples
      encodeReadyFrame(into: &output)
    }
    if offset > 0 {
      pcm.removeFirst(offset)
    }
  }

  /// Runs the filterbank on one granule of input and transforms the previous granule.
  private mutating func analyzeGranule(_ input: UnsafePointer<Float>) {
    for ch in 0..<channels {
      subbandNext[ch].withUnsafeMutableBufferPointer { out in
        analyzers[ch].analyze(input + ch, stride: channels, scale: 1, into: out.baseAddress!)
      }
    }

    // Decide which granules need short blocks. Granule g's short windows cover the last
    // third of subband granule g-1 and the first two thirds of granule g.
    var wantsShortNext = [Bool](repeating: false, count: channels)
    for ch in 0..<channels {
      let attacks = subbandNext[ch].withUnsafeBufferPointer { transients[ch].analyze($0.baseAddress!) }
      wantsShortNext[ch] = lateAttack[ch] || attacks.early
      lateAttack[ch] = attacks.late
    }
    if options.mode == .jointStereo {
      let any = wantsShortNext.contains(true)
      wantsShortNext = Array(repeating: any, count: channels)
    }

    if subbandGranules > 0 {
      let slot = mdctGranules & 1
      for ch in 0..<channels {
        let type = Self.blockType(
          previous: lastBlockType[ch],
          wantsShort: wantsShortCurrent[ch],
          nextWantsShort: wantsShortNext[ch]
        )
        lastBlockType[ch] = type
        blockTypes[slot][ch] = type
        subbandPrevious[ch].withUnsafeBufferPointer { prev in
          subbandCurrent[ch].withUnsafeBufferPointer { cur in
            shortOrder.withUnsafeBufferPointer { order in
              spectra[slot][ch].withUnsafeMutableBufferPointer { out in
                MDCT.transform(
                  previous: prev.baseAddress!,
                  current: cur.baseAddress!,
                  blockType: type,
                  shortOrder: order.baseAddress!,
                  output: out.baseAddress!
                )
              }
            }
          }
        }
      }
      for ch in 0..<channels {
        var spectrum = [Float]()
        swap(&spectrum, &spectra[slot][ch])
        spectrum.withUnsafeMutableBufferPointer { applyLowpass($0.baseAddress!, blockType: blockTypes[slot][ch]) }
        swap(&spectrum, &spectra[slot][ch])
      }
      mdctGranules += 1
    }

    swap(&subbandPrevious, &subbandCurrent)
    swap(&subbandCurrent, &subbandNext)
    wantsShortCurrent = wantsShortNext
    subbandGranules += 1
  }

  /// Chooses a window shape that keeps the start/short/stop sequence valid.
  private static func blockType(previous: BlockType, wantsShort: Bool, nextWantsShort: Bool) -> BlockType {
    if wantsShort { return .short }
    if previous == .short {
      // There is no window that is short on both sides, so stay short.
      return nextWantsShort ? .short : .stop
    }
    return nextWantsShort ? .start : .normal
  }

  private func applyLowpass(_ spectrum: UnsafeMutablePointer<Float>, blockType: BlockType) {
    if blockType == .short {
      guard lowpassShort < 192 else { return }
      for w in 0..<3 {
        for line in lowpassShort..<192 {
          spectrum[shortOrder[w * 192 + line]] = 0
        }
      }
    } else if lowpassLong < 576 {
      for line in lowpassLong..<576 {
        spectrum[line] = 0
      }
    }
  }

  /// Lowpass frequency for a bitrate: 10% above LAME's bandwidth table for stereo, which
  /// scored better on ViSQOL at every bitrate tested (96-320 kbps).
  static func lowpassFrequency(kbps: Int, channels: Int, sampleRate: Int) -> Double {
    let table: [(Int, Double)] = [
      (8, 2000), (16, 3700), (24, 3900), (32, 5500), (40, 7000), (48, 7500), (56, 10_000),
      (64, 11_000), (80, 13_500), (96, 15_100), (112, 15_600), (128, 17_000), (160, 17_500),
      (192, 18_600), (224, 19_400), (256, 19_700), (320, 20_500),
    ]
    // A mono stream gets the bandwidth of a stereo stream with twice the bitrate.
    let equivalent = Double(channels == 1 ? kbps * 2 : kbps)
    var cutoff = table.last!.1
    if equivalent <= Double(table[0].0) {
      cutoff = table[0].1
    } else {
      for i in 1..<table.count where equivalent <= Double(table[i].0) {
        let (k0, f0) = table[i - 1]
        let (k1, f1) = table[i]
        cutoff = f0 + (f1 - f0) * (equivalent - Double(k0)) / Double(k1 - k0)
        break
      }
    }
    return min(cutoff * 1.1, Double(sampleRate) / 2)
  }

  // MARK: - Frame encoding

  private mutating func encodeReadyFrame(into output: inout [UInt8]) {
    guard mdctGranules >= format.granules else { return }
    mdctGranules = 0
    encodeFrame()
    emitReadyFrames(into: &output)
  }

  private mutating func encodeFrame() {
    // Psychoacoustic analysis of the coded channels.
    let granules = format.granules
    var masking = [[MaskingResult]]()
    for gr in 0..<granules {
      var row = [MaskingResult]()
      for ch in 0..<channels {
        row.append(analyze(granule: gr, channel: ch))
      }
      masking.append(row)
    }

    var useMidSide = false
    var sideRatio = [Float](repeating: 0.5, count: granules)
    if options.mode == .jointStereo {
      (useMidSide, sideRatio) = decideMidSide(&masking)
    }

    // Bitrate and frame size.
    let totalPE = masking.joined().reduce(Float(0)) { $0 + $1.perceptualEntropy }
    let sampleRate = format.sampleRate
    let crc = options.crcProtected
    var bitrateIndex = targetBitrateIndex
    // VBR: quantize every granule just finely enough to mask its noise, then pick the
    // smallest bitrate that holds the result, capped by the average-bitrate target.
    var transparent: [[GranuleInfo]]?
    if options.vbr {
      var rows = [[GranuleInfo]]()
      var demand = 0
      for gr in 0..<granules {
        var row = [GranuleInfo]()
        for ch in 0..<channels {
          let info = spectra[gr][ch].withUnsafeBufferPointer { spectrum in
            quantizer.quantizeTransparent(
              spectrum: spectrum.baseAddress!,
              blockType: blockTypes[gr][ch],
              allowedNoise: masking[gr][ch].allowedNoise
            )
          }
          if let info {
            row.append(info)
            demand += info.part23Length
          } else {
            demand = .max / 2
          }
        }
        rows.append(row)
      }
      let ceiling = bitrate.choose(target: targetBitrateIndex, format: format, perceptualEntropy: totalPE)
      bitrateIndex = 1
      while bitrateIndex < ceiling {
        let slotBits = format.mainDataSize(frameLength: format.frameLength(bitrateIndex: bitrateIndex), crc: crc) * 8
        if demand <= slotBits + reservoirBits { break }
        bitrateIndex += 1
      }
      if rows.allSatisfy({ $0.count == channels }) {
        transparent = rows
      }
    }
    paddingRemainder += format.scaledFrameLength(bitrateIndex: bitrateIndex) % sampleRate
    let padding = paddingRemainder >= sampleRate
    if padding { paddingRemainder -= sampleRate }
    let frameLength = format.frameLength(bitrateIndex: bitrateIndex, padding: padding)
    let slotSize = format.mainDataSize(frameLength: frameLength, crc: crc)
    let slotBits = slotSize * 8
    let maxReservoir = max(0, min(format.maxMainDataBegin * 8, 7680 - frameLength * 8)) & ~7

    let mainDataBegin = reservoirBits / 8
    var reservoir = reservoirBits
    let meanBits = slotBits / granules

    for value in masking.joined().map(\.perceptualEntropy) {
      averageEntropy = averageEntropy == 0 ? max(value, 1) : averageEntropy + 0.02 * (value - averageEntropy)
    }

    var coded = [[GranuleInfo]]()
    var usedBits = 0
    if let transparent, transparent.joined().reduce(0, { $0 + $1.part23Length }) <= reservoirBits + slotBits {
      coded = transparent
      usedBits = transparent.joined().reduce(0) { $0 + $1.part23Length }
    } else {
      for gr in 0..<granules {
        let pe = masking[gr].map(\.perceptualEntropy)
        let targets = Self.allocateBits(
          meanBits: meanBits,
          reservoir: reservoir,
          maxReservoir: maxReservoir,
          perceptualEntropy: pe,
          averageEntropy: max(averageEntropy, 1),
          sideRatio: useMidSide ? sideRatio[gr] : nil,
          constantBitrate: !options.vbr
        )
        var row = [GranuleInfo]()
        for ch in 0..<channels {
          let info = spectra[gr][ch].withUnsafeBufferPointer { spectrum in
            quantizer.quantize(
              spectrum: spectrum.baseAddress!,
              blockType: blockTypes[gr][ch],
              allowedNoise: masking[gr][ch].allowedNoise,
              maxBits: targets[ch]
            )
          }
          reservoir -= info.part23Length
          usedBits += info.part23Length
          row.append(info)
        }
        reservoir += meanBits
        coded.append(row)
      }
    }

    // Share scalefactors between granules where they repeat.
    var scfsi = Array(repeating: [0, 0, 0, 0], count: channels)
    if format.isMPEG1 {
      for ch in 0..<channels {
        let saved = applySCFSI(&coded[1][ch], first: coded[0][ch], scfsi: &scfsi[ch])
        usedBits -= saved
      }
    }

    // Main data.
    var writer = BitWriter(capacity: (usedBits + 7) / 8 + 8)
    let regions = quantizer.regions
    for gr in 0..<granules {
      for ch in 0..<channels {
        let info = coded[gr][ch]
        let start = writer.bitCount
        writeScalefactors(info, granule: gr, scfsi: scfsi[ch], into: &writer)
        spectra[gr][ch].withUnsafeBufferPointer { spectrum in
          HuffmanWriter.write(info, spectrum: spectrum.baseAddress!, regions: regions, into: &writer)
        }
        assert(writer.bitCount - start == info.part23Length, "part2_3_length mismatch")
      }
    }
    writer.alignToByte()
    var data = writer.bytes

    // Reservoir bookkeeping: anything beyond the reservoir limit becomes stuffing.
    var nextReservoir = reservoirBits + slotBits - data.count * 8
    precondition(nextReservoir >= 0, "main data overflowed the frame and reservoir")
    if nextReservoir > maxReservoir {
      let stuffing = (nextReservoir - maxReservoir) / 8
      data.append(contentsOf: repeatElement(0, count: stuffing))
      nextReservoir -= stuffing * 8
    }
    reservoirBits = nextReservoir

    let header = frameHeader(
      bitrateIndex: bitrateIndex,
      padding: padding,
      midSide: useMidSide,
      granules: coded,
      scfsi: scfsi,
      mainDataBegin: mainDataBegin
    )
    pendingFrames.append((header, slotSize))
    mainData.append(contentsOf: data)
    if options.vbr {
      bitrate.record(frameBytes: frameLength)
    }
  }

  private mutating func analyze(granule gr: Int, channel ch: Int) -> MaskingResult {
    spectra[gr][ch].withUnsafeBufferPointer { spectrum in
      if blockTypes[gr][ch] == .short {
        return psychoacoustics.analyzeShort(spectrum.baseAddress!, previous: &previousAllowedNoise[ch])
      }
      return psychoacoustics.analyzeLong(spectrum.baseAddress!, previous: &previousAllowedNoise[ch])
    }
  }

  /// Converts the frame to mid/side when that needs fewer bits. Returns whether it did,
  /// and each granule's share of energy in the side channel.
  private mutating func decideMidSide(_ masking: inout [[MaskingResult]]) -> (Bool, [Float]) {
    let granules = format.granules
    guard (0..<granules).allSatisfy({ blockTypes[$0][0] == blockTypes[$0][1] }) else {
      return (false, [Float](repeating: 0.5, count: granules))
    }
    let scale = Float(0.5).squareRoot()
    var mid = [[Float]]()
    var side = [[Float]]()
    var midSideMasking = [[MaskingResult]]()
    var ratios = [Float]()
    var entropyLR: Float = 0
    var entropyMS: Float = 0
    for gr in 0..<granules {
      var m = [Float](repeating: 0, count: 576)
      var s = [Float](repeating: 0, count: 576)
      var midEnergy: Float = 0
      var sideEnergy: Float = 0
      for i in 0..<576 {
        let l = spectra[gr][0][i]
        let r = spectra[gr][1][i]
        m[i] = (l + r) * scale
        s[i] = (l - r) * scale
        midEnergy += m[i] * m[i]
        sideEnergy += s[i] * s[i]
      }
      ratios.append(midEnergy + sideEnergy > 0 ? sideEnergy / (midEnergy + sideEnergy) : 0.5)

      // Mid and side share the stricter of the left and right thresholds.
      let left = masking[gr][0]
      let right = masking[gr][1]
      let allowed = zip(left.allowedNoise, right.allowedNoise).map { min($0, $1) }
      let ms = [m, s].map { channel -> MaskingResult in
        let energy = Self.bandEnergies(channel, like: left.energy.count, psychoacoustics)
        var pe: Float = 0
        for b in 0..<energy.count where energy[b] > allowed[b] {
          pe += Float(Self.bandWidth(b, count: energy.count, psychoacoustics)) * 0.5 * log2(energy[b] / allowed[b])
        }
        return MaskingResult(allowedNoise: allowed, energy: energy, perceptualEntropy: pe)
      }
      entropyLR += left.perceptualEntropy + right.perceptualEntropy
      entropyMS += ms[0].perceptualEntropy + ms[1].perceptualEntropy
      mid.append(m)
      side.append(s)
      midSideMasking.append(ms)
    }
    guard entropyMS < entropyLR else { return (false, ratios) }
    for gr in 0..<granules {
      spectra[gr][0] = mid[gr]
      spectra[gr][1] = side[gr]
    }
    masking = midSideMasking
    return (true, ratios)
  }

  private static func bandEnergies(_ x: [Float], like count: Int, _ model: PsychoacousticModel) -> [Float] {
    var energy = [Float](repeating: 0, count: count)
    for b in 0..<count {
      let (start, end): (Int, Int)
      if count == 22 {
        (start, end) = (model.longBands[b], model.longBands[b + 1])
      } else {
        let band = b / 3
        let width = model.shortBands[band + 1] - model.shortBands[band]
        start = model.shortBands[band] * 3 + (b % 3) * width
        end = start + width
      }
      var sum: Float = 0
      for i in start..<end { sum += x[i] * x[i] }
      energy[b] = sum
    }
    return energy
  }

  private static func bandWidth(_ b: Int, count: Int, _ model: PsychoacousticModel) -> Int {
    if count == 22 { return model.longBands[b + 1] - model.longBands[b] }
    return model.shortBands[b / 3 + 1] - model.shortBands[b / 3]
  }

  /// Distributes a granule's bits between channels (after LAME's reservoir and `on_pe` logic).
  static func allocateBits(
    meanBits: Int,
    reservoir: Int,
    maxReservoir: Int,
    perceptualEntropy pe: [Float],
    averageEntropy: Float,
    sideRatio: Float?,
    constantBitrate: Bool
  ) -> [Int] {
    let channels = pe.count
    var target = meanBits
    var add = 0
    if reservoir > maxReservoir * 9 / 10 {
      add = reservoir - maxReservoir * 9 / 10
      target += add
    } else if constantBitrate && maxReservoir > 0 {
      // Save a little in easy granules to build up the reservoir.
      target -= meanBits / 10
    }
    let extra = max(0, min(reservoir, maxReservoir * 6 / 10) - add)

    let perChannel = target / channels
    var bits = [Int](repeating: 0, count: channels)
    var extras = [Int](repeating: 0, count: channels)
    var totalExtra = 0
    for ch in 0..<channels {
      bits[ch] = min(4095, perChannel)
      var e = Int(Float(bits[ch]) * pe[ch] / averageEntropy) - bits[ch]
      e = min(max(e, 0), meanBits * 3 / 4)
      e = min(e, 4095 - bits[ch])
      extras[ch] = e
      totalExtra += e
    }
    if totalExtra > extra && totalExtra > 0 {
      for ch in 0..<channels {
        extras[ch] = extra * extras[ch] / totalExtra
      }
    }
    for ch in 0..<channels {
      bits[ch] += extras[ch]
    }

    // Give the mid channel more of the bits when the side channel is quiet.
    if let ratio = sideRatio, channels == 2 {
      let factor = min(max(0.33 * (0.5 - ratio) / 0.5, 0), 0.5)
      var move = Int(factor * 0.5 * Float(bits[0] + bits[1]))
      move = max(0, min(move, 4095 - bits[0]))
      if bits[1] >= 125 {
        if bits[1] - move > 125 {
          bits[0] += move
          bits[1] -= move
        } else {
          bits[0] += bits[1] - 125
          bits[1] = 125
        }
      }
    }
    return bits
  }

  /// Sets `scfsi` for scalefactor groups that granule 1 can copy from granule 0, and
  /// returns the bits saved.
  private func applySCFSI(_ second: inout GranuleInfo, first: GranuleInfo, scfsi: inout [Int]) -> Int {
    guard !first.isShort, !second.isShort, second.part23Length > 0 else { return 0 }
    let groups = [0..<6, 6..<11, 11..<16, 16..<21]
    var any = false
    for (g, range) in groups.enumerated() where range.allSatisfy({ first.scalefactors[$0] == second.scalefactors[$0] }) {
      scfsi[g] = 1
      any = true
    }
    guard any else { return 0 }

    // Choose the smallest slen that fits the scalefactors still transmitted.
    var max1 = 0
    var max2 = 0
    for (g, range) in groups.enumerated() where scfsi[g] == 0 {
      for band in range {
        if band < 11 { max1 = max(max1, second.scalefactors[band]) } else { max2 = max(max2, second.scalefactors[band]) }
      }
    }
    var bestIndex = second.scalefacCompress
    var bestBits = Int.max
    for (index, (slen1, slen2)) in MPEG.slen.enumerated() where max1 < (1 << slen1) && max2 < (1 << slen2) {
      var bits = 0
      for (g, range) in groups.enumerated() where scfsi[g] == 0 {
        bits += range.count * (range.lowerBound < 11 ? slen1 : slen2)
      }
      if bits < bestBits {
        bestBits = bits
        bestIndex = index
      }
    }
    let saved = second.part2Length - bestBits
    second.scalefacCompress = bestIndex
    second.part2Length = bestBits
    second.part23Length -= saved
    return saved
  }

  private func writeScalefactors(_ info: GranuleInfo, granule: Int, scfsi: [Int], into writer: inout BitWriter) {
    if !format.isMPEG1 {
      // MPEG-2: scalefactors in coding order, in groups that share an `slen`.
      let (table, slen) = LSFScalefactors.decode(info.scalefacCompress)
      var entry = 0
      for (group, count) in LSFScalefactors.groups[table][info.isShort ? 1 : 0].enumerated() {
        for e in entry..<(entry + count) {
          writer.write(info.scalefactors[e], slen[group])
        }
        entry += count
      }
      return
    }
    let (slen1, slen2) = MPEG.slen[info.scalefacCompress]
    if info.isShort {
      for band in 0..<12 {
        let slen = band < 6 ? slen1 : slen2
        for w in 0..<3 {
          writer.write(info.scalefactors[band * 3 + w], slen)
        }
      }
    } else {
      let groups = [0..<6, 6..<11, 11..<16, 16..<21]
      for (g, range) in groups.enumerated() where granule == 0 || scfsi[g] == 0 {
        for band in range {
          writer.write(info.scalefactors[band], band < 11 ? slen1 : slen2)
        }
      }
    }
  }

  // MARK: - Bitstream

  private func frameHeader(
    bitrateIndex: Int,
    padding: Bool,
    midSide: Bool,
    granules: [[GranuleInfo]],
    scfsi: [[Int]],
    mainDataBegin: Int
  ) -> [UInt8] {
    var bytes = headerBytes(bitrateIndex: bitrateIndex, padding: padding, crc: options.crcProtected, midSide: midSide)

    let mpeg1 = format.isMPEG1
    var side = BitWriter(capacity: format.sideInfoSize)
    if mpeg1 {
      side.write(mainDataBegin, 9)
      side.write(0, channels == 1 ? 5 : 3)
      for ch in 0..<channels {
        for g in 0..<4 { side.write(scfsi[ch][g], 1) }
      }
    } else {
      side.write(mainDataBegin, 8)
      side.write(0, channels)
    }
    for gr in 0..<format.granules {
      for ch in 0..<channels {
        let info = granules[gr][ch]
        side.write(info.part23Length, 12)
        side.write(info.bigValues, 9)
        side.write(info.globalGain, 8)
        side.write(info.scalefacCompress, mpeg1 ? 4 : 9)
        if info.blockType != .normal {
          side.write(1, 1)
          side.write(info.blockType.rawValue, 2)
          side.write(0, 1)  // mixed_block_flag
          side.write(info.tableSelect[0], 5)
          side.write(info.tableSelect[1], 5)
          side.write(0, 9)  // subblock_gain
        } else {
          side.write(0, 1)
          side.write(info.tableSelect[0], 5)
          side.write(info.tableSelect[1], 5)
          side.write(info.tableSelect[2], 5)
          side.write(info.region0Count, 4)
          side.write(info.region1Count, 3)
        }
        if mpeg1 {
          // MPEG-2 derives preflag from scalefac_compress.
          side.write(info.preflag, 1)
        }
        side.write(info.scalefacScale, 1)
        side.write(info.count1TableSelect, 1)
      }
    }
    side.alignToByte()
    assert(side.bytes.count == format.sideInfoSize)

    if options.crcProtected {
      let crc = CRC16.mpeg(bytes[2...3], side.bytes[...])
      bytes.append(UInt8(crc >> 8))
      bytes.append(UInt8(crc & 0xFF))
    }
    bytes.append(contentsOf: side.bytes)
    return bytes
  }

  /// The four-byte frame header.
  private func headerBytes(bitrateIndex: Int, padding: Bool, crc: Bool, midSide: Bool) -> [UInt8] {
    var header = BitWriter(capacity: 4)
    header.write(0x7FF, 11)
    header.write(format.version.headerBits, 2)
    header.write(0b01, 2)  // Layer III
    header.write(crc ? 0 : 1, 1)
    header.write(bitrateIndex, 4)
    header.write(format.sampleRateIndex, 2)
    header.write(padding ? 1 : 0, 1)
    header.write(0, 1)  // Private bit
    let modeBits: Int
    switch options.mode {
    case .mono: modeBits = 0b11
    case .jointStereo: modeBits = 0b01
    case .stereo: modeBits = 0b00
    }
    header.write(modeBits, 2)
    header.write(midSide ? 0b10 : 0b00, 2)
    header.write(options.copyright ? 1 : 0, 1)
    header.write(options.original ? 1 : 0, 1)
    header.write(0, 2)  // No emphasis
    return header.bytes
  }

  /// Outputs every queued frame whose main data slot can be filled.
  private mutating func emitReadyFrames(into output: inout [UInt8]) {
    while pendingFrameIndex < pendingFrames.count {
      let frame = pendingFrames[pendingFrameIndex]
      guard mainData.count - mainDataOffset >= frame.slotSize else { break }
      output.append(contentsOf: frame.header)
      output.append(contentsOf: mainData[mainDataOffset..<(mainDataOffset + frame.slotSize)])
      musicCRC = CRC16.lame(frame.header[...], continuing: musicCRC)
      musicCRC = CRC16.lame(mainData[mainDataOffset..<(mainDataOffset + frame.slotSize)], continuing: musicCRC)
      mainDataOffset += frame.slotSize
      pendingFrameIndex += 1

      let size = frame.header.count + frame.slotSize
      frameCount += 1
      totalBytes += size
      frameSizes.append(UInt16(size))
    }
    if pendingFrameIndex > 64 {
      pendingFrames.removeFirst(pendingFrameIndex)
      pendingFrameIndex = 0
    }
    if mainDataOffset > 8192 {
      mainData.removeFirst(mainDataOffset)
      mainDataOffset = 0
    }
  }

  // MARK: - Xing header

  /// Samples of delay the encoder adds before the first input sample, not counting the
  /// decoder's own 529-sample delay.
  static let encoderDelay = 528

  /// Size of the Xing/Info frame produced by ``generateXingHeader()``.
  var xingFrameSize: Int {
    format.frameLength(bitrateIndex: xingBitrateIndex)
  }

  /// Bitrate for the Xing frame: the stream's bitrate, raised if needed to fit the tags.
  private var xingBitrateIndex: Int {
    let needed = 4 + format.sideInfoSize + 120 + 36
    var index = targetBitrateIndex
    while index < 14 && format.frameLength(bitrateIndex: index) < needed {
      index += 1
    }
    return index
  }

  /// Generates a Xing/Info header frame for seeking, duration, and gapless playback.
  ///
  /// Call this after encoding is complete to get accurate frame and byte counts.
  /// The returned data should be prepended to the encoded audio data. Uses the
  /// "Xing" tag for VBR and "Info" tag for CBR streams, followed by a LAME tag that
  /// records the encoder delay and padding so players can trim them.
  ///
  /// - Returns: A complete MP3 frame containing the Xing/Info header.
  public func generateXingHeader() -> Data {
    let frameSize = xingFrameSize
    var frame = headerBytes(bitrateIndex: xingBitrateIndex, padding: false, crc: false, midSide: false)
    frame.append(contentsOf: repeatElement(0, count: format.sideInfoSize))
    frame.append(contentsOf: Array((options.vbr ? "Xing" : "Info").utf8))
    frame.append(contentsOf: UInt32(0x0F).bigEndianBytes)  // Frames, bytes, TOC, quality
    frame.append(contentsOf: UInt32(frameCount).bigEndianBytes)
    let streamBytes = UInt32(clamping: totalBytes + frameSize)
    frame.append(contentsOf: streamBytes.bigEndianBytes)
    frame.append(contentsOf: generateTOC(headerSize: frameSize))
    frame.append(contentsOf: UInt32(100 - 10 * options.quality).bigEndianBytes)
    frame.append(contentsOf: lameTag())
    frame.append(contentsOf: streamBytes.bigEndianBytes)
    frame.append(UInt8(musicCRC >> 8))
    frame.append(UInt8(musicCRC & 0xFF))
    let tagCRC = CRC16.lame(frame[...])
    frame.append(UInt8(tagCRC >> 8))
    frame.append(UInt8(tagCRC & 0xFF))
    if frame.count < frameSize {
      frame.append(contentsOf: repeatElement(0, count: frameSize - frame.count))
    }
    return Data(frame)
  }

  /// The LAME tag fields from the version string through the preset (28 bytes).
  private func lameTag() -> [UInt8] {
    var tag = Array("LAMESwMP3".utf8)
    tag.append(options.vbr ? 2 : 1)  // Revision 0; VBR method: ABR or CBR
    tag.append(UInt8(min(255, (lowpassHz + 50) / 100)))
    tag.append(contentsOf: replayGainFields())
    tag.append(0)  // Encoding flags and ATH type
    tag.append(UInt8(min(255, options.bitrateKbps)))
    let delay = Self.encoderDelay
    let padding = min(max(frameCount * format.samplesPerFrame - delay - inputFrameCount, 0), 4095)
    tag.append(UInt8(delay >> 4))
    tag.append(UInt8((delay & 0xF) << 4 | padding >> 8))
    tag.append(UInt8(padding & 0xFF))
    let sourceRate: UInt8
    switch options.sampleRate {
    case ...32_000: sourceRate = 0
    case ...44_100: sourceRate = 1
    case ...48_000: sourceRate = 2
    default: sourceRate = 3
    }
    let stereoMode: UInt8
    switch options.mode {
    case .mono: stereoMode = 0
    case .stereo: stereoMode = 1
    case .jointStereo: stereoMode = 3
    }
    tag.append(sourceRate << 6 | stereoMode << 2)
    tag.append(0)  // MP3 gain
    tag.append(contentsOf: [0, 0])  // Surround and preset
    return tag
  }

  /// Peak amplitude (fixed point, 1.0 = 2^23) and the radio (track) gain, with no
  /// audiophile (album) gain.
  private func replayGainFields() -> [UInt8] {
    let peak = UInt32(min(Double(loudness.peak), 255) * Double(1 << 23)).bigEndianBytes
    var radio: UInt16 = 0
    if let gain = loudness.trackGain {
      // Name code 1 (radio), originator 3 (automatic), sign, then tenths of a dB.
      let tenths = UInt16(min((abs(gain) * 10).rounded(), 511))
      radio = 0b001 << 13 | 0b011 << 10 | (gain < 0 ? 1 << 9 : 0) | tenths
    }
    return peak + [UInt8(radio >> 8), UInt8(radio & 0xFF), 0, 0]
  }

  /// The 100-entry seek table: byte position (scaled to 0-255) of each percent of duration.
  private func generateTOC(headerSize: Int) -> [UInt8] {
    guard !frameSizes.isEmpty else {
      return (0..<100).map { UInt8($0 * 255 / 99) }
    }
    let total = Double(totalBytes + headerSize)
    var toc = [UInt8]()
    toc.reserveCapacity(100)
    var position = headerSize
    var frame = 0
    for percent in 0..<100 {
      let target = percent * frameSizes.count / 100
      while frame < target {
        position += Int(frameSizes[frame])
        frame += 1
      }
      toc.append(UInt8(min(255, Int(Double(position) / total * 256))))
    }
    return toc
  }
}

/// Detects attacks in subband samples to trigger short blocks.
struct TransientDetector {
  private var history: [Float] = [0, 0, 0]

  /// Splits a granule of subband samples into six segments of three slots and flags
  /// segments whose high-frequency energy jumps well above the recent past.
  mutating func analyze(_ subbands: UnsafePointer<Float>) -> (early: Bool, late: Bool) {
    var early = false
    var late = false
    for segment in 0..<6 {
      var energy: Float = 0
      for slot in (segment * 3)..<(segment * 3 + 3) {
        for sb in 2..<32 {
          let v = subbands[slot * 32 + sb]
          energy += v * v
        }
      }
      let reference = max(history[0], history[1], history[2])
      if energy > reference * 10 && energy > 1e-6 {
        if segment < 4 { early = true } else { late = true }
      }
      history[0] = history[1]
      history[1] = history[2]
      history[2] = energy
    }
    return (early, late)
  }
}

/// Picks the per-frame bitrate ceiling for VBR so the average stays near the target.
struct BitrateController {
  private var averageEntropy: Float = 0
  private var targetBytes: Double = 0
  private var actualBytes: Double = 0

  mutating func choose(target targetIndex: Int, format: StreamFormat, perceptualEntropy: Float) -> Int {
    let target = Double(format.scaledFrameLength(bitrateIndex: targetIndex)) / Double(format.sampleRate)
    if averageEntropy == 0 { averageEntropy = max(perceptualEntropy, 1) }
    averageEntropy += 0.02 * (perceptualEntropy - averageEntropy)
    let ratio = Double(min(max(perceptualEntropy / max(averageEntropy, 1), 0.5), 2))
    // Steer back toward the target average over roughly 50 frames. Savings from frames
    // that needed less than the target are only banked up to half a frame's worth of boost.
    targetBytes = min(targetBytes, actualBytes + 25 * target)
    let drift = (targetBytes - actualBytes) * 0.02
    let desired = target * ratio + drift
    targetBytes += target
    var index = 1
    while index < 14 && Double(format.frameLength(bitrateIndex: index)) < desired {
      index += 1
    }
    return index
  }

  mutating func record(frameBytes: Int) {
    actualBytes += Double(frameBytes)
  }
}

/// CRC-16 with the MPEG polynomial (0x8005), initialized to 0xFFFF.
enum CRC16 {
  /// Table for the reflected form of the polynomial (0xA001) used by LAME tags.
  private static let reflectedTable: [UInt16] = (0..<256).map { byte in
    var crc = UInt16(byte)
    for _ in 0..<8 {
      crc = crc & 1 != 0 ? (crc >> 1) ^ 0xA001 : crc >> 1
    }
    return crc
  }

  /// The LAME tag's CRC-16 (reflected, initialized to 0), continued from `crc`.
  static func lame(_ bytes: ArraySlice<UInt8>, continuing crc: UInt16 = 0) -> UInt16 {
    var crc = crc
    reflectedTable.withUnsafeBufferPointer { table in
      for byte in bytes {
        crc = (crc >> 8) ^ table[Int((crc ^ UInt16(byte)) & 0xFF)]
      }
    }
    return crc
  }

  static func mpeg(_ parts: ArraySlice<UInt8>...) -> UInt16 {
    var crc: UInt16 = 0xFFFF
    for part in parts {
      for byte in part {
        var value = UInt16(byte) << 8
        for _ in 0..<8 {
          let feedback = (crc ^ value) & 0x8000
          crc <<= 1
          value <<= 1
          if feedback != 0 { crc ^= 0x8005 }
        }
      }
    }
    return crc
  }
}

extension UInt32 {
  /// The value serialized as 4 bytes in big-endian (network) byte order.
  fileprivate var bigEndianBytes: [UInt8] {
    [UInt8((self >> 24) & 0xFF), UInt8((self >> 16) & 0xFF), UInt8((self >> 8) & 0xFF), UInt8(self & 0xFF)]
  }
}
