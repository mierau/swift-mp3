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
  private let channels: Int
  private let sampleRateIndex: Int
  private let sideInfoSize: Int

  // Input
  private var pcm: [Float] = []
  private var inputFrames = 0
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
    channels = options.channelCount
    let srIndex = MPEG.sampleRateIndex(for: options.sampleRate) ?? 0
    sampleRateIndex = srIndex
    sideInfoSize = channels == 1 ? 17 : 32

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
    let shortBands = MPEG.shortBands(sampleRateIndex: srIndex)
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

    let sampleRate = MPEG.sampleRates[srIndex]
    let kbps = MPEG.bitrates[MPEG.bitrateIndex(for: options.bitrateKbps)]
    let cutoff = Self.lowpassFrequency(kbps: kbps, channels: channels, sampleRate: sampleRate)
    lowpassLong = min(576, Int((cutoff / Double(sampleRate / 2) * 576).rounded()))
    lowpassShort = min(192, Int((cutoff / Double(sampleRate / 2) * 192).rounded()))

    psychoacoustics = PsychoacousticModel(sampleRate: sampleRate, sampleRateIndex: srIndex, quality: options.quality)
    quantizer = Quantizer(sampleRateIndex: srIndex, quality: options.quality)
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
    guard !finished else { return Data() }
    pcm.append(contentsOf: samples)
    inputFrames += samples.count / channels
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
    guard inputFrames > 0 else { return Data() }

    // An input sample affects subband granules up to (n + 480) / 576; the MDCT needs one
    // more granule, and block-type lookahead one more after that.
    let lastSubbandGranule = (inputFrames - 1 + 480) / MPEG.samplesPerGranule
    let frames = (lastSubbandGranule + 2 + 1) / 2
    let neededGranules = frames * 2 + 1
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

  /// Lowpass frequency for a bitrate, after LAME's bandwidth table for stereo.
  static func lowpassFrequency(kbps: Int, channels: Int, sampleRate: Int) -> Double {
    let table: [(Int, Double)] = [
      (32, 5500), (40, 7000), (48, 7500), (56, 10_000), (64, 11_000), (80, 13_500),
      (96, 15_100), (112, 15_600), (128, 17_000), (160, 17_500), (192, 18_600),
      (224, 19_400), (256, 19_700), (320, 20_500),
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
    return min(cutoff, Double(sampleRate) / 2)
  }

  // MARK: - Frame encoding

  private mutating func encodeReadyFrame(into output: inout [UInt8]) {
    guard mdctGranules >= 2 else { return }
    mdctGranules = 0
    encodeFrame()
    emitReadyFrames(into: &output)
  }

  private mutating func encodeFrame() {
    // Psychoacoustic analysis of the coded channels.
    var masking = [[MaskingResult]]()
    for gr in 0..<2 {
      var row = [MaskingResult]()
      for ch in 0..<channels {
        row.append(analyze(granule: gr, channel: ch))
      }
      masking.append(row)
    }

    var useMidSide = false
    var sideRatio = [Float](repeating: 0.5, count: 2)
    if options.mode == .jointStereo {
      (useMidSide, sideRatio) = decideMidSide(&masking)
    }

    // Bitrate and frame size.
    let totalPE = masking.joined().reduce(Float(0)) { $0 + $1.perceptualEntropy }
    let sampleRate = MPEG.sampleRates[sampleRateIndex]
    let crcSize = options.crcProtected ? 2 : 0
    var bitrateIndex = MPEG.bitrateIndex(for: options.bitrateKbps)
    // VBR: quantize every granule just finely enough to mask its noise, then pick the
    // smallest bitrate that holds the result, capped by the average-bitrate target.
    var transparent: [[GranuleInfo]]?
    if options.vbr {
      var rows = [[GranuleInfo]]()
      var demand = 0
      for gr in 0..<2 {
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
      let ceiling = bitrate.choose(targetKbps: options.bitrateKbps, sampleRate: sampleRate, perceptualEntropy: totalPE)
      bitrateIndex = 1
      while bitrateIndex < ceiling {
        let slotBits = (MPEG.frameLength(bitrateIndex: bitrateIndex, sampleRate: sampleRate) - 4 - crcSize - sideInfoSize) * 8
        if demand <= slotBits + reservoirBits { break }
        bitrateIndex += 1
      }
      if rows.allSatisfy({ $0.count == channels }) {
        transparent = rows
      }
    }
    let numerator = 144_000 * MPEG.bitrates[bitrateIndex]
    paddingRemainder += numerator % sampleRate
    let padding = paddingRemainder >= sampleRate
    if padding { paddingRemainder -= sampleRate }
    let frameLength = MPEG.frameLength(bitrateIndex: bitrateIndex, sampleRate: sampleRate, padding: padding)
    let slotSize = frameLength - 4 - crcSize - sideInfoSize
    let slotBits = slotSize * 8
    let maxReservoir = max(0, min(511 * 8, 7680 - frameLength * 8)) & ~7

    let mainDataBegin = reservoirBits / 8
    var reservoir = reservoirBits
    let meanBits = slotBits / 2

    for value in masking.joined().map(\.perceptualEntropy) {
      averageEntropy = averageEntropy == 0 ? max(value, 1) : averageEntropy + 0.02 * (value - averageEntropy)
    }

    var granules = [[GranuleInfo]]()
    var usedBits = 0
    if let transparent, transparent.joined().reduce(0, { $0 + $1.part23Length }) <= reservoirBits + slotBits {
      granules = transparent
      usedBits = transparent.joined().reduce(0) { $0 + $1.part23Length }
    } else {
      for gr in 0..<2 {
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
        granules.append(row)
      }
    }

    // Share scalefactors between granules where they repeat.
    var scfsi = Array(repeating: [0, 0, 0, 0], count: channels)
    for ch in 0..<channels {
      let saved = applySCFSI(&granules[1][ch], first: granules[0][ch], scfsi: &scfsi[ch])
      usedBits -= saved
    }

    // Main data.
    var writer = BitWriter(capacity: (usedBits + 7) / 8 + 8)
    let longBands = MPEG.longBands(sampleRateIndex: sampleRateIndex)
    for gr in 0..<2 {
      for ch in 0..<channels {
        let info = granules[gr][ch]
        let start = writer.bitCount
        writeScalefactors(info, granule: gr, scfsi: scfsi[ch], into: &writer)
        spectra[gr][ch].withUnsafeBufferPointer { spectrum in
          HuffmanWriter.write(info, spectrum: spectrum.baseAddress!, longBands: longBands, into: &writer)
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
      granules: granules,
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
    guard blockTypes[0][0] == blockTypes[0][1], blockTypes[1][0] == blockTypes[1][1] else {
      return (false, [0.5, 0.5])
    }
    let scale = Float(0.5).squareRoot()
    var mid = [[Float]]()
    var side = [[Float]]()
    var midSideMasking = [[MaskingResult]]()
    var ratios = [Float]()
    var entropyLR: Float = 0
    var entropyMS: Float = 0
    for gr in 0..<2 {
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
    for gr in 0..<2 {
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
    var header = BitWriter(capacity: 4)
    header.write(0x7FF, 11)
    header.write(0b11, 2)  // MPEG-1
    header.write(0b01, 2)  // Layer III
    header.write(options.crcProtected ? 0 : 1, 1)
    header.write(bitrateIndex, 4)
    header.write(sampleRateIndex, 2)
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

    var side = BitWriter(capacity: sideInfoSize)
    side.write(mainDataBegin, 9)
    side.write(0, channels == 1 ? 5 : 3)
    for ch in 0..<channels {
      for g in 0..<4 { side.write(scfsi[ch][g], 1) }
    }
    for gr in 0..<2 {
      for ch in 0..<channels {
        let info = granules[gr][ch]
        side.write(info.part23Length, 12)
        side.write(info.bigValues, 9)
        side.write(info.globalGain, 8)
        side.write(info.scalefacCompress, 4)
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
        side.write(info.preflag, 1)
        side.write(info.scalefacScale, 1)
        side.write(info.count1TableSelect, 1)
      }
    }
    side.alignToByte()

    var bytes = header.bytes
    if options.crcProtected {
      let crc = CRC16.mpeg(bytes[2...3], side.bytes[...])
      bytes.append(UInt8(crc >> 8))
      bytes.append(UInt8(crc & 0xFF))
    }
    bytes.append(contentsOf: side.bytes)
    return bytes
  }

  /// Outputs every queued frame whose main data slot can be filled.
  private mutating func emitReadyFrames(into output: inout [UInt8]) {
    while pendingFrameIndex < pendingFrames.count {
      let frame = pendingFrames[pendingFrameIndex]
      guard mainData.count - mainDataOffset >= frame.slotSize else { break }
      output.append(contentsOf: frame.header)
      output.append(contentsOf: mainData[mainDataOffset..<(mainDataOffset + frame.slotSize)])
      mainDataOffset += frame.slotSize
      pendingFrameIndex += 1

      let size = frame.header.count + frame.slotSize
      frameCount += 1
      totalBytes += UInt32(size)
      frameSizes.append(size)
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

  /// Size of the Xing/Info frame produced by ``generateXingHeader()``.
  var xingFrameSize: Int {
    MPEG.frameLength(bitrateIndex: xingBitrateIndex, sampleRate: MPEG.sampleRates[sampleRateIndex])
  }

  /// Bitrate for the Xing frame: the stream's bitrate, raised if needed to fit the tag.
  private var xingBitrateIndex: Int {
    let needed = 4 + sideInfoSize + 4 + 4 + 4 + 4 + 100
    var index = MPEG.bitrateIndex(for: options.bitrateKbps)
    let sampleRate = MPEG.sampleRates[sampleRateIndex]
    while index < 14 && MPEG.frameLength(bitrateIndex: index, sampleRate: sampleRate) < needed {
      index += 1
    }
    return index
  }

  /// Generates a Xing/Info header frame for seeking and duration metadata.
  ///
  /// Call this after encoding is complete to get accurate frame and byte counts.
  /// The returned data should be prepended to the encoded audio data. Uses the
  /// "Xing" tag for VBR and "Info" tag for CBR streams.
  ///
  /// - Returns: A complete MP3 frame containing the Xing/Info header.
  public func generateXingHeader() -> Data {
    let frameSize = xingFrameSize
    var header = BitWriter(capacity: 4)
    header.write(0x7FF, 11)
    header.write(0b11, 2)
    header.write(0b01, 2)
    header.write(1, 1)  // No CRC
    header.write(xingBitrateIndex, 4)
    header.write(sampleRateIndex, 2)
    header.write(0, 1)  // No padding
    header.write(0, 1)
    switch options.mode {
    case .mono: header.write(0b11, 2)
    case .jointStereo: header.write(0b01, 2)
    case .stereo: header.write(0b00, 2)
    }
    header.write(0, 2)
    header.write(options.copyright ? 1 : 0, 1)
    header.write(options.original ? 1 : 0, 1)
    header.write(0, 2)

    var frame = header.bytes
    frame.append(contentsOf: repeatElement(0, count: sideInfoSize))
    frame.append(contentsOf: Array((options.vbr ? "Xing" : "Info").utf8))
    frame.append(contentsOf: UInt32(0x07).bigEndianBytes)  // Frames, bytes, TOC
    frame.append(contentsOf: frameCount.bigEndianBytes)
    frame.append(contentsOf: (totalBytes + UInt32(frameSize)).bigEndianBytes)
    frame.append(contentsOf: generateTOC(headerSize: frameSize))
    if frame.count < frameSize {
      frame.append(contentsOf: repeatElement(0, count: frameSize - frame.count))
    }
    return Data(frame)
  }

  /// The 100-entry seek table: byte position (scaled to 0-255) of each percent of duration.
  private func generateTOC(headerSize: Int) -> [UInt8] {
    guard !frameSizes.isEmpty else {
      return (0..<100).map { UInt8($0 * 255 / 99) }
    }
    let total = Double(Int(totalBytes) + headerSize)
    var toc = [UInt8]()
    toc.reserveCapacity(100)
    var position = headerSize
    var frame = 0
    for percent in 0..<100 {
      let target = percent * frameSizes.count / 100
      while frame < target {
        position += frameSizes[frame]
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

  mutating func choose(targetKbps: Int, sampleRate: Int, perceptualEntropy: Float) -> Int {
    let target = 144_000 * Double(targetKbps) / Double(sampleRate)
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
    while index < 14 && Double(MPEG.frameLength(bitrateIndex: index, sampleRate: sampleRate)) < desired {
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
