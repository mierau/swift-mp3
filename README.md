# SwiftMP3

[![CI](https://github.com/mierau/swift-mp3/actions/workflows/ci.yml/badge.svg)](https://github.com/mierau/swift-mp3/actions/workflows/ci.yml)

A pure Swift MP3 encoder with no dependencies. Encode PCM audio to MPEG Layer III (MP3) entirely in Swift, on Apple platforms and Linux.

## Features

- MPEG-1 Layer III at 32, 44.1, and 48 kHz, and MPEG-2/2.5 at 8 to 24 kHz for low-bitrate speech
- Any input sample rate: 88.2, 96, 192 kHz or anything else MP3 can't store is resampled
- CBR and VBR encoding modes, with a bit reservoir
- Scores higher than LAME on ViSQOL at the same bitrate, with smaller VBR files (see [Quality](#quality))
- Mono, stereo, and joint stereo
- Streaming encoding — feed samples incrementally
- Async/await support with `AsyncSequence` input
- File encoding with automatic Xing header, from an async sequence or straight from an audio file, with progress
- `mp3encode`, a small command-line encoder
- Xing/Info and LAME tags for accurate seeking and gapless playback
- ReplayGain 2.0 track gain and peak (ITU-R BS.1770 loudness) in the LAME tag
- Float, 16-bit integer, planar, and `AVAudioPCMBuffer` input, with downmixing to mono
- Psychoacoustic model (spreading, tonality, absolute threshold of hearing, pre-echo control)
- Noise-shaping quantization with scalefactors, pre-emphasis, and scalefactor sharing
- Transient detection with long/short block switching
- Mid/side stereo chosen per frame, and intensity stereo for the high bands at 80 kbps and below
- ISO 11172-3 compliant Huffman coding and polyphase filterbank
- Encodes 40-250x faster than realtime on one core, using Accelerate on Apple platforms and SIMD elsewhere
- ID3v2.3 metadata tagging (title, artist, album artist, composer, disc, lyrics, artwork, ReplayGain, etc.)
- Fully `Sendable` — safe to use from any concurrency context

## Requirements

- Swift 6.0+ (Xcode 16+)
- macOS 13+, iOS 16+, tvOS 16+, watchOS 9+, visionOS 1+, or Linux
- The decode round-trip tests use AVFoundation on Apple platforms and ffmpeg on Linux

## Installation

Add SwiftMP3 to your `Package.swift`:

```swift
dependencies: [
  .package(url: "https://github.com/mierau/swift-mp3.git", from: "1.0.0")
]
```

Then add `SwiftMP3` as a dependency of your target:

```swift
.target(
  name: "MyApp",
  dependencies: [
    .product(name: "SwiftMP3", package: "swift-mp3")
  ]
)
```

In Xcode, choose **File › Add Package Dependencies…** and enter the repository URL.

## Usage

### Basic Encoding

Use `newSession()` to create a mutable encoding session for synchronous, incremental encoding:

```swift
import SwiftMP3

let encoder = try MP3Encoder(options: MP3EncoderOptions(
  sampleRate: 44_100,
  bitrateKbps: 128,
  mode: .jointStereo
))
var session = encoder.newSession()

// Feed interleaved PCM float samples (L, R, L, R, ...)
var mp3Data = session.encode(samples: pcmSamples)

// Flush any remaining buffered samples
mp3Data.append(session.flush())

// Generate the Info frame for seeking and gapless playback (prepend to file)
let xingHeader = session.generateXingHeader()

// Write the complete MP3 file
var file = Data()
file.append(xingHeader)
file.append(mp3Data)
```

`MP3Encoder(options:)` throws an `MP3EncoderError` if the sample rate, bitrate, or quality isn't supported.

### Other Input Formats

Sessions also take 16-bit integer samples, one array per channel, or an `AVAudioPCMBuffer`:

```swift
var data = session.encode(samples: int16Samples)    // [Int16], interleaved
data.append(session.encode(planar: [left, right]))  // [[Float]], one array per channel
data.append(try session.encode(buffer: pcmBuffer))  // AVAudioPCMBuffer, float or int16
```

The buffer must match the encoder's input sample rate. A mono encoder averages planar or buffer input with any number of channels, and a stereo encoder plays mono input in both channels.

`encode(_:)` and `encode(_:to:)` take an `AsyncSequence` of `[Int16]` as well as `[Float]`, and `encode(_:to:)` also takes a sequence of `AVAudioPCMBuffer`.

### Sample Rates

Set `sampleRate` to the rate of your input. If MP3 can't store that rate, the encoder resamples it, by default to the nearest rate that keeps its bandwidth (44.1 or 48 kHz above 48 kHz). Set `outputSampleRate` to choose the MP3's rate yourself, for example 22.05 kHz for speech:

```swift
let encoder = try MP3Encoder(options: MP3EncoderOptions(
  sampleRate: 96_000,          // input
  outputSampleRate: 44_100,    // MP3
  bitrateKbps: 192
))
```

Resampling keeps the output sample-exact, so gapless playback still trims correctly.

### Gapless Playback

The Info frame from `generateXingHeader()` (and from `encode(_:to:)`) carries a LAME tag with the encoder delay and padding. Players that read it, such as ffmpeg and mpg123, trim them, so the decoded audio has exactly as many samples as the input and loops or consecutive tracks play without a gap.

### Loudness

The LAME tag also records the track's ReplayGain 2.0 gain (the change that brings it to -18 LUFS, measured per ITU-R BS.1770) and its peak sample, so players that support ReplayGain can level it. The same values are available as `session.replayGain` and `session.peakAmplitude`.

### Streaming Encode

Use `encode(_:)` with any `AsyncSequence` of `[Float]` to get an `AsyncThrowingStream` of MP3 frame data:

```swift
let encoder = try MP3Encoder(options: options)

for try await frame in encoder.encode(sampleSource) {
  // Each `frame` is a Data containing one or more MP3 frames
  send(frame)
}
```

### File Encode

Use `encode(_:to:)` to write directly to a file, including the Xing header for seeking. If encoding fails or is cancelled, the partly written file is removed:

```swift
let encoder = try MP3Encoder(options: options)
try await encoder.encode(sampleSource, to: outputURL)
```

### Encoding a File

`encode(contentsOf:to:progress:)` reads an audio file and writes the MP3, calling `progress` with the fraction done after each chunk. On Apple platforms it reads anything `AVAudioFile` does (WAV, AIFF, CAF, FLAC, M4A, ...); elsewhere it reads uncompressed WAV. The file's sample rate replaces `sampleRate`, and a mono file makes a mono MP3:

```swift
let encoder = try MP3Encoder(options: MP3EncoderOptions(bitrateKbps: 192))
try await encoder.encode(contentsOf: wavURL, to: mp3URL) { fraction in
  print("\(Int(fraction * 100))%")
}
```

The `progress` closure is `@Sendable`; hop to the main actor to update UI from it.

### Command Line

`mp3encode` wraps the same call:

```sh
swift run -c release mp3encode input.wav output.mp3 -b 192 --title "My Song" --artist "Artist"
```

Run `mp3encode --help` for the bitrate, VBR, quality, mode, sample rate, and ID3 options.

### Metadata

Embed ID3v2.3 tags (title, artist, album, artwork, etc.) in your MP3 files:

```swift
let options = MP3EncoderOptions(
  id3Tag: ID3Tag(title: "My Song", artist: "Artist", album: "Album")
)
let encoder = try MP3Encoder(options: options)
try await encoder.encode(source, to: url)
```

For album artwork:

```swift
let artwork = try Data(contentsOf: coverImageURL)
let options = MP3EncoderOptions(
  id3Tag: ID3Tag(
    title: "My Song",
    artist: "Artist",
    albumArt: artwork,
    albumArtMIMEType: "image/png"
  )
)
```

When the encoder writes a file (`encode(_:to:)` or `encode(contentsOf:to:)`), the ID3 tag also gets `REPLAYGAIN_TRACK_GAIN` and `REPLAYGAIN_TRACK_PEAK` frames, for players that read ReplayGain from ID3 rather than the LAME tag. With a session, call `generateID3Tag()` after `flush()` to include them.

### Live Microphone Recording

Feed audio from `AVAudioEngine` into an `AsyncStream` and encode to a file:

```swift
let engine = AVAudioEngine()
let input = engine.inputNode
let format = input.outputFormat(forBus: 0)

let source = AsyncStream<[Float]> { continuation in
  input.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
    let samples = Array(UnsafeBufferPointer(
      start: buffer.floatChannelData?[0],
      count: Int(buffer.frameLength)
    ))
    continuation.yield(samples)
  }
  continuation.onTermination = { _ in
    input.removeTap(onBus: 0)
    engine.stop()
  }
}

try engine.start()
let encoder = try MP3Encoder(options: MP3EncoderOptions(sampleRate: Int(format.sampleRate), mode: .mono))
try await encoder.encode(source, to: recordingURL)
```

### Background Encoding

`MP3Encoder` is `Sendable`, so you can create it on the main actor and hand it off to a background task:

```swift
let encoder = try MP3Encoder(options: options)

Task.detached {
  try await encoder.encode(sampleSource, to: outputURL)
}
```

### Network Streaming

Encode and send frames over a network connection:

```swift
let encoder = try MP3Encoder(options: options)

for try await frame in encoder.encode(sampleSource) {
  try await connection.send(frame)
}
```

## API Reference

### `MP3EncoderOptions`

Configuration for the encoder.

| Property | Type | Default | Description |
|---|---|---|---|
| `sampleRate` | `Int` | `44_100` | Input sample rate in Hz; any positive rate |
| `outputSampleRate` | `Int?` | `nil` | MP3 sample rate; `nil` picks one from `sampleRate`. See [Sample Rates and Bitrates](#sample-rates-and-bitrates) |
| `bitrateKbps` | `Int` | `128` | Bitrate in kbps; the maximum average for VBR |
| `vbr` | `Bool` | `false` | Give each frame only the bits its masking threshold calls for |
| `mode` | `Mode` | `.jointStereo` | Channel mode: `.mono`, `.stereo`, or `.jointStereo` (mid/side per frame) |
| `quality` | `Int` | `5` | Encoding quality 0 (best) to 9 (fastest) |
| `crcProtected` | `Bool` | `false` | Add CRC error detection to frames |
| `original` | `Bool` | `true` | Set the original bit in frame headers |
| `copyright` | `Bool` | `false` | Set the copyright bit in frame headers |
| `id3Tag` | `ID3Tag?` | `nil` | ID3v2.3 metadata to embed at the start of the file |

### `MP3Encoder`

Stateless, `Sendable` encoder that holds configuration. Safe to share across tasks.

| Method | Description |
|---|---|
| `init(options:) throws` | Create an encoder; throws `MP3EncoderError` for unsupported options |
| `newSession() -> EncoderSession` | Create a mutable encoding session for synchronous use |
| `encode(_:) -> AsyncThrowingStream<Data, Error>` | Stream MP3 frames from an `AsyncSequence` of `[Float]` or `[Int16]` |
| `encode(_:to:) async throws` | Encode an `AsyncSequence` of `[Float]`, `[Int16]`, or `AVAudioPCMBuffer` to a file with Xing header (includes ID3 tag if set) |
| `encode(contentsOf:to:progress:) async throws` | Encode an audio file to an MP3 file, reporting progress |

### `EncoderSession`

Mutable encoding session created via `MP3Encoder.newSession()`. Not `Sendable` — use from a single context.

| Method / Property | Description |
|---|---|
| `encode(samples: [Float]) -> Data` | Feed interleaved PCM samples, returns any complete MP3 frames |
| `encode(samples: [Int16]) -> Data` | Feed interleaved 16-bit PCM samples |
| `encode(planar: [[Float]]) -> Data` | Feed one array of samples per channel |
| `encode(buffer: AVAudioPCMBuffer) throws -> Data` | Feed an audio buffer (Apple platforms) |
| `flush() -> Data` | Pads and encodes any remaining buffered samples |
| `generateID3Tag() -> Data` | Generates the ID3v2.3 tag data (empty if no tag configured); after `flush()` it includes ReplayGain |
| `generateXingHeader() -> Data` | Generates the Xing/Info frame with a LAME tag (call after encoding is complete) |
| `encodedFrameCount: Int` | Number of MP3 frames encoded so far |
| `encodedByteCount: Int` | Total bytes of encoded audio data so far |
| `sampleRate: Int` / `outputSampleRate: Int` | Input and MP3 sample rates |
| `replayGain: Double?` | ReplayGain 2.0 track gain of the input so far, in dB |
| `peakAmplitude: Float` | Largest absolute input sample so far |

### `ID3Tag`

Metadata for ID3v2.3 tags, embedded at the start of the MP3 file.

| Property | Type | Default | Description |
|---|---|---|---|
| `title` | `String?` | `nil` | Track title (TIT2) |
| `artist` | `String?` | `nil` | Artist name (TPE1) |
| `album` | `String?` | `nil` | Album name (TALB) |
| `albumArtist` | `String?` | `nil` | Album artist (TPE2) |
| `composer` | `String?` | `nil` | Composer (TCOM) |
| `track` | `UInt16?` | `nil` | Track number (TRCK) |
| `trackTotal` | `UInt16?` | `nil` | Total tracks (TRCK denominator) |
| `disc` | `UInt16?` | `nil` | Disc number (TPOS) |
| `discTotal` | `UInt16?` | `nil` | Total discs (TPOS denominator) |
| `year` | `UInt16?` | `nil` | Release year (TYER) |
| `genre` | `String?` | `nil` | Genre name (TCON) |
| `comment` | `String?` | `nil` | Comment (COMM) |
| `lyrics` | `String?` | `nil` | Unsynchronized lyrics (USLT) |
| `albumArt` | `Data?` | `nil` | Album artwork JPEG or PNG data (APIC) |
| `albumArtMIMEType` | `String` | `"image/jpeg"` | MIME type for album artwork |

### Sample Format

Input samples are interleaved `Float` values in the range `-1.0` to `1.0`. NaN and infinite samples are encoded as silence:

- **Mono**: `[s0, s1, s2, ...]`
- **Stereo / Joint Stereo**: `[L0, R0, L1, R1, L2, R2, ...]`

### Sample Rates and Bitrates

The MP3's sample rate picks the MPEG version, which sets the bitrates available. Input at any other rate is resampled to one of these:

| Version | Sample rates (Hz) | Bitrates (kbps) |
|---|---|---|
| MPEG-1 | 32000, 44100, 48000 | 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 |
| MPEG-2 | 16000, 22050, 24000 | 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160 |
| MPEG-2.5 | 8000, 11025, 12000 | 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160 |

## Quality

Measured with [ViSQOL](https://github.com/google/visqol) 3.3.3 in audio mode (MOS-LQO, 1 to 5, higher is better), averaged over ten 48 kHz stereo music clips from ViSQOL's conformance set, against LAME 3.100 with its default quality. CBR sizes are equal by definition.

| Mode | SwiftMP3 | LAME |
|---|---|---|
| CBR 96 kbps | 4.22 | 4.16 |
| CBR 128 kbps | 4.42 | 4.35 |
| CBR 192 kbps | 4.58 | 4.56 |
| CBR 320 kbps | 4.71 | 4.70 |
| VBR, quality 5 vs. LAME `-V 2` | 4.49 at 2.49 MB | 4.48 at 3.03 MB |
| VBR, quality 9 vs. LAME `-V 4` | 4.43 at 1.92 MB | 4.35 at 2.35 MB |

ViSQOL is a model of listening, not a listening test, but it tracks the artifacts that matter at these bitrates: missing high frequencies and audible quantization noise.

## Documentation

API documentation can be built locally in Xcode with **Product › Build Documentation**.

## Development

```sh
swift build
swift test
swift run -c release SwiftMP3Benchmark   # encode speed on synthetic signals
swift run -c release mp3encode --help    # command-line encoder
```

`SwiftMP3Benchmark --hashes` also prints a hash of each encode, to check that a change leaves the output byte-for-byte the same.

## License

MIT License. See [LICENSE](LICENSE) for details.
